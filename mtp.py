%%capture
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

!pip install pinecone
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import uuid
from typing import List, Tuple
import time


try:
    import pinecone
    import unsloth
    from datasets import load_dataset
    from transformers import TrainingArguments, BitsAndBytesConfig
    from trl import SFTTrainer
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing required packages...")
    !pip install -q pinecone-client datasets transformers trl sentence-transformers
    !pip install -q unsloth
    !pip uninstall -y unsloth && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

    import pinecone
    from datasets import load_dataset
    from transformers import TrainingArguments, BitsAndBytesConfig
    from trl import SFTTrainer
    from sentence_transformers import SentenceTransformer
    from unsloth import FastLanguageModel, is_bfloat16_supported



from unsloth import FastLanguageModel, is_bfloat16_supported


from pinecone import Pinecone, ServerlessSpec


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = SentenceTransformer(embedding_model_name, device=device)



try:

    pc = Pinecone(api_key="pcsk_5qetgB_EtLAKPcLHP3m5K4kFmbCvJBpmwu8kKzHLJNoi2DkQEx5BYrLYpeUpWmvpCH4aj6")
except Exception as e:
    print(f"Failed to initialize Pinecone. Please check your API key and environment: {e}")
    pc = None

index_name = "text-hyperparams"
index = None

if pc:
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Ensure this is a valid region for your Pinecone setup
            )
        )
    index = pc.Index(index_name)
    print(f"Pinecone index '{index_name}' connected.")
else:
    print("Pinecone client not initialized. Database operations will be skipped.")


def get_embedding(text: str) -> np.ndarray:
    embedding_vector = embedding_model.encode(text, convert_to_numpy=True)
    return embedding_vector

def store_text_with_hyperparams(text: str, hyperparams: List[float], loss: float, text_id: str = None):
    if not index:
        print("Skipping store_text_with_hyperparams: Pinecone index not available.")
        return None
    if text_id is None:
        text_id = f"text_{uuid.uuid4()}"
    embedding_vector = get_embedding(text)
    hyperparams_str = [str(param) for param in hyperparams]
    try:
        index.upsert(vectors=[
            {
                "id": text_id,
                "values": embedding_vector.tolist(),
                "metadata": {
                    "hyperparams": hyperparams_str,
                    "loss": str(loss),
                    "original_text": text[:1000] # Store a snippet
                }
            }
        ])
        return text_id
    except Exception as e:
        print(f"Error storing text in Pinecone: {e}")
        return None

def find_similar_texts(query_text: str, top_k: int = 10) -> List[Tuple[List[float], float]]:
    if not index:
        print("Skipping find_similar_texts: Pinecone index not available.")
        return []
    query_words = query_text.split()[:1000] # Truncate query if too long for embedding model
    query_text_truncated = " ".join(query_words)
    query_embedding_vector = get_embedding(query_text_truncated)
    try:
        results = index.query(
            vector=query_embedding_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []
    similar_items = []
    for match in results.matches:
        if "hyperparams" in match.metadata and isinstance(match.metadata["hyperparams"], list):
            try:
                hyperparams_str = match.metadata["hyperparams"]
                hyperparams = [float(param) for param in hyperparams_str]
                similarity = match.score
                similar_items.append((hyperparams, similarity))
            except ValueError as ve:
                print(f"Warning: Could not parse hyperparams from metadata for match ID {match.id}: {ve}")
            except Exception as e_meta:
                 print(f"Warning: Error processing metadata for match ID {match.id}: {e_meta}")
    return similar_items

INT_PARAM_INDICES = [2, 3, 4] # warmup_steps, batch_size, grad_accum_steps

def initialize_population_with_db(population_size, num_variables, input_text_for_query, bounds):
    population_list = []
    db_count_target = population_size // 2

    if index and db_count_target > 0:
        print(f"Initializing population: Attempting to fetch up to {db_count_target} individuals from DB based on similarity to input text.")
        similar_hyperparams_from_db = find_similar_texts(input_text_for_query, top_k=db_count_target)

        for hyperparams, _ in similar_hyperparams_from_db:
            if len(hyperparams) == num_variables:
                processed_hyperparams = []
                for i, param_val in enumerate(hyperparams):
                    b_min, b_max = bounds[i]
                    val = max(b_min, min(b_max, param_val))
                    if i in INT_PARAM_INDICES:
                        val = int(round(val))
                        val = max(int(b_min), min(int(b_max), val))
                    processed_hyperparams.append(val)
                population_list.append(processed_hyperparams)
            if len(population_list) >= db_count_target:
                break
        print(f"Initialized {len(population_list)} individuals from DB.")
    elif not index:
        print("Pinecone index not available, skipping DB initialization for population part.")
    elif db_count_target == 0:
         print("Target for DB individuals is 0, generating fully random population.")

    remaining_to_generate = population_size - len(population_list)
    if remaining_to_generate > 0:
        print(f"Initializing population: Generating {remaining_to_generate} random individuals.")
        for _ in range(remaining_to_generate):
            individual = []
            for i, b in enumerate(bounds):
                b_min, b_max = b
                if i in INT_PARAM_INDICES:
                    val = np.random.randint(int(b_min), int(b_max) + 1)
                else:
                    val = np.random.uniform(b_min, b_max)
                individual.append(val)
            population_list.append(individual)

    if not population_list and population_size > 0:
        print("Warning: Population initialization resulted in an empty list. Generating fully random population as fallback.")
        for _ in range(population_size):
            individual = []
            for i, b in enumerate(bounds):
                b_min, b_max = b
                if i in INT_PARAM_INDICES:
                    val = np.random.randint(int(b_min), int(b_max) + 1)
                else:
                    val = np.random.uniform(b_min, b_max)
                individual.append(val)
            population_list.append(individual)

    if population_list:
        random.shuffle(population_list)
    return np.array(population_list)

def prepare_and_format_dataset(tokenizer, first_n_rows=100):
    print(f"Preparing and formatting dataset using tokenizer: {tokenizer.name_or_path} for {first_n_rows} rows...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train", streaming=False)
    if first_n_rows is not None and first_n_rows > 0:
        dataset = dataset.select(range(min(first_n_rows, len(dataset))))

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
    EOS_TOKEN = tokenizer.eos_token
    if EOS_TOKEN is None:
        print("Warning: tokenizer.eos_token is None. Using '</s>' as a fallback.")
        EOS_TOKEN = "</s>"

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples.get("input", ["" for _ in range(len(instructions))])
        outputs = examples.get("output", ["" for _ in range(len(instructions))])
        texts = []
        for i in range(len(instructions)):
            instruction = instructions[i] if instructions[i] is not None else ""
            input_text = inputs[i] if i < len(inputs) and inputs[i] is not None else ""
            output_text = outputs[i] if i < len(outputs) and outputs[i] is not None else ""
            text = alpaca_prompt.format(str(instruction), str(input_text), str(output_text)) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    formatted_dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=min(4, torch.multiprocessing.cpu_count())
    )
    print(f"Dataset prepared and formatted. Number of examples: {len(formatted_dataset)}")
    if len(formatted_dataset) == 0:
        print("CRITICAL WARNING: Formatted dataset is empty!")
    return formatted_dataset

def train_model(
    formatted_train_dataset,
    llm_base_model,
    llm_tokenizer,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=5,
    batch_size=2,
    grad_accum_steps=4,
    max_steps_train=5
):
    max_seq_length = 2048
    peft_model_for_training = None
    trainer = None

    try:
        peft_model_for_training = FastLanguageModel.get_peft_model(
            llm_base_model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407, # Fixed random state for reproducibility
            use_rslora=False,
            loftq_config=None,
        )

        batch_size_int = int(max(1, batch_size))
        grad_accum_steps_int = int(max(1, grad_accum_steps))
        warmup_steps_int = int(max(1, warmup_steps))
        max_steps_int = int(max(1, max_steps_train))

        trainer = SFTTrainer(
            model=peft_model_for_training,
            tokenizer=llm_tokenizer,
            train_dataset=formatted_train_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=min(2, torch.multiprocessing.cpu_count()),
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size_int,
                gradient_accumulation_steps=grad_accum_steps_int,
                warmup_steps=warmup_steps_int,
                max_steps=max_steps_int,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=weight_decay,
                lr_scheduler_type="linear",
                seed=3407, # Fixed seed for SFTTrainer
                output_dir="outputs",
                report_to="none",
            ),
        )

        trainer_stats = trainer.train()
        loss = 10.0
        if trainer.state.log_history:
            last_log = trainer.state.log_history[-1]
            if 'loss' in last_log:
                loss = last_log['loss']
            elif 'train_loss' in last_log:
                loss = last_log['train_loss']
            elif hasattr(trainer_stats, 'training_loss') and trainer_stats.training_loss is not None:
                loss = trainer_stats.training_loss
        elif hasattr(trainer_stats, 'training_loss') and trainer_stats.training_loss is not None:
             loss = trainer_stats.training_loss

        return loss

    except Exception as e:
        print(f"Error in training model: {e}")
        import traceback
        traceback.print_exc()
        return 10.0
    finally:
        if trainer is not None and hasattr(trainer, 'model') and trainer.model is not None:
            del trainer.model
        del trainer
        del peft_model_for_training

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def fitness_function(hyperparams, text_to_store_in_db: str, formatted_train_dataset_for_ga, llm_base_model, llm_tokenizer, target_loss=0.5, max_steps_train=5):
    learning_rate, weight_decay, warmup_steps_float, batch_size_float, grad_accum_steps_float = hyperparams

    warmup_steps_int = int(round(max(1, warmup_steps_float)))
    batch_size_int = int(round(max(1, batch_size_float)))
    grad_accum_steps_int = int(round(max(1, grad_accum_steps_float)))

    loss = train_model(
        formatted_train_dataset=formatted_train_dataset_for_ga,
        llm_base_model=llm_base_model,
        llm_tokenizer=llm_tokenizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps_int,
        batch_size=batch_size_int,
        grad_accum_steps=grad_accum_steps_int,
        max_steps_train=max_steps_train
    )
    fitness = abs(loss - target_loss)
    print(f"Hyperparams (raw from GA): {hyperparams}, Parsed Ints for train: ws={warmup_steps_int}, bs={batch_size_int}, ga={grad_accum_steps_int}, Loss: {loss:.4f}, Fitness: {fitness:.4f}")

    store_text_with_hyperparams(
        text=text_to_store_in_db,
        hyperparams=hyperparams.tolist() if isinstance(hyperparams, np.ndarray) else hyperparams,
        loss=loss
    )
    return fitness, loss

def evaluate_population(population, text_to_store_in_db: str, formatted_train_dataset_for_ga, llm_base_model, llm_tokenizer, max_steps_train=5):
    fitness_scores = []
    actual_losses = []

    for i, individual_hyperparams in enumerate(population):
        print(f"Evaluating individual {i+1}/{len(population)}...")
        fitness, loss = fitness_function(
            individual_hyperparams,
            text_to_store_in_db,
            formatted_train_dataset_for_ga,
            llm_base_model,
            llm_tokenizer,
            max_steps_train=max_steps_train
        )
        fitness_scores.append(fitness)
        actual_losses.append(loss)
    return np.array(fitness_scores), np.array(actual_losses)

def tournament_selection(population, fitness_scores, num_selected, tournament_size=3):
    selected_indices = []
    population_indices = list(range(len(population)))
    if not population_indices : return np.array([])
    for _ in range(num_selected):
        current_tournament_size = min(tournament_size, len(population_indices))
        if current_tournament_size == 0: break
        aspirants_indices_from_pop = random.sample(population_indices, current_tournament_size)
        best_aspirant_overall_idx = aspirants_indices_from_pop[0]
        min_fitness_in_tournament = fitness_scores[best_aspirant_overall_idx]
        for aspirant_overall_idx in aspirants_indices_from_pop[1:]:
            if fitness_scores[aspirant_overall_idx] < min_fitness_in_tournament:
                min_fitness_in_tournament = fitness_scores[aspirant_overall_idx]
                best_aspirant_overall_idx = aspirant_overall_idx
        selected_indices.append(best_aspirant_overall_idx)
    return population[selected_indices]

def crossover(parents):
    offspring = []
    num_parents = len(parents)
    if num_parents < 2:
        return parents.copy() if num_parents == 1 else np.array([])
    shuffled_parent_indices = np.random.permutation(num_parents)
    for i in range(0, num_parents -1 , 2):
        idx1, idx2 = shuffled_parent_indices[i], shuffled_parent_indices[i+1]
        parent1, parent2 = parents[idx1], parents[idx2]
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        for k_idx in INT_PARAM_INDICES:
            child1[k_idx] = int(round(child1[k_idx]))
            child2[k_idx] = int(round(child2[k_idx]))
        offspring.extend([child1, child2])
    return np.array(offspring) if offspring else np.array([])

def mutation(offspring, bounds, mutation_rate=0.1):
    if offspring.size == 0: return offspring
    mutated_offspring = offspring.copy()
    num_variables = offspring.shape[1]
    for child_idx in range(len(mutated_offspring)):
        if np.random.rand() < mutation_rate:
            param_idx_to_mutate = random.randint(0, num_variables - 1)
            bound_min, bound_max = bounds[param_idx_to_mutate]
            mutation_strength = (bound_max - bound_min) * 0.1
            mutation_val = np.random.normal(0, mutation_strength)

            mutated_offspring[child_idx][param_idx_to_mutate] += mutation_val

            if param_idx_to_mutate in INT_PARAM_INDICES:
                 mutated_offspring[child_idx][param_idx_to_mutate] = int(round(mutated_offspring[child_idx][param_idx_to_mutate]))
                 mutated_offspring[child_idx][param_idx_to_mutate] = np.clip(
                    mutated_offspring[child_idx][param_idx_to_mutate], int(bound_min), int(bound_max)
                )
            else:
                mutated_offspring[child_idx][param_idx_to_mutate] = np.clip(
                    mutated_offspring[child_idx][param_idx_to_mutate], bound_min, bound_max
                )
    return mutated_offspring

def elitism(population, fitness_scores, elitism_count):
    if elitism_count == 0 or len(population) == 0:
        return np.array([])
    elite_indices = np.argsort(fitness_scores)[:elitism_count]
    return population[elite_indices].copy()

def genetic_algorithm_optimization(
    text_for_ga_tasks: str,
    llm_base_model,
    llm_tokenizer,
    population_size=10,
    num_generations=5,
    mutation_rate=0.2,
    elitism_count=1,
    max_steps_per_eval=5,
    dataset_rows_per_eval=10
):
    bounds = [
        (1e-5, 5e-4),
        (0.001, 0.1),
        (1, 10),
        (1, 4),
        (1, 8)
    ]
    num_variables = len(bounds)

    print("Preparing shared dataset for all GA evaluations...")
    formatted_dataset_for_ga = prepare_and_format_dataset(
        tokenizer=llm_tokenizer,
        first_n_rows=dataset_rows_per_eval
    )
    if not formatted_dataset_for_ga or len(formatted_dataset_for_ga) == 0:
        print("CRITICAL: Dataset preparation for GA failed or resulted in an empty dataset. Aborting.")
        return None, float('inf')

    population = initialize_population_with_db(population_size, num_variables, text_for_ga_tasks, bounds)
    if not isinstance(population, np.ndarray) or population.shape[0] != population_size:
        print(f"Population initialization issue. Expected {population_size}, got {population.shape[0] if isinstance(population, np.ndarray) else 0}. Re-initializing randomly.")
        pop_list = []
        for _ in range(population_size):
            indiv = []
            for i, b_bounds_val in enumerate(bounds):
                b_min, b_max = b_bounds_val
                if i in INT_PARAM_INDICES:
                    val = np.random.randint(int(b_min), int(b_max) + 1)
                else:
                    val = np.random.uniform(b_min, b_max)
                indiv.append(val)
            pop_list.append(indiv)
        population = np.array(pop_list)
        if population.shape[0] != population_size:
             print("Critical: Random population initialization also failed to create correct size.")
             return None, float('inf')

    historical_best_fitness = []
    overall_best_fitness = float('inf')
    overall_best_hyperparams = None
    overall_best_loss = float('inf')

    for generation in range(num_generations):
        print(f"\n===== Generation {generation + 1}/{num_generations} =====")

        current_gen_fitness_scores, current_gen_actual_losses = evaluate_population(
            population,
            text_for_ga_tasks,
            formatted_dataset_for_ga,
            llm_base_model,
            llm_tokenizer,
            max_steps_train=max_steps_per_eval
        )

        if len(current_gen_fitness_scores) > 0:
            best_idx_in_gen = np.argmin(current_gen_fitness_scores)
            current_gen_best_fitness_val = current_gen_fitness_scores[best_idx_in_gen]
            current_gen_best_loss_val = current_gen_actual_losses[best_idx_in_gen]

            if current_gen_best_fitness_val < overall_best_fitness:
                overall_best_fitness = current_gen_best_fitness_val
                overall_best_hyperparams = population[best_idx_in_gen].copy()
                overall_best_loss = current_gen_best_loss_val
                print(f"🎉 New Overall Best! Fitness: {overall_best_fitness:.4f}, Loss: {overall_best_loss:.4f}, Hyperparams: {overall_best_hyperparams}")
        else:
            print("Warning: No fitness scores returned for the current generation. Skipping generation update.")
            if len(population) == 0: print("Critical: Population became empty."); break
            current_gen_best_fitness_val = float('inf')

        historical_best_fitness.append(overall_best_fitness)
        print(f"Generation {generation + 1}: Best Fitness in Gen = {current_gen_best_fitness_val:.4f}, Overall Best Fitness = {overall_best_fitness:.4f}, Overall Best Loss: {overall_best_loss:.4f}")

        if generation == num_generations - 1: break

        elite_individuals = elitism(population, current_gen_fitness_scores, elitism_count)
        num_offspring_needed = population_size - len(elite_individuals)
        next_gen_population_list = list(elite_individuals)

        if num_offspring_needed > 0:
            num_parents_to_select = num_offspring_needed if num_offspring_needed % 2 == 0 else num_offspring_needed + 1
            num_parents_to_select = min(num_parents_to_select, len(population))

            parents = np.array([])
            if num_parents_to_select > 0 and len(population) > 0 and len(current_gen_fitness_scores) > 0 :
                 parents = tournament_selection(population, current_gen_fitness_scores, num_parents_to_select)

            offspring_from_crossover = np.array([])
            if len(parents) >=2:
                offspring_from_crossover = crossover(parents)

            final_offspring_list = list(offspring_from_crossover)
            idx = 0
            while len(final_offspring_list) < num_offspring_needed:
                if len(parents) > 0:
                    parent_to_add_raw = parents[idx % len(parents)].copy()
                    for k_idx_p_add in INT_PARAM_INDICES:
                        parent_to_add_raw[k_idx_p_add] = int(round(parent_to_add_raw[k_idx_p_add]))
                    final_offspring_list.append(parent_to_add_raw)
                    idx +=1
                else:
                    rand_indiv_fill = []
                    for i_fill, b_fill in enumerate(bounds):
                        val_fill = np.random.randint(int(b_fill[0]), int(b_fill[1])+1) if i_fill in INT_PARAM_INDICES else np.random.uniform(b_fill[0], b_fill[1])
                        rand_indiv_fill.append(val_fill)
                    final_offspring_list.append(np.array(rand_indiv_fill))

            actual_offspring_for_mutation = np.array(final_offspring_list[:num_offspring_needed])
            if actual_offspring_for_mutation.size > 0:
                 mutated_offspring = mutation(actual_offspring_for_mutation, bounds, mutation_rate)
                 next_gen_population_list.extend(list(mutated_offspring))

        population = np.array(next_gen_population_list)
        if len(population) < population_size:
            fill_count = population_size - len(population)
            print(f"Repopulating: Adding {fill_count} random individuals to maintain population size.")
            for _ in range(fill_count):
                rand_indiv_fill_pop = []
                for i_fill_pop, b_fill_pop in enumerate(bounds):
                    val_fill_pop = np.random.randint(int(b_fill_pop[0]), int(b_fill_pop[1])+1) if i_fill_pop in INT_PARAM_INDICES else np.random.uniform(b_fill_pop[0], b_fill_pop[1])
                    rand_indiv_fill_pop.append(val_fill_pop)

                if population.size > 0:
                    population = np.vstack([population, np.array(rand_indiv_fill_pop)])
                else:
                    population = np.array([rand_indiv_fill_pop])
        elif len(population) > population_size:
            population = population[:population_size]

    if historical_best_fitness and plt:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(historical_best_fitness) + 1), historical_best_fitness, marker='o', linestyle='-')
            plt.xlabel("Generation")
            plt.ylabel("Overall Best Fitness (lower is better)")
            plt.title("Fitness Trend Over Generations")
            plt.grid(True)
            plt.xticks(range(1, len(historical_best_fitness) + 1))
            plt.savefig("fitness_trend.png")
            print("Fitness trend plot saved to fitness_trend.png")
        except Exception as e:
            print(f"Could not plot fitness trend: {e}")
    else:
        print("No fitness data to plot or matplotlib (plt) not available.")

    print("\n===== Final Results =====")
    if overall_best_hyperparams is not None:
        print(f"Best Hyperparameters Found: {overall_best_hyperparams}")
        print(f"Corresponding Best Training Loss: {overall_best_loss:.4f}")
        print(f"Corresponding Best Fitness (abs(loss - target_loss)): {overall_best_fitness:.4f}")
    else:
        print("No best hyperparameters were found during the optimization process.")
    return overall_best_hyperparams, overall_best_loss

def main():
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True


    ga_population_size_usr = 200
    ga_num_generations_usr = 3
    ga_max_steps_per_eval_usr = 5
    ga_dataset_rows_per_eval_usr = 10



    warmup_specific_lr = 1.85632449e-04
    warmup_specific_wd = 7.30033885e-02
    warmup_specific_ws = 8
    warmup_specific_bs = 3
    warmup_specific_ga = 7


    if torch.cuda.is_available():
        print(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Initial GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    llm_base_model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    print(f"Loading base model: {llm_base_model_name}...")
    llm_base_model_original, llm_tokenizer = FastLanguageModel.from_pretrained(
        model_name=llm_base_model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto"
    )
    print("Base model and tokenizer loaded.")

    if torch.cuda.is_available():
        print(f"GPU Memory Allocated after base model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved after base model load: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    print("\nStarting ULTRA-SPECIFIC JIT warm-up training run...")
    try:

        warmup_dataset_raw = load_dataset("yahma/alpaca-cleaned", split="train", streaming=False).select(range(ga_dataset_rows_per_eval_usr))

        alpaca_prompt_warmup = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
        EOS_TOKEN_WARMUP = llm_tokenizer.eos_token if llm_tokenizer.eos_token else "</s>"

        def format_warmup_examples(examples):
            instructions = examples["instruction"]
            inputs = examples.get("input", ["" for _ in range(len(instructions))])
            outputs = examples.get("output", ["" for _ in range(len(instructions))])
            texts = []
            for i_warmup in range(len(instructions)):
                instr = instructions[i_warmup] if instructions[i_warmup] is not None else ""
                inp = inputs[i_warmup] if i_warmup < len(inputs) and inputs[i_warmup] is not None else ""
                outp = outputs[i_warmup] if i_warmup < len(outputs) and outputs[i_warmup] is not None else ""
                text = alpaca_prompt_warmup.format(str(instr), str(inp), str(outp)) + EOS_TOKEN_WARMUP
                texts.append(text)
            return {"text": texts}

        warmup_dataset_formatted = warmup_dataset_raw.map(format_warmup_examples, batched=True, num_proc=1)

        if not warmup_dataset_formatted or len(warmup_dataset_formatted) == 0:
            print("Warning: Warm-up dataset is empty. Skipping warm-up.")
        else:
            print("Applying full PEFT configuration for warm-up run (matching GA)...")
            warmup_peft_model = FastLanguageModel.get_peft_model(
                llm_base_model_original,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha=16, lora_dropout=0, bias="none",
                use_gradient_checkpointing="unsloth", random_state=42, # Different PEFT random_state for warmup
                use_rslora=False, loftq_config=None,
            )
            warmup_trainer = SFTTrainer(
                model=warmup_peft_model,
                tokenizer=llm_tokenizer, train_dataset=warmup_dataset_formatted,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                dataset_num_proc=1, packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=warmup_specific_bs,
                    gradient_accumulation_steps=warmup_specific_ga,
                    warmup_steps=warmup_specific_ws,
                    max_steps=ga_max_steps_per_eval_usr,
                    learning_rate=warmup_specific_lr,
                    weight_decay=warmup_specific_wd,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=1, optim="adamw_8bit",
                    lr_scheduler_type="linear",
                    seed=42,
                    output_dir="outputs_warmup", report_to="none",
                ),
            )
            print(f"Executing warm-up training step with specific params (bs={warmup_specific_bs}, ga={warmup_specific_ga}, ws={warmup_specific_ws}, lr={warmup_specific_lr:.2e}, wd={warmup_specific_wd:.2e}) and structure (rows={ga_dataset_rows_per_eval_usr}, steps={ga_max_steps_per_eval_usr})...")
            warmup_trainer.train()
            print("Warm-up training step completed.")

            del warmup_trainer.model
            del warmup_trainer
            del warmup_peft_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"GPU Memory after warm-up: Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    except Exception as e_warmup:
        print(f"Error during JIT warm-up: {e_warmup}")
        import traceback
        traceback.print_exc()
        print("Continuing without successful warm-up...")


    temp_dataset_for_snippet = load_dataset("yahma/alpaca-cleaned", split="train", streaming=False)
    all_text_content = ""
    for i in range(min(10, len(temp_dataset_for_snippet))):
        instruction = temp_dataset_for_snippet[i].get("instruction", "")
        input_val = temp_dataset_for_snippet[i].get("input", "")
        output_val = temp_dataset_for_snippet[i].get("output", "")
        all_text_content += f"Instruction: {instruction} Input: {input_val} Output: {output_val}\n"

    representative_text_snippet = " ".join(all_text_content.split()[:100])
    print(f"Representative text snippet for GA tasks (first 200 chars for display):\n'{representative_text_snippet[:200]}...'")

    print("\nStarting genetic algorithm optimization for LLM fine-tuning...")
    best_hyperparams, best_loss = genetic_algorithm_optimization(
        text_for_ga_tasks=representative_text_snippet,
        llm_base_model=llm_base_model_original,
        llm_tokenizer=llm_tokenizer,
        population_size=ga_population_size_usr,
        num_generations=ga_num_generations_usr,
        mutation_rate=0.2,
        elitism_count=1,
        max_steps_per_eval=ga_max_steps_per_eval_usr,
        dataset_rows_per_eval=ga_dataset_rows_per_eval_usr
    )

    print("\nOptimization complete!")
    if best_hyperparams is not None:
        print(f"Overall Best hyperparameters found by GA: {best_hyperparams}")
        print(f"Overall Best training loss achieved with these hyperparams: {best_loss:.4f}")
        final_text_id = store_text_with_hyperparams(
            text=representative_text_snippet,
            hyperparams=best_hyperparams.tolist() if isinstance(best_hyperparams, np.ndarray) else best_hyperparams,
            loss=best_loss,
            text_id=f"ga_overall_best_{uuid.uuid4()}"
        )
        if final_text_id:
            print(f"Overall best result stored in Pinecone with ID: {final_text_id}")
        else:
            print("Failed to store overall best result in Pinecone (DB unavailable or error).")
    else:
        print("Genetic algorithm did not yield a best set of hyperparameters.")

    del llm_base_model_original
    del llm_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory Allocated after main cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved after main cleanup: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()
