#!/usr/bin/env bash


#export OPENAI_API_BASE="https://gptproxy.llmpaas.woa.com/v1"
export OPENAI_API_KEY="empty"
export OPENAI_ORGANIZATION=""  # personal

retriever_name="facebook/contriever"
llm_model='gpt-4o-mini-2024-07-18'   # 'meta-llama/Meta-Llama-3.1-8B-Instruct'   # 'gpt-35-turbo-1106'
available_gpus=0
syn_thresh=0.8 # float, e.g., 0.8
llm_api=openai
extraction_type=ner

haystack_config="115k"    # 115k 500sess
subset_type='hipporag-l3.1-8b-instruct/session/${haystack_config}_hipporag_trace'

# for qid in `ls data/long-mem-benchmark/${subset_type} | head -n 10`; do
for qid in `ls data/long-mem-benchmark/${subset_type}`; do
    
    data=long-mem-benchmark/${subset_type}/${qid}/${qid}

    mkdir -p output/openie_long-mem-benchmark/${subset_type}/${qid}/
    mkdir -p output/ircot/ircot_results_long-mem-benchmark/${subset_type}/${qid}/
    mkdir -p output/long-mem-benchmark/${subset_type}/${qid}/ 
    
    python3 src/openie_with_retrieval_option_parallel.py --dataset $data --llm $llm_api --model_name $llm_model --run_ner --num_passages all --num_processes 1
    python3 src/named_entity_extraction_parallel.py --dataset $data --llm $llm_api --model_name $llm_model --num_processes 1  # NER for queries

    python3 src/create_graph.py --dataset $data --model_name $retriever_name --extraction_model $llm_model --threshold $syn_thresh --extraction_type $extraction_type --cosine_sim_edges

    CUDA_VISIBLE_DEVICES=$available_gpus python3 src/RetrievalModule.py --retriever_name ${retriever_name} --string_filename output/query_to_kb.tsv
    CUDA_VISIBLE_DEVICES=$available_gpus python3 src/RetrievalModule.py --retriever_name ${retriever_name} --string_filename output/kb_to_kb.tsv
    CUDA_VISIBLE_DEVICES=$available_gpus python3 src/RetrievalModule.py --retriever_name ${retriever_name} --string_filename output/rel_kb_to_kb.tsv

    python3 src/create_graph.py --dataset $data --model_name ${retriever_name} --extraction_model $llm_model --threshold $syn_thresh --create_graph --extraction_type $extraction_type --cosine_sim_edges

    python3 src/ircot_hipporag.py --dataset long-mem-benchmark/${subset_type}/${qid}/${qid} --retriever ${retriever_name} --llm openai --llm_model ${llm_model} --max_steps 1 --doc_ensemble f --top_k 10 --sim_threshold 0.8 --damping 0.5
    
done

