# source ~/anaconda3/bin/activate pytorch

cd ~/CS598JBR-Team-5/MP3

bash -x setup_dataset.sh
bash -x setup_models.sh

seed="54789384577748203278933559465614931523"

input_python_dataset="selected_humanevalx_python_${seed}.jsonl"
task_1_crafted_jsonl="task_1_${seed}_crafted.jsonl"

python3.12 task_1.py \
  "${input_python_dataset}" \
  "deepseek-ai/deepseek-coder-6.7b-instruct" \
  "${task_1_crafted_jsonl}" \
  "False" | tee task_1_crafted.log