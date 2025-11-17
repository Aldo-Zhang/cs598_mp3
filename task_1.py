import jsonlines
import sys
import torch
import re
import subprocess
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################


def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)


def extract_java_code_from_response(response):
    """Extract Java code from model response between [Java Start] and [Java End] markers."""
    # Try to find code between markers
    pattern = r'\[Java Start\](.*?)\[Java End\]'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no markers, try to find code blocks
    pattern = r'```java\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = r'```\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code.startswith('java') or 'class Solution' in code or 'public' in code:
            return code.replace('java', '', 1).strip()

    return None


def create_vanilla_prompt(prompt_line, python_code):
    """Create vanilla prompt for code translation - matches instruction example."""
    prompt = f"""You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:
Can you translate the following Python code into Java?
The new Java code must be enclosed between [Java Start] and [Java End]

{prompt_line}
{python_code}

### Response:
"""
    return prompt


def create_crafted_prompt(prompt_line, python_code, declaration_python):
    return f"""You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:
Translate the following Python code to Java. Follow the structure shown in the example below.

**Example:**
Python code:
```python
def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    return [x for x in strings if x.startswith(prefix)]
```

Java code:
[Java Start]
import java.util.*;
import java.lang.*;

class Solution {{
    public List<String> filterByPrefix(List<String> strings, String prefix) {{
        List<String> result = new ArrayList<>();
        for (String x : strings) {{
            if (x.startsWith(prefix)) {{
                result.add(x);
            }}
        }}
        return result;
    }}
}}
[Java End]

**Requirements:**
- Put all code in class "Solution"  
- Use instance methods (not static)
- Convert snake_case names to camelCase (first letter lowercase)
- Use List<Type> for parameters (not ArrayList[], not int[], not String[])
- Match Python return types carefully:
  * List → List<Type> (use Integer not Double if values are whole numbers)
  * tuple → List
  * int → int
  * str → String
  * None → use Integer/Long (NOT int/long) or Optional<Type> based on context
- For division: If Python uses / but result values are whole numbers, use integer division and return integer types
- For nullable returns: Use boxed types (Integer, Long, Double) NOT primitives (int, long, double) when null is possible
- Add imports at top: import java.util.*; import java.lang.*;

**Critical Type Conversion Rules:**
1. Python division `/` producing whole numbers → Java integer division and List<Integer>
   Example: `my_list.append(i / 2 + 1)` where result is whole → use `myList.add(i / 2 + 1)` and return `List<Integer>`
   
2. Python None in return tuple → Java Optional<Type>
   Example: `return (max(list) if list else None, ...)` → wrap in Optional:
   ```
   Integer maxVal = list.isEmpty() ? null : Collections.max(list);
   Integer minVal = list2.isEmpty() ? null : Collections.min(list2);
   return Arrays.asList(maxVal != null ? Optional.of(maxVal) : Optional.empty(), 
                        minVal != null ? Optional.of(minVal) : Optional.empty());
   ```

**Python code to translate:**

{declaration_python}

{python_code}

Output ONLY the Java code between [Java Start] and [Java End]. No explanations.

### Response:
"""

class JavaEndStoppingCriteria(StoppingCriteria):
    """Stop generation when [Java End] marker is detected."""
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.end_marker = "[Java End]"
        
    def __call__(self, input_ids, scores, **kwargs):
        # Only check the newly generated tokens (after prompt)
        generated_ids = input_ids[0][self.prompt_length:]
        if len(generated_ids) < 10:  # Need at least a few tokens to check
            return False
        
        # Decode recent tokens to check for end marker
        recent_text = self.tokenizer.decode(generated_ids[-50:], skip_special_tokens=True)
        return self.end_marker in recent_text


def extract_method_with_braces(code):
    """Extract a complete method including nested braces."""
    # Find the start of a method (public/private/protected method)
    method_start = re.search(
        r'(public|private|protected)\s+(static\s+)?\w+\s+\w+\s*\([^)]*\)\s*\{', code)
    if not method_start:
        return None

    start_pos = method_start.start()
    brace_count = 0
    in_method = False
    end_pos = start_pos

    for i in range(start_pos, len(code)):
        if code[i] == '{':
            brace_count += 1
            in_method = True
        elif code[i] == '}':
            brace_count -= 1
            if in_method and brace_count == 0:
                end_pos = i + 1
                break

    if end_pos > start_pos and brace_count == 0:
        return code[start_pos:end_pos]
    return None


def validate_java_code(java_code, test_code, task_id):
    """Compile and run Java code with tests to validate translation."""
    # Create temporary directory for Java files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract the method from java_code and wrap it in Solution class if needed
            if 'class Solution' not in java_code:
                # Check if there's a different class name
                class_match = re.search(r'class\s+(\w+)', java_code)
                if class_match and class_match.group(1) != 'Solution':
                    # Replace class name with Solution
                    java_code = re.sub(
                        r'class\s+\w+', 'class Solution', java_code)

                if 'class Solution' in java_code:
                    # Class name was fixed, just ensure imports
                    if 'import java.util' not in java_code:
                        java_code = 'import java.util.*;\nimport java.util.stream.*;\nimport java.lang.*;\n\n' + java_code
                    elif 'import java.util.stream' not in java_code:
                        # Has java.util but missing stream import
                        java_code = java_code.replace('import java.util.*;', 'import java.util.*;\nimport java.util.stream.*;')
                    # Remove static keyword from methods if present
                    java_code = re.sub(r'public\s+static\s+',
                                       'public ', java_code)
                else:
                    # Try to extract just the method body with proper brace matching
                    method = extract_method_with_braces(java_code)
                    if method:
                        # Remove static keyword if present
                        method = re.sub(r'public\s+static\s+',
                                        'public ', method)
                        java_code = f"""import java.util.*;
import java.util.stream.*;
import java.lang.*;

class Solution {{
{method}
}}"""
                    else:
                        # If we can't find a method, try to wrap the whole thing
                        # Remove static keyword if present
                        java_code_cleaned = re.sub(
                            r'public\s+static\s+', 'public ', java_code)
                        java_code = f"""import java.util.*;
import java.util.stream.*;
import java.lang.*;

class Solution {{
{java_code_cleaned}
}}"""
            else:
                # Ensure imports are present
                if 'import java.util' not in java_code:
                    java_code = 'import java.util.*;\nimport java.util.stream.*;\nimport java.lang.*;\n\n' + java_code
                elif 'import java.util.stream' not in java_code:
                    # Has java.util but missing stream import
                    java_code = java_code.replace('import java.util.*;', 'import java.util.*;\nimport java.util.stream.*;')
                # Also ensure class name is exactly Solution (in case of case mismatch)
                java_code = re.sub(
                    r'class\s+\w+', 'class Solution', java_code, count=1)
                # Remove static keyword from methods if present (test code uses instance methods)
                java_code = re.sub(r'public\s+static\s+', 'public ', java_code)

            # Write Solution class
            solution_file = os.path.join(temp_dir, "Solution.java")
            save_file(java_code, solution_file)
            print(f"Generated java code:\n{java_code}")

            # Write test file
            test_file = os.path.join(temp_dir, "Main.java")
            save_file(test_code, test_file)
            print(f"Generated java test code:\n{test_code}")

            # Compile Solution.java
            compile_result = subprocess.run(
                ["javac", solution_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=10
            )

            if compile_result.returncode != 0:
                print(f"Compilation error for {task_id}: {compile_result.stderr}")
                return False, "compilation_error"

            # Compile Main.java
            compile_result = subprocess.run(
                ["javac", test_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=10
            )

            if compile_result.returncode != 0:
                print(f"Test compilation error for {task_id}: {compile_result.stderr}")
                return False, "compilation_error"

            # Run the test
            run_result = subprocess.run(
                ["java", "-cp", temp_dir, "Main"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=10
            )

            if run_result.returncode == 0:
                return True, "success"
            else:
                print(f"Test execution failed for {
                      task_id}: {run_result.stderr}")
                return False, "test_failure"

        except subprocess.TimeoutExpired:
            print(f"Timeout for {task_id}")
            return False, "timeout"
        except Exception as e:
            print(f"Error validating {task_id}: {str(e)}")
            return False, "error"


def prompt_model(dataset, model_name="deepseek-ai/deepseek-coder-6.7b-instruct", vanilla=True):
    print(f"Working with {model_name} prompt type {vanilla}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config
    )
    model.config.max_length = 2048

    # Load Java dataset for test code and declarations
    # Try to find Java dataset file in the same directory as Python dataset
    java_dataset = None
    python_file = sys.argv[1] if len(sys.argv) > 1 else None
    base_dir = os.path.dirname(
        python_file) if os.path.dirname(python_file) else '.'
    base_name = os.path.basename(python_file)

    # Try replacing python with java in filename
    java_file = base_name.replace('python', 'java').replace('Python', 'Java')
    java_file_path = os.path.join(base_dir, java_file)

    try:
        java_dataset = read_jsonl(java_file_path)
        print(f"Successfully loaded Java dataset from {java_file_path}")
    except FileNotFoundError:
        print(f"ERROR: Java dataset file not found at {java_file_path}")
        print("Please ensure the Java dataset is generated before running this script.")
        sys.exit(1)

    results = []
    for entry in dataset:
        task_id = entry['task_id']
        # Extract numeric ID (e.g., "0" from "HumanEval/0" or "Python/0")
        numeric_id = task_id.split('/')[-1]

        # Create prompt based on vanilla or crafted
        prompt_line = entry['prompt'].replace(
            "'''", "\"\"\"").split('\"\"\"')[0].strip()
        python_code = entry.get('canonical_solution', '')
        declaration_python = entry['declaration'].replace(
            "'''", "\"\"\"").split('\"\"\"')[0].strip()

        if vanilla:
            prompt = create_vanilla_prompt(prompt_line, python_code)
        else:
            prompt = create_crafted_prompt(prompt_line, python_code, declaration_python)

        # Prompt the model
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Create stopping criteria to stop when [Java End] is detected
        prompt_length = inputs['input_ids'].shape[1]
        stopping_criteria = StoppingCriteriaList([
            JavaEndStoppingCriteria(tokenizer, prompt_length)
        ])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )

        # Decode response (only the generated part)
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Extract Java code from response
        java_code = extract_java_code_from_response(response)

        # Validate the Java code
        verdict = False
        validation_status = "not_attempted"

        if java_code:
            if java_dataset:
                # Find corresponding Java entry
                java_entry = None
                for je in java_dataset:
                    if je['task_id'].split('/')[-1] == numeric_id:
                        java_entry = je
                        break

                if java_entry:
                    test_code = f'''import java.util.*;
{java_entry.get('test', '')}
'''
                    verdict, validation_status = validate_java_code(
                        java_code, test_code, task_id)
                else:
                    print(
                        f"Warning: Could not find corresponding Java entry for {task_id}")
                    validation_status = "no_java_entry"
            else:
                print(
                    f"Warning: Java dataset not loaded, cannot validate {task_id}")
                validation_status = "no_java_dataset"
        else:
            print(
                f"Warning: Could not extract Java code from response for {task_id}")
            validation_status = "extraction_failed"

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{
              prompt}\nresponse:\n{response}\nis_correct:\n{verdict}\nvalidation_status:\n{validation_status}")
        results.append({
            "task_id": task_id,
            "prompt": prompt,
            "response": response,
            "is_correct": verdict,
            "validation_status": validation_status
        })

    # Print summary
    print("\n" + "=" * 80)
    print("TRANSLATION SUMMARY")
    print("=" * 80)
    
    passed = [r for r in results if r['is_correct']]
    failed = [r for r in results if not r['is_correct']]
    
    print(f"Total problems: {len(results)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(passed)/len(results)*100:.1f}%")
    
    if passed:
        print(f"\n✓ Passed task_ids ({len(passed)}):")
        for r in passed:
            print(f"  - {r['task_id']}")
    
    if failed:
        print(f"\n✗ Failed task_ids ({len(failed)}):")
        for r in failed:
            status = r.get('validation_status', 'unknown')
            print(f"  - {r['task_id']} (status: {status})")
    
    print("=" * 80 + "\n")
    
    return results


def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            dataset.append(line)
    return dataset


def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])


if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code translation.
    Usage:
    `python3 task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt

    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3]  # True or False

    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")

    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")

    vanilla = True if if_vanilla == "True" else False

    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
