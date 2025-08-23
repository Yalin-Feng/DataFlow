import json
import os


def convert_to_alpaca(input_file, output_dir="../data"):
    """转换为Alpaca格式并创建LlamaFactory配置"""
    results = []

    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 学术论文专用instruction
    instruction = (
        "Please answer the following question based on the provided academic literature. "
        "Your response should:\n"
        "1. Provide accurate information from the source material\n"
        "2. Include relevant scientific reasoning and methodology\n"
        "3. Reference specific findings, data, or conclusions when applicable\n"
        "4. Maintain academic rigor and precision in your explanation\n\n"
        "Focus on delivering factual, evidence-based answers suitable for academic research."
    )

    # 处理每个QA对
    for item in data:
        qa_pairs = item.get("MultiHop_QA", {}).get("qa_pairs", [])
        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer_text = qa.get("answer", "").strip()

            # 跳过空问题或答案
            if not question or not answer_text:
                continue

            # 合并推理步骤
            reasoning_steps = qa.get("reasoning_steps", [])
            reasoning_text = "\n".join(
                [step.get("step", "").strip() for step in reasoning_steps if step.get("step", "").strip()])

            # 构建输出（推理过程 + 答案）
            if reasoning_text:
                output_text = f"{reasoning_text}\n\n{answer_text}"
            else:
                output_text = answer_text

            results.append({
                "instruction": instruction,
                "input": question,
                "output": output_text
            })

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存为qa.json（LlamaFactory标准格式）
    qa_file = os.path.join(output_dir, "qa.json")
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Conversion completed: {len(results)} QA pairs -> {qa_file}")
    return qa_file


def create_llamafactory_config(output_dir="../data"):
    """Create dataset_info.json for LlamaFactory"""

    # LlamaFactory dataset configuration
    dataset_info = {
        "kb_qa": {
            "file_name": "qa.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }

    config_file = os.path.join(output_dir, "dataset_info.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"LlamaFactory config created: {config_file}")
    print(f"Dataset name: kb_qa")
    return config_file


if __name__ == "__main__":
    # Convert QA data (using correct path format)
    input_file = "../.cache/gpu/batch_cleaning_step_step4.json"
    output_dir = "../data"

    print("Starting conversion to LlamaFactory format...")
    print(f"Input: {input_file}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # Convert data
    qa_file = convert_to_alpaca(input_file, output_dir)

    # Create config file
    config_file = create_llamafactory_config(output_dir)

    print(f"\nData conversion completed!")
    print(f"QA data: {qa_file}")
    print(f"Config: {config_file}")
    print(f"Dataset name: kb_qa")