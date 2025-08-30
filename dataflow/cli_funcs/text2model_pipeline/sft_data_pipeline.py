#!/usr/bin/env python3
"""
SFT Data Processing Pipeline
完整的SFT数据合成流水线：基础清洗 + SFT数据合成 + 质量过滤
"""

import argparse
import json
from pathlib import Path
from dataflow.utils.storage import FileStorage
from dataflow.operators.general_text import (
    RemoveExtraSpacesRefiner,
    RemoveEmojiRefiner,
    HtmlUrlRemoverRefiner,
    WordNumberFilter,
    ColonEndFilter,
    SentenceNumberFilter,
    MinHashDeduplicator,
    SFTGeneratorSeed
)
from dataflow.serving import LocalModelLLMServing_vllm


class SFTDataPipeline:
    def __init__(self, input_file=None, cache_path=None, input_keys=None):
        if input_file is None:
            caller_cwd = Path.cwd()
            cli_input = caller_cwd / ".cache" / "pt_input.jsonl"
            if cli_input.exists():
                input_file = str(cli_input)
            else:
                input_file = "../example_data/GeneralTextPipeline/pt_input.jsonl"

        if cache_path is None:
            caller_cwd = Path.cwd()
            cli_cache = caller_cwd / ".cache"
            if cli_cache.exists():
                cache_path = "./.cache"
            else:
                cache_path = "./cache"

        if input_keys is None:
            self.input_keys = ["text"]
        elif isinstance(input_keys, str):
            if input_keys.startswith('[') and input_keys.endswith(']'):
                try:
                    keys_str = input_keys[1:-1]
                    self.input_keys = [key.strip().strip("'\"") for key in keys_str.split(',') if key.strip()]
                except:
                    self.input_keys = ["text"]
            else:
                self.input_keys = [input_keys]
        elif isinstance(input_keys, list):
            self.input_keys = input_keys
        else:
            self.input_keys = ["text"]

        print(f"Input keys to process: {self.input_keys}")

        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path + "/gpu",
            file_name_prefix="sft_dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './.cache'

        # 基础清洗算子
        self.remove_extra_spaces_refiner = RemoveExtraSpacesRefiner()
        self.remove_emoji_refiner = RemoveEmojiRefiner()
        self.html_remove_refiner = HtmlUrlRemoverRefiner()
        self.minhash_deduplicator = MinHashDeduplicator(num_perm=128, threshold=0.9, use_n_gram=True, ngram=5)
        self.word_number_filter = WordNumberFilter(min_words=20, max_words=100000)
        self.colon_end_filter = ColonEndFilter()
        self.sentence_number_filter = SentenceNumberFilter(min_sentences=3, max_sentences=7500)

        # SFT数据合成算子
        self.llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=8192
        )
        self.sft_generator = SFTGeneratorSeed(
            llm_serving=self.llm_serving,
            custom_prompt=None
        )

    def process_fields(self, operator, storage, step_name):
        print(f"  处理字段: {self.input_keys}")
        for field in self.input_keys:
            print(f"    开始处理字段: {field}")
            try:
                # 检查输入数据
                input_data = storage.read("dataframe")
                print(f"    输入数据量: {len(input_data)}")

                if len(input_data) > 0:
                    # 处理DataFrame或list格式
                    if hasattr(input_data, 'iloc'):
                        sample = input_data.iloc[0]
                    elif isinstance(input_data, list) and len(input_data) > 0:
                        sample = input_data[0]
                    else:
                        sample = {}

                    if field in sample:
                        field_content = str(sample[field])
                        print(f"    字段 {field} 内容示例: {field_content[:100]}...")
                    else:
                        print(f"    警告: 字段 {field} 不存在于数据中")
                        available_fields = list(sample.keys()) if hasattr(sample, 'keys') else []
                        print(f"    可用字段: {available_fields}")

                # 运行算子
                operator.run(storage=storage, input_key=field)

                # 检查输出数据
                output_data = storage.read("dataframe")
                print(f"    输出数据量: {len(output_data)}")
                print(f"    过滤掉: {len(input_data) - len(output_data)} 条")

            except Exception as e:
                print(f"    处理字段 {field} 时出错: {e}")
                raise

    def forward(self):
        # 第一步需要先调用step()来初始化存储
        current_storage = self.storage.step()

        # Step 1-8: 基础清洗
        print("Step 1: 移除多余空格")
        self.process_fields(self.remove_extra_spaces_refiner, current_storage, "remove_extra_spaces")
        current_storage = current_storage.step()

        print("Step 2: 移除表情符号")
        self.process_fields(self.remove_emoji_refiner, current_storage, "remove_emoji")
        current_storage = current_storage.step()

        print("Step 3: 移除HTML和URL")
        self.process_fields(self.html_remove_refiner, current_storage, "html_remove")
        current_storage = current_storage.step()

        print("Step 4: MinHash去重")
        first_key = self.input_keys[0]
        print(f"  使用字段 '{first_key}' 进行去重")
        input_data = current_storage.read("dataframe")
        print(f"  去重前数据量: {len(input_data)}")
        self.minhash_deduplicator.run(storage=current_storage, input_key=first_key)
        output_data = current_storage.read("dataframe")
        print(f"  去重后数据量: {len(output_data)}")
        current_storage = current_storage.step()

        print("Step 5: 跳过黑名单过滤（已禁用）")
        # 黑名单过滤已禁用，但仍需要维护存储链的连续性

        print("Step 6: 词数过滤")
        self.process_fields(self.word_number_filter, current_storage, "word_number")
        current_storage = current_storage.step()

        print("Step 7: 冒号结尾过滤")
        self.process_fields(self.colon_end_filter, current_storage, "colon_end")
        current_storage = current_storage.step()

        print("Step 8: 句子数过滤")
        self.process_fields(self.sentence_number_filter, current_storage, "sentence_number")
        current_storage = current_storage.step()

        # 根据用户输入的字段灵活处理
        # 检查是否需要SFT数据生成
        needs_sft_generation = True
        for sft_field in ["instruction", "output", "response", "answer", "query", "question"]:
            if sft_field in self.input_keys:
                needs_sft_generation = False
                break

        if needs_sft_generation:
            # 需要SFT生成，使用第一个字段作为内容源
            content_field = self.input_keys[0]
            print(f"Step 9: SFT数据合成（从 {content_field} 生成）")
            input_data = current_storage.read("dataframe")
            print(f"  SFT生成前数据量: {len(input_data)}")
            self.sft_generator.run(storage=current_storage, input_key=content_field)
            output_data = current_storage.read("dataframe")
            print(f"  SFT生成后数据量: {len(output_data)}")
            current_storage = current_storage.step()

            # 注意：以下质量过滤算子需要先初始化才能使用
            # 这里暂时注释掉，因为在__init__中没有初始化这些算子
            """
            print("Step 10: SFT输出长度过滤")
            self.word_number_filter_sft.run(storage=current_storage, input_key="output")
            current_storage = current_storage.step()

            print("Step 11: 指令IFD分数过滤")
            self.super_filtering_filter.run(storage=current_storage, input_key="instruction")
            current_storage = current_storage.step()

            print("Step 12: 指令质量得分过滤")
            self.deita_quality_filter.run(storage=current_storage, input_key="instruction")
            current_storage = current_storage.step()

            print("Step 13: Instruction标签数过滤")
            self.instag_filter.run(storage=current_storage, input_key="instruction")
            """
            print("质量过滤步骤已跳过，需要在__init__中初始化相应的算子")

        else:
            # 已有SFT格式，直接质量过滤
            print("=== 检测到SFT格式字段，直接进行质量过滤 ===")

            # 注意：这里也需要初始化相应的质量过滤算子
            print("质量过滤步骤已跳过，需要在__init__中初始化相应的算子")

        print("\n处理完成！")
        final_data = current_storage.read("dataframe")
        print(f"最终数据量: {len(final_data)}")


def main():
    parser = argparse.ArgumentParser(description="SFT Data Processing Pipeline")
    parser.add_argument("--input", default=None, help="Input JSONL file path")
    parser.add_argument("--cache", default=None, help="Cache directory path")
    parser.add_argument("--input-keys", default=None, help="Fields to process")
    args = parser.parse_args()

    pipeline = SFTDataPipeline(
        input_file=args.input,
        cache_path=args.cache,
        input_keys=args.input_keys
    )
    pipeline.forward()


if __name__ == "__main__":
    main()