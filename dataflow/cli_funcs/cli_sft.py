#!/usr/bin/env python3
"""
DataFlow SFT CLI Module - dataflow/cli_funcs/cli_sft.py
Simple PDF to SFT training pipeline - just run the scripts in order
"""

import subprocess
import sys
from pathlib import Path
from colorama import Fore, Style
from dataflow import get_logger
from .paths import DataFlowPath

logger = get_logger()


def run_script(script_path: Path, description: str, cwd: str = None) -> bool:
    """Run a Python script with real-time output"""
    print(f"\n{Fore.BLUE}🔄 {description}{Style.RESET_ALL}")
    print(f"Running: python {script_path.name}")
    if cwd:
        print(f"Working directory: {cwd}")

    try:
        # 关键修改：去掉capture_output，改用stdout/stderr继承父进程输出
        result = subprocess.run([sys.executable, script_path.name],
                                cwd=cwd, check=True,
                                stdout=sys.stdout,  # 子进程stdout直接输出到终端
                                stderr=sys.stderr,  # 子进程stderr直接输出到终端
                                text=True)
        print(f"{Fore.GREEN}✅ {description} completed{Style.RESET_ALL}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ {description} failed{Style.RESET_ALL}")
        return False


def cli_sft(config_path: str = "train_config.yaml") -> bool:
    """
    Simple SFT pipeline - just run the 4 scripts in order
    """
    print(f"{Fore.GREEN}🚀 Starting SFT Pipeline{Style.RESET_ALL}")
    print("=" * 50)

    # Get script directory
    dataflow_dir = DataFlowPath.get_dataflow_dir()
    example_dir = dataflow_dir / "example" / "Pdf2QAPipeline"

    if not example_dir.exists():
        print(f"{Fore.RED}❌ Example directory not found: {example_dir}{Style.RESET_ALL}")
        return False

    # Step 1: PDF Detection
    script1 = example_dir / "data_process" / "path2jsonl_script.py"
    if not run_script(script1, "Step 1: PDF Detection", cwd=str(example_dir / "data_process")):
        return False

    # Step 2: Data Processing
    script2 = example_dir / "Pdf2QAPipeline.py"
    if not run_script(script2, "Step 2: Data Processing", cwd=str(example_dir)):
        return False

    # Step 3: Data Conversion
    script3 = example_dir / "data_process" / "merge&filterQApairs.py"
    if not run_script(script3, "Step 3: Data Conversion", cwd=str(example_dir / "data_process")):
        return False

    # Step 4: Training
    script4 = example_dir / "LlamaFactory.py"
    if not run_script(script4, "Step 4: Training", cwd=str(example_dir)):
        return False

    print(f"\n{Fore.GREEN}🎉 SFT Pipeline completed!{Style.RESET_ALL}")
    return True


if __name__ == "__main__":
    success = cli_sft()
    sys.exit(0 if success else 1)