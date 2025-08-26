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


def run_script_with_args(script_path: Path, description: str, args: list = None, cwd: str = None) -> bool:
    """Run a Python script with arguments and real-time output"""
    print(f"\n{Fore.BLUE}🔄 {description}{Style.RESET_ALL}")
    cmd = [sys.executable, script_path.name]
    if args:
        cmd.extend(args)
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    try:
        result = subprocess.run(cmd, cwd=cwd, check=True,
                               stdout=sys.stdout, stderr=sys.stderr, text=True)
        print(f"{Fore.GREEN}✅ {description} completed{Style.RESET_ALL}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ {description} failed{Style.RESET_ALL}")
        return False


def cli_sft(pdf_path: str = "./pdf", lf_yaml: str = "train_config.yaml", cache_path: str = "./") -> bool:
    """
    Simple SFT pipeline - just run the 4 scripts in order
    """
    print(f"{Fore.GREEN}🚀 Starting SFT Pipeline{Style.RESET_ALL}")
    print("=" * 50)

    # Get script directory - 代码在 dataflow/cli_funcs/ 中
    dataflow_dir = DataFlowPath.get_dataflow_dir()
    cli_funcs_dir = dataflow_dir / "cli_funcs"

    # 确保cache_path以/结尾，方便拼接
    if not cache_path.endswith(('/', '\\')):
        cache_path = cache_path + "/"

    if not cli_funcs_dir.exists():
        print(f"{Fore.RED}❌ CLI functions directory not found: {cli_funcs_dir}{Style.RESET_ALL}")
        return False

    # Step 1: PDF Detection
    script1 = cli_funcs_dir / "pdf2model_pipeline"/ "path2jsonl_script.py"
    args1 = [pdf_path, "--output", str(Path(cache_path) / ".cache" / "gpu" / "pdf_list.jsonl")]
    if not run_script_with_args(script1, "Step 1: PDF Detection", args1, cwd=str(cli_funcs_dir)):
        return False

    # Step 2: Data Processing
    script2 = cli_funcs_dir / "pdf2model_pipeline"/ "Pdf2QAPipeline.py"
    args2 = ["--cache", cache_path]
    if not run_script_with_args(script2, "Step 2: Data Processing", args2, cwd=str(cli_funcs_dir)):
        return False

    # Step 3: Data Conversion
    script3 = cli_funcs_dir /"pdf2model_pipeline" /"merge&filterQApairs.py"
    args3 = ["--cache", cache_path]
    if not run_script_with_args(script3, "Step 3: Data Conversion", args3, cwd=str(cli_funcs_dir)):
        return False

    # Step 4: Training
    script4 = cli_funcs_dir /"pdf2model_pipeline"/ "LlamaFactory.py"
    yaml_path = str(Path(cache_path) / ".cache" / lf_yaml) if not Path(lf_yaml).is_absolute() else lf_yaml
    args4 = ["--config", yaml_path, "--cache", cache_path]
    if not run_script_with_args(script4, "Step 4: Training", args4, cwd=str(cli_funcs_dir)):
        return False

    print(f"\n{Fore.GREEN}🎉 SFT Pipeline completed!{Style.RESET_ALL}")
    return True


if __name__ == "__main__":
    success = cli_sft()
    sys.exit(0 if success else 1)