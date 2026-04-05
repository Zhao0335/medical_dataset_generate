#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗干预智能体数据集生成脚本 - 主入口
严格落地两篇ToM论文核心方案
"""

import argparse
import sys

from dataset_generator import MedicalDatasetGenerator
from utils import ConfigurationError
from logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(
        description='Generate ToM-based medical intervention agent dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use OpenAI API
  python main.py --provider openai --api_key YOUR_API_KEY --model gpt-4

  # Use local model
  python main.py --provider local --local_model_path ~/models/alfworld-model

  # Use vLLM for high-performance inference
  python main.py --provider vllm --local_model_path ~/models/alfworld-model
        """
    )
    
    # LLM Provider arguments
    parser.add_argument('--provider', type=str, default='openai',
                        choices=['openai', 'local', 'vllm'],
                        help='LLM provider type (default: openai)')
    
    # OpenAI arguments
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='OpenAI API base URL (or set OPENAI_BASE_URL env)')
    parser.add_argument('--model', type=str, default='gpt-4',
                        help='LLM model name (default: gpt-4)')
    
    # Local model arguments
    parser.add_argument('--local_model_path', type=str, default=None,
                        help='Path to local model (for local/vllm provider)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for local model (default: auto)')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Load model in 8-bit mode')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load model in 4-bit mode')
    
    # Data processing arguments
    parser.add_argument('--input', type=str, default='ehr_bench_decision_making.jsonl',
                        help='Input EHR JSONL file path')
    parser.add_argument('--output', type=str, default='./output_tom',
                        help='Output directory for generated datasets')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['diagnosis', 'medrecon', 'prescriptions'],
                        help='Task types to generate')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay between API calls in seconds')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ToM-Based Medical Dataset Generator")
    logger.info("=" * 60)
    logger.info(f"Provider: {args.provider}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Task types: {args.tasks}")
    logger.info(f"Max samples: {args.max_samples or 'All'}")
    logger.info(f"API delay: {args.delay}s")
    
    if args.provider == 'local' or args.provider == 'vllm':
        logger.info(f"Local model path: {args.local_model_path}")
        logger.info(f"Device: {args.device}")
    
    try:
        generator = MedicalDatasetGenerator(
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            local_model_path=args.local_model_path,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )
        
        generator.process_ehr_file(
            input_file=args.input,
            output_dir=args.output,
            task_types=args.tasks,
            max_samples=args.max_samples,
            delay=args.delay
        )
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        if args.provider == 'openai':
            logger.error("Please set OPENAI_API_KEY environment variable or provide --api_key argument")
        else:
            logger.error("Please provide --local_model_path argument")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
