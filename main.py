#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗干预智能体数据集生成脚本 - 主入口
严格落地两篇ToM论文核心方案
"""

import argparse

from dataset_generator import MedicalDatasetGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate ToM-based medical intervention agent dataset')
    parser.add_argument('--input', type=str, default='ehr_bench_decision_making.jsonl',
                        help='Input EHR JSONL file path')
    parser.add_argument('--output', type=str, default='./output_tom',
                        help='Output directory for generated datasets')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['diagnosis', 'medrecon', 'prescriptions'],
                        help='Task types to generate')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=None,
                        help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default='gpt-4',
                        help='LLM model to use')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay between API calls in seconds')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ToM-Based Medical Dataset Generator")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Input file: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  Task types: {args.tasks}")
    print(f"  Max samples: {args.max_samples or 'All'}")
    print(f"  Model: {args.model}")
    print(f"  API delay: {args.delay}s")
    
    generator = MedicalDatasetGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model
    )
    
    generator.process_ehr_file(
        input_file=args.input,
        output_dir=args.output,
        task_types=args.tasks,
        max_samples=args.max_samples,
        delay=args.delay
    )


if __name__ == "__main__":
    main()
