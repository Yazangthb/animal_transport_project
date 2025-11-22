#!/usr/bin/env python3
"""
Demonstration script for the detailed evaluation results feature.

This script shows how to:
1. Parse and analyze the detailed evaluation results
2. Find common failure patterns
3. Understand model mistakes
4. Get insights for improving the model

Usage:
    python detailed_results_demo.py [path_to_results_file]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def load_evaluation_results(filepath: str) -> Dict:
    """Load and parse evaluation results file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_failures(detailed_results: List[Dict]) -> Dict:
    """Analyze common failure patterns in the results."""
    failures = {
        'json_parsing_failures': [],
        'schema_violations': [],
        'mode_errors': [],
        'reasoning_issues': [],
        'hallucinations': []
    }
    
    for result in detailed_results:
        if not result.get('valid_json', False):
            failures['json_parsing_failures'].append(result)
        
        if result.get('valid_json') and not result.get('schema_compliant', False):
            failures['schema_violations'].append(result)
        
        if result.get('valid_json'):
            if not result.get('exact_match', False):
                failures['mode_errors'].append(result)
            
            if not result.get('rule_attribution_correct', False):
                failures['reasoning_issues'].append(result)
            
            if result.get('hallucination_detected', False):
                failures['hallucinations'].append(result)
    
    return failures


def print_sample_analysis(title: str, results: List[Dict], max_samples: int = 3):
    """Print analysis of a sample of results."""
    print(f"\n{title}:")
    print("=" * len(title))
    print(f"Total failures: {len(results)}")
    
    if not results:
        print("No failures found!")
        return
    
    print(f"\nShowing first {min(max_samples, len(results))} examples:")
    print("-" * 50)
    
    for i, result in enumerate(results[:max_samples]):
        print(f"\nSample {i+1} (Index: {result.get('sample_index', 'N/A')}):")
        print(f"Input: {result.get('user_input', 'N/A')[:100]}...")
        print(f"Expected: {json.dumps(result.get('ground_truth', {}), indent=2)[:200]}...")
        print(f"Generated: {result.get('generated_response', 'N/A')[:200]}...")
        
        if result.get('generated_json'):
            print(f"Generated JSON: {json.dumps(result.get('generated_json', {}), indent=2)[:200]}...")
        
        # Analysis flags
        flags = []
        if not result.get('valid_json'):
            flags.append("INVALID_JSON")
        if not result.get('schema_compliant'):
            flags.append("SCHEMA_VIOLATION")
        if not result.get('exact_match'):
            flags.append("EXACT_MISMATCH")
        if not result.get('allowed_modes_correct'):
            flags.append("WRONG_MODES")
        if result.get('hallucination_detected'):
            flags.append("HALLUCINATION")
        
        if flags:
            print(f"Issues: {', '.join(flags)}")


def get_improvement_suggestions(failures: Dict) -> List[str]:
    """Generate suggestions for improving the model based on failure analysis."""
    suggestions = []
    
    json_failures = len(failures['json_parsing_failures'])
    schema_failures = len(failures['schema_violations'])
    mode_errors = len(failures['mode_errors'])
    reasoning_issues = len(failures['reasoning_issues'])
    hallucinations = len(failures['hallucinations'])
    
    if json_failures > 0:
        suggestions.append(f"• Focus on JSON formatting: {json_failures} samples failed JSON parsing")
        suggestions.append("  - Add more examples with proper JSON structure")
        suggestions.append("  - Consider adding JSON validation in training")
    
    if schema_failures > 0:
        suggestions.append(f"• Improve schema compliance: {schema_failures} samples violated schema")
        suggestions.append("  - Add schema validation examples to training data")
        suggestions.append("  - Consider strict JSON schema enforcement")
    
    if mode_errors > 0:
        suggestions.append(f"• Fix mode selection logic: {mode_errors} samples had incorrect modes")
        suggestions.append("  - Review training data for mode assignment rules")
        suggestions.append("  - Add more diverse examples for edge cases")
    
    if reasoning_issues > 0:
        suggestions.append(f"• Improve reasoning quality: {reasoning_issues} samples had poor reasoning")
        suggestions.append("  - Add more detailed reasoning examples")
        suggestions.append("  - Focus on rule attribution training")
    
    if hallucinations > 0:
        suggestions.append(f"• Reduce hallucinations: {hallucinations} samples contained hallucinations")
        suggestions.append("  - Improve training data quality")
        suggestions.append("  - Add negative examples showing what NOT to do")
    
    return suggestions


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python detailed_results_demo.py <path_to_results_file>")
        print("\nAvailable result files:")
        result_dir = Path("evaluation_results")
        if result_dir.exists():
            for file in result_dir.glob("*.json"):
                print(f"  - {file}")
        else:
            print("  No evaluation_results directory found.")
            print("  Run training with save_detailed_results=True first.")
        return
    
    results_file = sys.argv[1]
    if not Path(results_file).exists():
        print(f"Error: File {results_file} not found!")
        return
    
    print("🔍 DETAILED EVALUATION RESULTS ANALYSIS")
    print("=" * 50)
    
    # Load results
    data = load_evaluation_results(results_file)
    detailed_results = data.get('detailed_results', [])
    metadata = data.get('metadata', {})
    
    print(f"📊 Dataset Summary:")
    print(f"  Total samples: {metadata.get('total_samples', len(detailed_results))}")
    print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"  File: {results_file}")
    
    # Analyze failures
    failures = analyze_failures(detailed_results)
    
    # Print failure analysis
    print_sample_analysis("JSON Parsing Failures", failures['json_parsing_failures'])
    print_sample_analysis("Schema Violations", failures['schema_violations'])
    print_sample_analysis("Mode Selection Errors", failures['mode_errors'])
    print_sample_analysis("Reasoning Issues", failures['reasoning_issues'])
    print_sample_analysis("Hallucinations", failures['hallucinations'])
    
    # Generate suggestions
    suggestions = get_improvement_suggestions(failures)
    
    print("\n💡 IMPROVEMENT SUGGESTIONS")
    print("=" * 30)
    if suggestions:
        for suggestion in suggestions:
            print(suggestion)
    else:
        print("No significant issues found! The model is performing well.")
    
    # Additional insights
    print("\n📈 PERFORMANCE INSIGHTS")
    print("=" * 25)
    
    if detailed_results:
        total = len(detailed_results)
        valid_json = sum(1 for r in detailed_results if r.get('valid_json', False))
        exact_matches = sum(1 for r in detailed_results if r.get('exact_match', False))
        
        print(f"JSON validity rate: {valid_json/total*100:.1f}% ({valid_json}/{total})")
        print(f"Exact match rate: {exact_matches/total*100:.1f}% ({exact_matches}/{total})")
        
        # Most common mode errors
        mode_errors = [r for r in detailed_results if r.get('valid_json') and not r.get('exact_match', False)]
        if mode_errors:
            print(f"\nMode error breakdown:")
            print(f"  - Wrong allowed modes: {sum(1 for r in mode_errors if not r.get('allowed_modes_correct', False))}")
            print(f"  - Wrong disallowed modes: {sum(1 for r in mode_errors if not r.get('disallowed_modes_correct', False))}")
            print(f"  - Time estimation errors: {sum(1 for r in mode_errors if r.get('time_errors'))}")
    
    print("\n✅ Analysis complete!")
    print(f"Detailed results saved in: {Path(results_file).parent}")


if __name__ == "__main__":
    main()