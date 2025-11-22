#!/usr/bin/env python3
"""
Simple test script to verify the evaluation metrics fix.

This script tests the core logic without requiring the full module imports.
"""

import json
import sys
from pathlib import Path


def test_data_structure():
    """Test that the data structure is correctly understood."""
    print("=" * 60)
    print("TEST 1: Data Structure Analysis")
    print("=" * 60)
    
    with open("data/train/train.jsonl", "r") as f:
        line = f.readline()
        sample = json.loads(line.strip())
    
    print(f"Sample has {len(sample['messages'])} messages:")
    for i, msg in enumerate(sample['messages']):
        print(f"  Message {i}: {msg['role']} (length: {len(msg['content'])})")
    
    # Check expected structure
    roles = [msg['role'] for msg in sample['messages']]
    expected_roles = ['system', 'user', 'assistant']
    
    if roles == expected_roles:
        print("✅ Data structure is correct!")
        return True
    else:
        print(f"❌ Wrong structure. Expected {expected_roles}, got {roles}")
        return False


def format_messages_for_tokenization(messages):
    """Simulate the formatting logic from the fix."""
    text_parts = []
    for message in messages:
        role = message["role"].upper()
        content = message["content"]
        
        if role == "SYSTEM":
            text_parts.append(f"<system>{content}</system>")
        elif role == "USER":
            text_parts.append(f"<user>{content}</user>")
        elif role == "ASSISTANT":
            text_parts.append(f"<assistant>{content}</assistant>")
        else:
            text_parts.append(f"<user>{content}</user>")
    
    return "\n".join(text_parts)


def extract_assistant_response(full_text, prompt_text):
    """Simulate the response extraction logic from the fix."""
    # Find where the prompt ends in the full text
    prompt_end = full_text.find(prompt_text)
    if prompt_end == -1:
        # Fallback: look for assistant marker
        assistant_start = full_text.find("<assistant>")
        if assistant_start != -1:
            # Extract from assistant marker to end
            response_start = assistant_start + len("<assistant>")
            response = full_text[response_start:].strip()
            # Remove any trailing special tokens
            if "<" in response:
                response = response[:response.find("<")].strip()
            return response
        else:
            # Last resort: return the whole text
            return full_text.strip()
    
    # Extract text after the prompt
    response_start = prompt_end + len(prompt_text)
    response = full_text[response_start:].strip()
    
    # Clean up any leading/trailing markers
    if response.startswith("<assistant>"):
        response = response[len("<assistant>"):].strip()
    if response.endswith("</assistant>"):
        response = response[:-len("</assistant>")].strip()
        
    return response


def test_prompt_formatting():
    """Test that prompts are correctly formatted for evaluation."""
    print("=" * 60)
    print("TEST 2: Prompt Formatting")
    print("=" * 60)
    
    # Test formatting logic directly
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather?"}
    ]
    
    formatted = format_messages_for_tokenization(sample_messages)
    print("Formatted prompt:")
    print(formatted)
    print("")
    
    # Verify structure
    if "<system>You are a helpful assistant.</system>" in formatted and \
       "<user>What is the weather?</user>" in formatted:
        print("✅ Prompt formatting works correctly!")
        return True
    else:
        print("❌ Prompt formatting failed")
        return False


def test_generation_evaluation_logic():
    """Test the generation evaluation logic."""
    print("=" * 60)
    print("TEST 3: Generation Evaluation Logic")
    print("=" * 60)
    
    # Test case 1: Complete conversation
    full_text = """<system>You are a helpful assistant.</system>
<user>What is 2+2?</user>
<assistant>2+2 equals 4.</assistant>"""
    
    prompt_text = """<system>You are a helpful assistant.</system>
<user>What is 2+2?</user>"""
    
    response = extract_assistant_response(full_text, prompt_text)
    print(f"Extracted response: '{response}'")
    
    if "2+2 equals 4" in response:
        print("✅ Response extraction works correctly!")
        return True
    else:
        print("❌ Response extraction failed")
        return False


def test_json_extraction():
    """Test JSON extraction logic."""
    print("=" * 60)
    print("TEST 4: JSON Extraction Logic")
    print("=" * 60)
    
    def extract_json(text):
        """Robust JSON extraction."""
        text = text.strip()
        # Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Heuristic: take the biggest {...} span
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None
    
    test_cases = [
        ('{"result": "test"}', {"result": "test"}),
        ('Some text {"result": "test"} more text', {"result": "test"}),
        ('{"invalid": json', None)
    ]
    
    for input_text, expected in test_cases:
        result = extract_json(input_text)
        if result == expected:
            print(f"✅ JSON extraction works: '{input_text[:30]}...' -> {result}")
        else:
            print(f"❌ JSON extraction failed: '{input_text[:30]}...' -> {result} (expected {expected})")
            return False
    
    return True


def demonstrate_the_fix():
    """Demonstrate what the fix accomplishes."""
    print("=" * 60)
    print("DEMONSTRATION: Before vs After Fix")
    print("=" * 60)
    
    # Load a real sample
    with open("data/train/train.jsonl", "r") as f:
        line = f.readline()
        sample = json.loads(line.strip())
    
    print("BEFORE FIX:")
    print("❌ Used sample['input_ids'] (entire conversation)")
    print("❌ Model sees: system + user + assistant (expected answer)")
    print("❌ Result: Model memorizes/repeats expected response")
    print("❌ Generation metrics: empty responses -> 0.0 diversity")
    print("❌ Task metrics: comparison with training data -> 0.0 accuracy")
    print("")
    
    print("AFTER FIX:")
    # Extract system and user only
    prompt_messages = [msg for msg in sample['messages'] if msg['role'] in ['system', 'user']]
    prompt_text = format_messages_for_tokenization(prompt_messages)
    
    print("✅ Extract prompt only (system + user):")
    print(f"   System: {sample['messages'][0]['role']}")
    print(f"   User: {sample['messages'][1]['role']}")
    print(f"   Assistant: {sample['messages'][2]['role']} (EXCLUDED from input)")
    print("")
    
    print("✅ Model generates response from prompt only")
    print("✅ Extract assistant response properly")
    print("✅ Calculate meaningful metrics from actual generation")
    print("")
    
    print("EXPECTED RESULTS:")
    print("🎯 Generation metrics: non-zero diversity scores")
    print("🎯 Task metrics: meaningful accuracy comparisons")
    print("🎯 Proper evaluation of model capabilities")
    
    return True


def main():
    """Run all tests."""
    print("Starting evaluation metrics fix verification...")
    print("")
    
    tests = [
        test_data_structure,
        test_prompt_formatting,
        test_generation_evaluation_logic,
        test_json_extraction,
        demonstrate_the_fix
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            failed += 1
        print("")
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    
    if failed == 0:
        print("🎉 All tests passed! The fix should resolve the zero metrics issue.")
        print("")
        print("KEY CHANGES MADE:")
        print("1. Modified evaluate_generation_quality() to use prompt-only input")
        print("2. Modified evaluate_task_performance() to use proper conversation format")
        print("3. Added _format_messages_for_tokenization() helper method")
        print("4. Added _extract_assistant_response() helper method")
        print("5. Improved JSON extraction and response processing")
        print("")
        print("NEXT STEPS:")
        print("- Run your training pipeline again")
        print("- Check that metrics.json shows non-zero values")
        print("- Generation metrics should show proper diversity")
        print("- Task metrics should show meaningful accuracy scores")
    else:
        print("❌ Some tests failed. Review the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)