"""
Quick Fix for Missing Dependencies in LLM Training Notebook

This script provides the fix for the ModuleNotFoundError in the notebook.
Run this code to handle missing dependencies gracefully.
"""

# Add this code at the beginning of cell [5] in the notebook

# Optional imports - handle missing dependencies gracefully
try:
    import evaluate
    EVALUATE_AVAILABLE = True
    print("✓ 'evaluate' module available")
except ImportError:
    EVALUATE_AVAILABLE = False
    print("⚠️  Warning: 'evaluate' module not available. Some advanced evaluation metrics will be disabled.")
    print("   To install: pip install evaluate")

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
    print("✓ 'sklearn' module available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Warning: 'sklearn' not available. Some evaluation metrics will be disabled.")
    print("   To install: pip install scikit-learn")

# Now you can continue with the original code
# The LLMEvaluator class will work without these dependencies

print("\nDependency check completed. You can now run the evaluation code.")