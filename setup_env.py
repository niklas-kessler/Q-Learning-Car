"""
Environment setup script to prevent OpenMP conflicts.
Run this before running your main application if you encounter OpenMP errors.
"""
import os

# Set environment variable to handle potential OpenMP library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("Environment variable KMP_DUPLICATE_LIB_OK set to TRUE")
print("This prevents OpenMP library conflicts between PyTorch and other libraries.")