import difflib
import sys

def compare_files(file1, file2):
    """
    Compare two text files and print differences in a human-readable format.
    """
    # Read both files into lists of lines
    with open(file1, 'r', encoding='utf-8') as f1:
        text1 = f1.readlines()
    with open(file2, 'r', encoding='utf-8') as f2:
        text2 = f2.readlines()

    # Generate unified diff (similar to git diff)
    diff = difflib.unified_diff(
        text1, text2,
        fromfile=file1,
        tofile=file2,
        lineterm=''  # Avoid extra newline
    )

    # Print diff result line by line
    for line in diff:
        # Optional: Add simple coloring for readability
        if line.startswith('+') and not line.startswith('+++'):
            print(f"\033[92m{line}\033[0m")  # Green for additions
        elif line.startswith('-') and not line.startswith('---'):
            print(f"\033[91m{line}\033[0m")  # Red for deletions
        else:
            print(line)

if __name__ == "__main__":
    # Expect exactly two arguments: the file paths
    if len(sys.argv) != 3:
        print("Usage: python compare_yaml.py file1.txt file2.txt")
        sys.exit(1)

    compare_files(sys.argv[1], sys.argv[2])