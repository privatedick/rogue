import argparse
import ast
import asyncio
from pathlib import Path


async def analyze_file(file_path):
    """Analyze a single Python file to assess the potential use of decorators."""
    results = []
    try:
        async with aiofiles.open(file_path, mode="r") as file:
            content = await file.read()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                score = calculate_decorator_score(node)
                results.append(
                    {"function": node.name, "score": score, "file": file_path}
                )
    except Exception as e:
        print(f"Error analyzing file {file_path}: {e}")
    return results


def calculate_decorator_score(node):
    """Calculate a score indicating the potential benefit of adding a decorator to a function."""
    # Basic scoring logic; this can be enhanced to check for more patterns.
    score = 0
    if (
        len(node.body) > 10
    ):  # For example, longer functions might benefit more from decorators.
        score += 10
    return score


async def analyze_directory(directory, include_self=False):
    """Analyze all Python files in the given directory and its subdirectories."""
    tasks = []
    async for file_path in get_python_files(directory, include_self):
        tasks.append(analyze_file(file_path))
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]


async def get_python_files(directory, include_self):
    """Yield Python files from the directory and its subdirectories, optionally excluding this script."""
    async for path in directory.rglob("*.py"):
        if include_self or path.name != Path(__file__).name:
            yield path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Python functions and suggest decorator usage."
    )
    parser.add_argument("directory", help="The directory to analyze.")
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include this script in the analysis.",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {args.directory} is not a valid directory.")
        return

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(analyze_directory(directory, args.include_self))
    if results:
        print("Analysis Results:")
        for result in results:
            print(
                f"File: {result['file']}, Function: {result['function']}, Score: {result['score']}"
            )
    else:
        print("No Python files found or no functions with potential decorator usage.")


if __name__ == "__main__":
    main()
