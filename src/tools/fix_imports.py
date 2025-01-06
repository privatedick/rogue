"""Fix imports in all project files."""

from pathlib import Path


def fix_imports():
    """Fix imports in all Python files."""
    tools_dir = Path(__file__).parent

    # Fixa tools/__init__.py
    init_file = tools_dir / "__init__.py"
    if init_file.exists():
        init_file.write_text('"""Tools package."""\n')

    # Fixa alla Python-filer i tools-katalogen
    for py_file in tools_dir.glob("*.py"):
        if py_file.name == "fix_imports.py":
            continue

        content = py_file.read_text()
        # Vi behåller src.tools för entry points men använder relativa importer inom paketet
        fixed_content = content.replace("from src.tools.", "from .")
        fixed_content = fixed_content.replace("import src.tools.", "import .")

        if fixed_content != content:
            print(f"Fixing imports in {py_file.name}")
            py_file.write_text(fixed_content)


if __name__ == "__main__":
    fix_imports()
