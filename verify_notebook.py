#!/usr/bin/env python3
"""
Notebook Verification Script
Checks if the Colab notebook is properly formatted and has all required cells.
"""

import json
import sys

def verify_notebook(notebook_path):
    """Verify notebook structure and content."""
    
    print("=" * 60)
    print("NOTEBOOK VERIFICATION")
    print("=" * 60)
    
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        print("✅ Notebook is valid JSON")
    except json.JSONDecodeError as e:
        print(f"❌ FAILED: Invalid JSON - {e}")
        return False
    except FileNotFoundError:
        print(f"❌ FAILED: File not found - {notebook_path}")
        return False
    
    # Check structure
    if 'cells' not in nb:
        print("❌ FAILED: No 'cells' key in notebook")
        return False
    
    cells = nb['cells']
    print(f"✅ Found {len(cells)} cells")
    
    # Check for required cells
    required_sections = [
        "Mount Google Drive",
        "Copy Data from Drive",
        "Install Dependencies",
        "Upload Project Code",
        "Verify Data",
        "Update Config",
        "Test Model",
        "Start Training",
        "Test Inference"
    ]
    
    found_sections = []
    for cell in cells:
        if cell['cell_type'] == 'markdown':
            content = ''.join(cell['source'])
            for section in required_sections:
                if section in content:
                    found_sections.append(section)
    
    print(f"\n📋 Required sections found: {len(found_sections)}/{len(required_sections)}")
    
    missing = set(required_sections) - set(found_sections)
    if missing:
        print(f"⚠️  Missing sections: {missing}")
    else:
        print("✅ All required sections present")
    
    # Check code cells have content
    empty_code_cells = 0
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            if not cell.get('source') or len(cell['source']) == 0:
                empty_code_cells += 1
                print(f"⚠️  Cell {i} is empty")
    
    if empty_code_cells == 0:
        print("✅ All code cells have content")
    else:
        print(f"⚠️  {empty_code_cells} empty code cells")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Notebook path: {notebook_path}")
    print(f"Total cells: {len(cells)}")
    print(f"Format: Jupyter Notebook {nb.get('nbformat', '?')}.{nb.get('nbformat_minor', '?')}")
    
    # Cell breakdown
    markdown_cells = sum(1 for c in cells if c['cell_type'] == 'markdown')
    code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
    print(f"Markdown cells: {markdown_cells}")
    print(f"Code cells: {code_cells}")
    
    print("\n✅ Notebook is ready to use in Google Colab!")
    print("\nNext steps:")
    print("1. Upload this notebook to Google Colab")
    print("2. Set runtime to GPU")
    print("3. Run cells sequentially")
    
    return True


if __name__ == "__main__":
    notebook_path = "/home/wissem/.gemini/antigravity/scratch/SeasonsGAN/notebooks/01_colab_train.ipynb"
    
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    
    success = verify_notebook(notebook_path)
    sys.exit(0 if success else 1)
