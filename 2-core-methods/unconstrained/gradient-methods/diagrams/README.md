# Line Search Algorithm Architecture (Draw.io Template)

This directory contains Draw.io diagram source files for line search documentation.

## 📐 Included Diagrams

### 1. `line_search_architecture.drawio`
High-level architecture showing:
- How line search fits into optimization algorithms
- Relationship between direction computation and step size selection
- Different line search variants

### 2. `backtracking_flowchart.drawio`
Detailed flowchart of backtracking algorithm:
- Start with α_init
- Armijo condition check
- Shrinking loop
- Acceptance and return

### 3. `wolfe_conditions.drawio`
Visual explanation of Wolfe conditions:
- φ(α) curve
- Armijo line (sufficient decrease)
- Curvature condition regions
- Acceptable step size zones

## 🎨 How to Use These Diagrams

### Opening in VS Code (with Draw.io Extension)

1. **Open any `.drawio` file** in VS Code
2. **Edit directly** in the editor
3. **Export** to PNG/SVG/PDF for inclusion in documentation

### Creating New Diagrams

1. `Ctrl+Shift+P` → "Draw.io: Create New Diagram"
2. Choose template or start blank
3. Save with `.drawio` extension
4. Export to image format for README

### Embedding in Markdown

After exporting to PNG/SVG:

```markdown
![Line Search Architecture](./diagrams/line_search_architecture.png)
```

## 🖼️ Diagram Style Guide

**Consistent styling across all diagrams:**

- **Colors:**
  - Primary flow: Blue (#2196F3)
  - Decision points: Orange (#FF9800)
  - Success: Green (#4CAF50)
  - Failure/Error: Red (#F44336)
  - Background: White/Light Gray

- **Shapes:**
  - Start/End: Rounded rectangle
  - Process: Rectangle
  - Decision: Diamond
  - Data/Storage: Parallelogram

- **Fonts:**
  - Title: 16pt, Bold
  - Labels: 12pt, Regular
  - Annotations: 10pt, Italic

## 📂 Directory Structure

```
gradient-methods/
├── README.md                                    # Main documentation
├── line_search_interactive.ipynb                # Jupyter notebook
├── VISUALIZATION_GUIDE.md                       # This guide
└── diagrams/                                    # Draw.io diagrams
    ├── line_search_architecture.drawio          # High-level overview
    ├── line_search_architecture.png             # Exported image
    ├── backtracking_flowchart.drawio            # Backtracking detail
    ├── backtracking_flowchart.png               # Exported image
    ├── wolfe_conditions.drawio                  # Wolfe visualization
    └── wolfe_conditions.png                     # Exported image
```

## 🎯 Recommended Workflow

### For Documentation

1. **README.md** → High-level explanation with LaTeX math
2. **Draw.io diagrams** → Visual algorithm flow and concepts
3. **Jupyter notebook** → Interactive exploration and experiments

### For Learning

1. **Start with README** → Understand theory
2. **View Draw.io diagrams** → Visualize algorithm structure
3. **Run Jupyter notebook** → Experiment interactively
4. **Check existing .py files** → See production code

### For Presentations

1. **Create diagrams in Draw.io** → Professional visuals
2. **Generate plots in Jupyter** → Data-driven insights
3. **Export to PDF** → Complete document from Markdown Preview Enhanced

## 💡 Pro Tips

1. **Reusable Components**: Create a "components.drawio" file with reusable shapes
2. **Layers**: Use layers in Draw.io for complex diagrams (e.g., show/hide implementation details)
3. **Links**: Link Draw.io diagrams to specific code files for traceability
4. **Version Control**: .drawio files are XML, so they work well with git
5. **Export Multiple Formats**: Keep SVG for scalability, PNG for compatibility

## 🔗 Integration Examples

### In README.md

```markdown
## Algorithm Overview

The line search algorithm follows this general structure:

![Line Search Architecture](./diagrams/line_search_architecture.png)

### Backtracking Method Details

Here's the detailed flowchart for backtracking line search:

![Backtracking Flowchart](./diagrams/backtracking_flowchart.png)

See the [interactive notebook](./line_search_interactive.ipynb) for live examples!
```

### In Jupyter Notebook

```python
from IPython.display import Image, display

# Display diagram in notebook
display(Image(filename='./diagrams/wolfe_conditions.png'))
```

## 🎓 Educational Use

### For Teaching

1. **Lecture slides**: Export Draw.io diagrams as high-res images
2. **Assignments**: Link students to interactive notebook
3. **Study guides**: Combine README theory with visual diagrams

### For Self-Study

1. **Read section in README**
2. **View corresponding diagram**
3. **Run notebook cells**
4. **Modify parameters and observe changes**

## 📦 Required VS Code Extensions

```vscode-extensions
hediet.vscode-drawio,shd101wyy.markdown-preview-enhanced,ms-toolsai.jupyter
```

All three work together seamlessly!

---

**Created:** October 18, 2025  
**Maintained by:** ewdhp/optimization project
