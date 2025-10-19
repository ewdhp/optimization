# 🎯 Quick Reference: Visualization Toolkit

## ⚡ Instant Commands

| What You Want | Command | Extension |
|---------------|---------|-----------|
| Preview markdown with math | `Ctrl+K V` | Markdown Preview Enhanced |
| Open Jupyter notebook | Click `.ipynb` file | Jupyter |
| Create new diagram | `Ctrl+Shift+P` → "Draw.io: Create" | Draw.io |
| Run Python code in markdown | Right-click preview → "Run Code Chunk" | Markdown Preview Enhanced |
| Export to PDF | Right-click preview → Export → PDF | Markdown Preview Enhanced |

## 📊 When to Use What

### Use **Markdown Preview Enhanced** for:
✅ Theory documentation with LaTeX  
✅ Quick inline plots  
✅ Mermaid flowcharts  
✅ Final documentation (export to PDF)  

### Use **Jupyter Notebook** for:
✅ Interactive analysis  
✅ Parameter exploration with widgets  
✅ Complex multi-step computations  
✅ Data science workflows  

### Use **Draw.io** for:
✅ Architecture diagrams  
✅ Professional flowcharts  
✅ System design diagrams  
✅ Presentation-quality visuals  

## 🔥 Most Useful Features

### Markdown Preview Enhanced

```markdown
<!-- Math -->
$$\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k$$

<!-- Mermaid diagram -->
```mermaid
graph LR
    A --> B --> C
```

<!-- Python plot -->
```python {cmd=true matplotlib=true}
import matplotlib.pyplot as plt
plt.plot([1,2,3], [1,4,9])
plt.show()
```
```

### Jupyter Interactive Widgets

```python
from ipywidgets import interact, FloatSlider

@interact(alpha=FloatSlider(min=0, max=1, value=0.5))
def plot(alpha):
    # Your code here
    pass
```

### Draw.io Shortcuts

- `Ctrl+D` - Duplicate
- `Ctrl+G` - Group
- `Alt+Shift+Arrow` - Auto-connect
- `Ctrl+Shift+F9` - Export

## 📁 Recommended Structure

```
your-project/
├── README.md                    # Main docs (Markdown)
├── analysis.ipynb              # Interactive (Jupyter)
├── diagrams/
│   ├── architecture.drawio     # Editable (Draw.io)
│   └── architecture.png        # Export
└── TOOLKIT_GUIDE.md            # This file
```

## 🚀 Quick Start Templates

### Template 1: Documentation Page

```markdown
# Algorithm Name

## Theory
$$f(x) = ...$$

## Architecture
![Diagram](./diagrams/arch.png)

## Demo
See [notebook](./demo.ipynb)

## Code
```python
def algorithm():
    pass
```
```

### Template 2: Jupyter Cell Structure

```python
# Cell 1: Imports + Setup
import numpy as np
import matplotlib.pyplot as plt

# Cell 2: Functions
def my_function(x):
    return x**2

# Cell 3: Interactive
@interact(x=FloatSlider(...))
def plot(x):
    plt.plot(...)
    plt.show()

# Cell 4: Analysis
results = ...
print(results)
```

## 💊 Common Fixes

| Problem | Solution |
|---------|----------|
| Math doesn't render | Use MPE preview, not built-in |
| Code doesn't execute | Check Python path in settings |
| Jupyter kernel fails | `pip install ipykernel` |
| Diagrams don't show | Check file path is correct |

## 🎨 Style Guide

**Colors (consistent across all):**
- Primary: #2196F3 (Blue)
- Success: #4CAF50 (Green)
- Warning: #FF9800 (Orange)
- Error: #F44336 (Red)

**Fonts:**
- Headings: Bold, 14-16pt
- Body: Regular, 11-12pt
- Code: Monospace, 10pt

## 📚 Learn More

- Markdown Preview Enhanced: [Docs](https://shd101wyy.github.io/markdown-preview-enhanced/)
- Jupyter Widgets: [Guide](https://ipywidgets.readthedocs.io/)
- Draw.io: [Examples](https://www.diagrams.net/example-diagrams)

---

**Print this page and keep it handy! 📌**
