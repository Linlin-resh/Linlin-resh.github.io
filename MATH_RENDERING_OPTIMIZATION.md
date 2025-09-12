# Math Formula Rendering Optimization

This document describes the optimizations made to improve math formula rendering on the website.

## Problem

The original math formulas were not displaying well on the web compared to Obsidian, with issues including:
- Poor spacing around mathematical expressions
- Inconsistent font rendering
- Lack of responsive design for mobile devices
- No dark mode support
- Poor alignment and readability

## Solution

### 1. MathJax Configuration

Switched from KaTeX to MathJax for better rendering and more features:

**File**: `hugo.toml`
```toml
[params.math]
  enable = true
  provider = 'mathjax'
  defer = true
  [params.math.mathjax]
    enable = true
    version = '3.2.2'
    inlineMath = [['$', '$'], ['\\(', '\\)']]
    displayMath = [['$$', '$$'], ['\\[', '\\]']]
    processEscapes = true
    processEnvironments = true
    skipHtmlTags = ['script', 'noscript', 'style', 'textarea', 'pre']
    ignoreHtmlClass = 'tex2jax_ignore'
    processHtmlClass = 'tex2jax_process'
```

### 2. Custom Math Styling

**File**: `static/css/math-enhancement.css`

Key improvements:
- **Better spacing**: Increased margins and padding around math formulas
- **Enhanced typography**: Improved font rendering and line height
- **Responsive design**: Math formulas scale properly on mobile devices
- **Dark mode support**: Proper styling for both light and dark themes
- **Formula numbering**: Automatic numbering for display math
- **Visual containers**: Math blocks have distinct styling with borders and backgrounds

### 3. Head Customization

**Files**: 
- `layouts/partials/head/math.html` - MathJax configuration
- `layouts/partials/head/custom.html` - Typography improvements

### 4. Typography Enhancements

- **Font families**: Added Source Serif Pro for better math readability
- **Code fonts**: Source Code Pro for code blocks
- **Line height**: Improved spacing for better readability
- **Responsive sizing**: Math formulas scale appropriately on different screen sizes

## Features

### Inline Math
- Proper spacing around inline formulas
- Background highlighting for better visibility
- Responsive font sizing

### Display Math
- Centered alignment
- Automatic numbering
- Visual containers with borders
- Better spacing from surrounding text

### Responsive Design
- **Desktop**: Full-size math formulas with optimal spacing
- **Tablet**: Slightly reduced font size for better fit
- **Mobile**: Compact display with maintained readability

### Dark Mode Support
- Proper color contrast in dark theme
- Consistent styling across themes
- Enhanced visibility

## Usage

### Writing Math Formulas

**Inline math**: Use single dollar signs
```markdown
The formula $E = mc^2$ is famous.
```

**Display math**: Use double dollar signs
```markdown
$$P(k) = \frac{\text{Number of nodes with degree } k}{n}$$
```

**Math containers**: For complex formulas
```markdown
<div class="math-container">
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
</div>
```

### Testing

Use the test page `content/posts/math-test.md` to verify math rendering:
- Inline formulas
- Display formulas
- Complex mathematical expressions
- Tables with math
- Code blocks with math references

## Browser Support

- **Chrome**: Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Edge**: Full support
- **Mobile browsers**: Responsive design

## Performance

- **MathJax 3.2.2**: Latest version with improved performance
- **Deferred loading**: Math formulas load after page content
- **CDN delivery**: Fast loading from jsdelivr.net
- **Caching**: Browser caching for repeated visits

## Maintenance

To update math rendering:
1. Modify `static/css/math-enhancement.css` for styling changes
2. Update `hugo.toml` for MathJax configuration changes
3. Test with `content/posts/math-test.md`
4. Verify across different browsers and devices

## Troubleshooting

### Math not rendering
- Check that `[params.math]` is enabled in `hugo.toml`
- Verify MathJax CDN is accessible
- Check browser console for JavaScript errors

### Poor spacing
- Verify `math-enhancement.css` is loaded
- Check for CSS conflicts
- Test with different screen sizes

### Dark mode issues
- Ensure CSS variables are properly defined
- Check `@media (prefers-color-scheme: dark)` rules
- Verify theme switching works correctly

## Future Improvements

- **LaTeX support**: Add support for LaTeX environments
- **Math search**: Implement search functionality for math formulas
- **Export options**: Add PDF export with proper math rendering
- **Accessibility**: Improve screen reader support for math content
