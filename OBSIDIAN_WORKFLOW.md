# Obsidian + Hugo 工作流指南

## 概述

本指南说明如何在 Obsidian 中管理您的 Hugo 站点内容，实现从写作到发布的完整工作流。

## 初始设置

### 1. 在 Obsidian 中打开站点仓库

1. 打开 Obsidian
2. 选择 "Open folder as vault"
3. 选择您的 Hugo 站点目录：`D:\Code\20250829-linlinewbsite`
4. 这将创建一个新的 Vault

### 2. 安装必要的插件

#### Obsidian Git（必需）
- 在 Community Plugins 中搜索 "Obsidian Git"
- 安装并启用
- 配置自动备份间隔（推荐：每 5 分钟）

#### 其他推荐插件
- **Calendar**: 管理文章发布时间
- **Templater**: 创建文章模板
- **Tag Wrangler**: 管理标签
- **Word Count**: 显示字数统计

## 文章写作工作流

### 1. 创建新文章

#### 使用模板（推荐）
1. 按 `Ctrl+N` 创建新笔记
2. 选择 "Post Template" 模板
3. 填写必要信息

#### 手动创建
```markdown
---
title: "文章标题"
date: 2025-08-29
draft: false
description: "文章描述"
tags: ["tag1", "tag2", "tag3"]
showToc: true
TocOpen: true
---

## 引言

文章内容...

## 数学公式

使用 KaTeX 语法：
- 行内公式：$E = mc^2$
- 块级公式：$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

## 代码块

```python
def hello_world():
    print("Hello, World!")
```

## 图片

![图片描述](/images/image-name.jpg)

## 结论

总结...
```

### 2. 文章分类

根据内容类型选择适当的目录：

- **`content/posts/reading-notes/`**: 阅读笔记
- **`content/posts/ideas/`**: 想法和思考
- **`content/posts/progress/`**: 研究进展
- **`content/posts/tools/`**: 工具和代码

### 3. 标签管理

使用一致的标签系统：

#### 内容类型标签
- `reading-notes`: 阅读笔记
- `ideas`: 想法和思考
- `research-progress`: 研究进展
- `tools`: 工具和代码

#### 主题标签
- `network-theory`: 网络理论
- `materials-science`: 材料科学
- `machine-learning`: 机器学习
- `graph-theory`: 图论
- `nanowires`: 纳米线
- `mathematics`: 数学

#### 技术标签
- `python`: Python 代码
- `julia`: Julia 代码
- `networkx`: NetworkX 库
- `pytorch`: PyTorch 框架

## 图片管理

### 1. 图片存储位置
- 所有图片存储在 `static/images/` 目录
- 按年份或主题组织子目录
- 使用描述性文件名

### 2. 图片引用
```markdown
![图片描述](/images/2025/08/network-analysis.png)
```

### 3. 图片优化建议
- 使用 WebP 格式（更好的压缩）
- 保持合理的文件大小（< 500KB）
- 提供适当的 alt 文本

## Git 工作流

### 1. 自动备份设置

在 Obsidian Git 设置中：
- 启用 "Auto backup"
- 设置备份间隔：5 分钟
- 启用 "Auto push after commit"

### 2. 手动 Git 操作

#### 提交更改
1. 在 Obsidian 中按 `Ctrl+Shift+A` 打开 Git 面板
2. 查看更改的文件
3. 输入提交信息
4. 点击 "Commit"
5. 点击 "Push" 推送到远程仓库

#### 常用提交信息格式
```
feat: add new article on transformer architectures
fix: correct mathematical formula in random graphs post
docs: update about page with new research interests
style: improve code formatting in tools post
```

### 3. 分支管理
- **main**: 主要开发分支
- **draft**: 草稿文章分支（可选）
- **feature**: 新功能分支（可选）

## 发布流程

### 1. 本地预览
```bash
# 在终端中运行
hugo server -D
```
- 访问 `http://localhost:1313`
- 检查文章显示和格式
- 验证数学公式和代码高亮

### 2. 发布到 GitHub Pages
1. 在 Obsidian 中提交并推送更改
2. GitHub Actions 自动构建和部署
3. 等待几分钟后访问 `https://Linlin-resh.github.io/`

### 3. 发布检查清单
- [ ] 文章内容完整
- [ ] 数学公式正确显示
- [ ] 代码高亮正常
- [ ] 标签设置正确
- [ ] 图片正常显示
- [ ] 本地预览通过

## 高级技巧

### 1. 使用 Templater 创建智能模板

```javascript
---
title: "<% tp.file.title %>"
date: <% tp.date.now("YYYY-MM-DD") %>
draft: false
description: "<% tp.file.selection() %>"
tags: [<% tp.file.cursor(1) %>]
showToc: true
TocOpen: true
---

## 引言

<% tp.file.cursor(2) %>

## 主要内容

<% tp.file.cursor(3) %>

## 结论

<% tp.file.cursor(4) %>
```

### 2. 批量操作

#### 批量重命名标签
1. 使用 "Tag Wrangler" 插件
2. 选择要重命名的标签
3. 输入新标签名
4. 确认更改

#### 批量更新 Front Matter
1. 使用 "Find and Replace" 功能
2. 搜索模式：`^tags: \[.*\]`
3. 替换为新的标签格式

### 3. 内容组织

#### 使用 MOC（Map of Content）
创建索引文件来组织相关内容：

```markdown
# 网络理论文章索引

## 基础概念
- [[random-graphs]] - 随机图理论
- [[percolation-theory]] - 渗流理论

## 应用
- [[nanowire-networks]] - 纳米线网络
- [[disordered-materials]] - 无序材料

## 工具
- [[network-analysis-tools]] - 网络分析工具
- [[graph-visualization]] - 图可视化
```

## 故障排除

### 常见问题

#### 1. Hugo 命令无法识别
- 重启终端
- 检查 Hugo 安装路径
- 重新安装 Hugo

#### 2. 数学公式不显示
- 检查 KaTeX 配置
- 验证公式语法
- 清除浏览器缓存

#### 3. 图片不显示
- 检查图片路径
- 确认图片文件存在
- 验证文件名大小写

#### 4. Git 推送失败
- 检查网络连接
- 验证 GitHub 权限
- 拉取最新更改

### 获取帮助

- **Hugo 文档**: https://gohugo.io/documentation/
- **PaperMod 主题**: https://github.com/adityatelange/hugo-PaperMod
- **Obsidian 社区**: https://forum.obsidian.md/

## 总结

通过这个工作流，您可以：

1. **在 Obsidian 中舒适地写作**
2. **自动备份和版本控制**
3. **快速预览和调试**
4. **自动部署到 GitHub Pages**
5. **维护专业的研究博客**

记住：**写作是核心，工具是辅助**。专注于内容质量，让技术流程自动化处理。

---

*最后更新：2025-08-29*

