你现在是我的**网站搭建与自动化助手**。请严格按以下需求一步步完成实现，并在每步给出：要执行的命令、文件变更、关键配置、可复制的代码片段，以及“完成判定标准（验收点）”。遇到可选项，请给出推荐与理由。

## 目标与定位

* 站点类型：**知识博客为主**（短文、随笔、小主题为主，科研读者可读）
* 受众：**科研圈 / AI for Science / 材料科学 / Graph Theory**
* 写作体验：**Obsidian 写 Markdown → Git 同步 → 自动部署到 GitHub Pages**
* 技术栈：**Hugo（静态生成） + PaperMod 主题 + GitHub Actions（CI/CD）**
* 语言与地区：界面英文为主，但支持中英混写、公式、代码高亮
* SEO 与传播：RSS、站点地图、favicon、Open Graph、Twitter Card 
* 评论与讨论：**Giscus（GitHub Discussions）**（如不便可先留存配置位）

## 我提供的变量（请在生成配置时引用）

* GitHub 用户名：`Linlin-resh`
* 站点仓库名：`Linlin-resh.github.io`（
* 站点标题：`Notes on AI4Science & Graph Theory`
* 站点描述：`Thoughts on Partially Disordered Networks, Silver Nanowires, Graph Theory, and Machine Learning` 
* 自定义域名（可选）：`{{CUSTOM_DOMAIN}}`（没有就留空）
* 作者信息：`{{AUTHOR_NAME}}`、`{{AUTHOR_EMAIL}}`（可选）
* 社交链接（可选）：Google Scholar、ORCID、GitHub、X/微博 等 URL
* 深色模式默认：`{{dark|light|auto}}`

## 站点结构（Hugo 内容模型）

* 首页：最近文章瀑布流 + 置顶若干文章（可选）
* 导航：Home / Blog / Tags / About / Search
* 内容分类（用 Section 或仅使用 tags 均可，推荐 tags 主导）：

  * **Reading Notes**（文献/书籍短评）
  * **Ideas & Thoughts**（随想/灵感）
  * **Research Progress**（阶段性总结）
  * **Tools & Code**（小技巧与片段）
* 文章元信息与模板（Front Matter 必含）：`title`, `date`, `tags`, `draft`, `description`
* 必备页面：About / 404 / 搜索页（本地索引或第三方）/ RSS / sitemap.xml / robots.txt

## 功能清单（必须实现或保留配置位）

1. **数学公式**：KaTeX（优先）或 MathJax（仅选其一，默认 KaTeX）
2. **代码高亮**：Hugo Chroma（支持常见语言；行号可选）
3. **图片优化**：建议 webp 与等比缩放，保留原图链接；支持图片标题与说明
4. **暗色模式**：跟随系统或可切换
5. **SEO**：`<meta>` 基础、OG、Twitter、站点地图、robots、canonical
6. **RSS**：为博客文章启用
7. **搜索**：本地 lunr/fuse 索引或 Algolia（先默认本地）
8. **评论**：Giscus（可用 env 或 config 中开关）
9. **统计**：可留 GA4/umami/plausible 配置位（先关闭）
10. **多端适配**：移动端菜单、排版、触控优化

## 目录与文件（请输出最终结构与说明）

* 根目录：Hugo 站点结构（`/content`, `/layouts`, `/static`, `/themes/PaperMod`, `/config.*` 等）
* **示例文章**：请生成 4 篇示例 Markdown（各 300–800 字，含中英文与公式/代码示例）

  * `content/posts/reading-notes-newman-ch3.md`
  * `content/posts/idea-transformer-for-local-graph.md`
  * `content/posts/progress-2025-08.md`
  * `content/posts/tool-snippets-structural-gt.md`
* About 页面：`content/about/_index.md`
* 搜索页与脚本：如需 `static/` 下放置索引或 js，请给出文件
* 站点图标：生成占位 favicon 与 `static/favicon.ico`

## 配置与脚本（请给出完整可用内容）

1. **Hugo 初始化与主题引入命令**
2. **`config.toml`（或 `config.yaml`）完整示例**：包含 PaperMod 典型项、菜单、社交、KaTeX、RSS、Sitemap、robots、评论占位、代码高亮、分页、显示摘要长度等
3. **GitHub Actions**：`/.github/workflows/hugo.yml`，要求：

   * 触发：`push` 到 `main`（或 `master`）
   * 版本固定：hugo-extended 稳定版本
   * 构建产物发布到 `gh-pages` 分支（启用 Pages）
   * 构建缓存（modules/node）优化
4. **CNAME**：若 `{{CUSTOM_DOMAIN}}` 非空，则在 `static/CNAME` 写入域名
5. **robots.txt 与 sitemap**：提供默认策略
6. **本地开发命令**：`hugo server -D`（草稿可见）

## Obsidian 工作流（请输出明确操作步骤）

* 在 Obsidian 中将站点仓库设为 Vault
* 推荐安装与配置 **Obsidian Git** 插件：定时/手动 commit & push
* 新建文章时的 Front Matter 模板（提供可复制片段）
* 图片粘贴策略：相对路径、统一 `static/images`；给出 Obsidian 设置建议
* 写作到发布的“最短路径”：创建 → 写作 → push → 自动部署

## 验收标准（每项请给出自检方法）

* 本地可运行并可见示例文章、数学公式、代码高亮与暗色模式
* GitHub Actions 首次运行成功，`gh-pages` 分支生成，Pages 正常访问
* 搜索功能可用；Tags/Archive/Recent Posts 可访问
* RSS 输出正常（校验 feed）
* 站点地图与 robots 可访问
* Giscus 若未配置 token，前端应优雅降级（不报错）

## 可复制的关键片段（请务必给出）

* `config.toml` 全文（含 PaperMod 推荐项）
* `hugo.yml`（GitHub Actions）全文
* 文章 Front Matter 模板
* KaTeX 支持示例（文章中如何写公式）
* 代码块示例（三引号+语言）
* robots.txt/sitemap 配置
* 示例导航菜单配置
* Giscus 配置位（注释说明如何填）

请开始执行，并先给出**整体步骤总览清单**，然后逐步输出可复制的命令与文件内容。每完成一部分，请列出“本步完成判定点”。在我回复“继续”之前，不要跳到下一大步。

---

## 🧩 附：我偏好的默认值（可直接用在示例里）

* 主题：`adityatelange/hugo-PaperMod`
* 分支：主分支 `main`，发布分支 `gh-pages`
* 站点 URL：`https://{{YOUR_GITHUB_USERNAME}}.github.io/`
* 暗色模式：`auto`
* 分页：`10`
* 代码高亮：行号关闭、复制按钮开启
* KaTeX：开启，文章中通过 `$$ ... $$` 与 `$ ... $` 书写
* 评论：默认关闭（保留配置位）

---

### 你可以直接把上面的整段提示词丢给 AI 使用。

如果你想，我也可以直接把\*\*`config.toml` 与 GitHub Actions 工作流\*\*按你的变量生成好，连同 4 篇示例文章一次性给你。需要的话回我：

> “用这些变量生成：用户名=…，仓库=…，标题=…，描述=…（是否自定义域名）”
