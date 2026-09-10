# GitHub 首次发布指南

## 1. 提交前检查

```powershell
git status
git diff
git diff --stat
```

确认数据、模型权重、账号令牌和私人文件没有出现在待提交列表中。

## 2. 创建第一个提交

```powershell
git add .
git commit -m "chore: initialize small object detection lab"
```

提交信息建议使用 `feat:`、`fix:`、`docs:`、`test:`、`chore:` 前缀。

## 3. 在 GitHub 网页创建仓库

1. 打开 GitHub，点击 `New repository`；
2. 仓库名填写 `small-object-detection-lab`；
3. 不要勾选自动创建 README、`.gitignore` 或 License，因为本地已经存在；
4. 根据数据是否公开决定仓库为 Public 或 Private；
5. 创建后复制远程仓库地址。

## 4. 连接远程仓库并推送

HTTPS 示例：

```powershell
git remote add origin https://github.com/<你的用户名>/small-object-detection-lab.git
git push -u origin main
```

如果启用了 GitHub 两步验证，HTTPS 密码需要替换成 Personal Access Token，不要使用账号登录密码。令牌应设置为最小权限，并且不要写入代码、README 或脚本。

SSH 示例：

```powershell
git remote add origin git@github.com:<你的用户名>/small-object-detection-lab.git
git push -u origin main
```

## 5. 日常开发流程

```powershell
git switch -c feat/p2-head
git status
git add <明确修改的文件>
git commit -m "feat: add P2 detection head"
git push -u origin feat/p2-head
```

然后在 GitHub 创建 Pull Request。即使是个人项目，也建议用分支和 PR 留下研究轨迹。

## 6. 发布版本

完成一组可复现实验后：

```powershell
git tag -a v0.2.0 -m "Reproducible baseline"
git push origin v0.2.0
```

随后在 GitHub 的 Releases 页面创建 Release，写清：

- 数据集和版本；
- 环境与硬件；
- 实验命令；
- 主要指标；
- 已知限制。

## 7. 常见错误

- `src` 不存在：确认自己在项目根目录运行命令；
- 推送被拒绝：先 `git pull --rebase origin main` 再提交；
- 文件过大：使用 `.gitignore` 排除，已提交的大文件不要继续叠加版本；
- 误传令牌：立即在 GitHub 撤销令牌，并从历史中彻底清除，不能只删除当前文件。
