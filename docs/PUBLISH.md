# GitHub 发布准备

拟使用仓库名 `ECG-digitizer`，描述“复现 Open-ECG-Digitizer，并加入合成数据、多 GPU 训练和数值适配”。发布目录与原工作树分开，不沿用旧项目的 remote，也不修改旧仓库未提交变更。

用户已确认先创建为 **Private（私有）**；未经进一步确认，不改为公开。

## 已整理

- 源码、两份可移植Slurm脚本、新数据生成工具与完整README。
- 第39轮模型、原始分割模型、原始导联文字模型；所有.pt用LFS。
- 数据来源与生成脚本说明、汇总生成报告、训练曲线、模型hash及文件盘点。
- LICENSE / 上游署名；不复制原自动发布workflow和私有输入。
- README 单独编写，直接说明复现原项目和自己的修改；不复制原仓库 README。
- 已脱敏用户名、集群账号、绝对数据路径、内部配置标签和任务编号；shell 脚本采用通用资源参数及环境变量。
- 数据来源/独立test缺失/历史4卡结果未证实/恢复训练局限均在README说明。

## 大文件

GitHub普通Git对大文件有大小限制；此仓库3份模型虽各小于100MiB，仍用LFS，避免模型进普通Git历史。模型共约200MiB，LFS账户存储/下载额度以GitHub当前政策为准。
参考：[Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

.gitattributes已包含模型规则；.gitignore不忽略checkpoints中的发布.pt，但忽略outputs/data/logs/cache等。
自建PNG/mask、源波形及逐条manifest均不进入提交，只发布数据说明、生成脚本和统计。自训练模型仅发布用户指定的npj_newdata_weights_39.pt，其余epoch权重保留本地。原项目必需的固定增强资源、小型测试样例及上游模型保留，来源单独标注。

## 准备初始提交

```bash
git status --short
git lfs ls-files
git diff --cached --check
git commit -m "Add ECG digitizer reproduction, synthetic data pipeline and DDP training"
```

如果Git未配置提交者，先按自己的身份设置仓库local user.name / user.email。不要把示例邮箱当作真实身份。

## 创建和推送

新仓库名为ECG-digitizer，可见性已确认先设为Private（私有）；发布前仍需确认GitHub账号/组织并完成认证。当前没有为此副本配置remote。不要直接推到原工作树已有的旧仓库。

在GitHub创建Private空仓库后：

```bash
git remote add origin git@github.com:<YOUR_USERNAME>/ECG-digitizer.git
git push -u origin main
```

也可以用已登录的GitHub CLI创建，但执行前需确认账号和public/private。模型LFS对象会随push上传。克隆页面和README里的<YOUR_USERNAME>需要在账号确定后替换。

## 替换包含敏感信息的旧仓库

删除前需要认证并核实旧仓库的完整 owner/repository，不能仅凭未登录查询的404判断仓库已经删除。删除远程仓库不会删除原本地研究目录。

新仓库从本目录创建全新的初始提交，不复制旧仓库的`.git`、分支或标签；报告也不发布旧项目的commit引用。不要从旧工作树再次push，以免带回含敏感信息的历史。

如果旧仓库泄露的是token、密码或密钥，先撤销或轮换。别人已有的clone/fork不会因为删除远程而消失，缓存或其他引用可能还需要GitHub Support处理。参见[GitHub敏感数据清理说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)。

## 验证范围

reports/release_validation.json记录本地已执行检查。发布准备在无CUDA设备的节点进行，能检查Python语法、配置、CPU单元测试、checkpoint有限值和加载、双进程CPU/Gloo DDP的两次优化器更新；没有在这里重跑A100训练或完整临床数字化评测。

历史恢复训练文件缺失、自定义优化器状态恢复尚未完整；不能写成精确断点恢复复现。本项目定位为研究复现与扩展，指标来源为本地40轮合成数据验证。
