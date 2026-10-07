# GitHub 上传说明（中文版）

这个文件夹已经是可以上传到 GitHub 的项目目录。

## 推荐做法：网页上传

1. 在 GitHub 新建仓库，建议命名：
   `AI4S-ThinFilm-2024270023`
2. Visibility 选 `Public`（如果老师要求可访问）。
3. 创建仓库后进入仓库首页。
4. 点击 `Add file` → `Upload files`。
5. **先解压 ZIP**，再把解压后的项目文件和文件夹拖到上传区域。
6. 不要只上传一个 ZIP 文件；老师需要直接看到 `README.md`、源码和脚本。
7. Commit message 可填：`Initial AI4S thin-film project upload`
8. 点击提交。
9. 上传完成后，把浏览器地址栏中的真实仓库链接复制到论文的 `Data and Code Availability`。

## 上传后仓库首页应至少能看到

- `README.md`
- `requirements.txt`
- `config.py`
- `run_all.py`
- `src/`
- `scripts/`
- `results/`
- `figures/`

本包还附带 `data/` 和 `models/`，方便展示已经复现的结果。

## 本项目的个人参数

- λtarget = 680 nm
- seed = 270023
- design_seed = 270024

## 注意

课程说明只写“按目标波长反射率排序”，没有明确写最大化还是最小化。本包默认采用**最大化 680 nm 反射率**。如果老师课堂要求最小化，请把 `config.py` 中：

`DESIGN_OBJECTIVE = "maximize"`

改为：

`DESIGN_OBJECTIVE = "minimize"`

然后重新运行 `python run_all.py`，并同步更新论文结果。
