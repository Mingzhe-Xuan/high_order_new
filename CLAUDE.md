# High Order Project Guidance

## 项目目的

本项目用于材料体系的高阶等变/不变图神经网络建模，围绕原子结构、邻接关系、标量性质与张量性质数据，进行自监督训练、标量性质预测和张量性质预测。

## 大致框架流程

1. `main.py` 是主要入口，负责解析/汇总训练参数、设置随机种子、构建数据加载器并调度训练与测试流程。
2. `data/` 负责数据集、数据库读取和 dataloader 构建，包括 Materials Project、Alexandria、标量性质数据和张量性质数据。
3. `src/model/` 负责模型结构，包括 embedding、invariant layer、equivariant layer、tensor product、readout、MLP 和相关 e3nn 工具。
4. `src/train_test/` 负责训练、测试、checkpoint、指标保存和可视化。
5. `src/model/Jd.pt` 与 `src/model/z_rot_indices_lmax12.pt` 是模型运行所需的常量文件，应保留在仓库中。

## 修改规范

1. 每次修改代码后，都必须运行全面的单元测试；如果测试未通过，必须继续迭代修改，直到单元测试通过。
2. 单元测试通过后，不要直接提交 commit；必须先向用户申请是否提交 commit。
3. 不要提交生成文件、缓存文件或本地数据文件，例如 `__pycache__/`、`*.pyc`、数据库文件、训练输出、checkpoint、临时日志等。
4. 修改依赖时应同步更新依赖文件，并说明 CPU/CUDA、PyTorch、PyG 扩展之间的兼容关系。
5. 保持导入路径稳定，优先使用项目根目录下的 `src`、`data` 包导入或包内相对导入，避免依赖临时 `sys.path` 修改。
6. 不要全局屏蔽 warning；训练和数据加载相关 warning 应保留，便于定位数值、依赖和数据问题。
