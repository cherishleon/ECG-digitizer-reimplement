# 发布权重

这些 .pt 文件由 Git LFS 管理，克隆后运行 git lfs pull。

| 文件 | 使用模型 | norm_eps | 标注 |
| --- | --- | ---: | --- |
| npj_newdata_weights_39.pt | UNet(3,4,depth=2,dims=[32,64,128,256,320,320,320,320]) | 1e-2 | newdata，日志最低val_loss在epoch39 |
| unet_weights_07072025.pt | 同上 | 1e-5 | 上游分割权重，本项目未训练 |
| lead_name_unet_weights_07072025.pt | 按 src/config/lead_name_unet.yml | 1e-5 | 上游导联文字权重，本项目未训练 |

第39轮验证loss为0.043514484568928063，train_loss为0.05739559382041686。norm_eps不是state_dict中的tensor，加载正确key不代表eps正确；公开 inference_newdata.yml已写1e-2。

本项目自训练权重仅发布 `npj_newdata_weights_39.pt`，与指定的本地原文件SHA-256一致：`4c515b238abe7a93c240e60184b461734b465ba16d978e6d0b57d426603108c4`。不公开本机保存目录。另两个文件是上游原有模型，不是本项目训练成果。

所有发布权重为model state_dict，没有optimizer/scheduler/scaler/RNG状态，不能用来完整恢复训练。保存目录没有留下checkpoint_last.pt，本次不能补造完整训练checkpoint。

SHA-256、字节数、tensor数量和有限值检查记录于 reports/checkpoints.json。历史其他epoch文件保留在原工作目录，inventory见 reports/local_checkpoint_inventory.json；默认不提交全部2.1GB中间权重。若确需发布全部，可以后续单独使用LFS或Release资产，并扩充清单。

这些是研究用权重，未随仓库提供独立临床测试成绩。模型架构中的legacy未使用层保留key以兼容上游文件，但被冻结。
