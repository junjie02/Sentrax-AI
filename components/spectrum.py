import yaml
from transformers import TrainerCallback


#冻结参数callback
class FreezeCallback(TrainerCallback):
    def __init__(self, student_model, unfrozen_layers_file):
        self.student_model = student_model
        self.unfrozen_layers_file = unfrozen_layers_file
        # 加载需要解冻的层（spectrum配置）
        with open(unfrozen_layers_file, 'r') as file:
            self.unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
        # 初始状态：所有参数可训练（前4个epoch）

    def set_spectrum_freeze(self):
        """根据spectrum配置冻结参数（仅保留指定层可训练）"""
        for name, param in self.student_model.named_parameters():
            if any(layer in name for layer in self.unfrozen_layers):
                param.requires_grad = True  # 解冻指定层
            else:
                param.requires_grad = False  # 冻结其他层

    def on_epoch_begin(self, args, state, control,** kwargs):
        # state.epoch是从0开始的浮点数（如3.0表示第4个epoch开始）
        current_epoch = int(state.epoch)  # 转换为整数（0→第1个epoch，4→第5个epoch）
        
        if current_epoch < 1:  # 前4个epoch（0-3对应第1-4个epoch）：全参数训练
            #不冻结任何参数
            if int(state.epoch) == 0:  # 只在首次epoch开始时打印一次
                print("前1个epoch：所有参数可训练（不冻结）")
        else:  # 第5个epoch及以后：启用spectrum冻结
            self.set_spectrum_freeze()
            print(f"第{current_epoch+1}个epoch：启用spectrum参数冻结逻辑")