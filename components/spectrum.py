import yaml
from transformers import TrainerCallback


#冻结参数callback
class FreezeCallback(TrainerCallback):
    def __init__(self, student_model, unfrozen_layers_file1, unfrozen_layers_file2, unfrozen_layers_file3, unfrozen_layers_file4):
        self.student_model = student_model
        self.unfrozen_layers_file1 = unfrozen_layers_file1
        self.unfrozen_layers_file2 = unfrozen_layers_file2
        self.unfrozen_layers_file3 = unfrozen_layers_file3
        self.unfrozen_layers_file4 = unfrozen_layers_file4
        # 加载需要解冻的层（spectrum配置）
        with open(unfrozen_layers_file1, 'r') as file1:
            self.unfrozen_layers1 = yaml.safe_load(file1)['unfrozen_parameters'] #20%
        with open(unfrozen_layers_file2, 'r') as file2:
            self.unfrozen_layers2 = yaml.safe_load(file2)['unfrozen_parameters'] #40%
        with open(unfrozen_layers_file3, 'r') as file3:
            self.unfrozen_layers3 = yaml.safe_load(file3)['unfrozen_parameters'] #60%
        with open(unfrozen_layers_file4, 'r') as file4:
            self.unfrozen_layers4 = yaml.safe_load(file4)['unfrozen_parameters'] #80%
        # 初始状态：所有参数可训练（前4个epoch）

    def set_spectrum_freeze1(self):
        """根据spectrum配置冻结参数（仅保留指定层可训练）"""
        for name, param in self.student_model.named_parameters():
            if any(layer in name for layer in self.unfrozen_layers1):
                param.requires_grad = True  # 解冻指定层
            else:
                param.requires_grad = False  # 冻结其他层
    
    def set_spectrum_freeze2(self):
        """根据spectrum配置冻结参数（仅保留指定层可训练）"""
        for name, param in self.student_model.named_parameters():
            if any(layer in name for layer in self.unfrozen_layers2):
                param.requires_grad = True  # 解冻指定层
            else:
                param.requires_grad = False  # 冻结其他层

    def set_spectrum_freeze3(self):
        """根据spectrum配置冻结参数（仅保留指定层可训练）"""
        for name, param in self.student_model.named_parameters():
            if any(layer in name for layer in self.unfrozen_layers3):
                param.requires_grad = True  # 解冻指定层
            else:
                param.requires_grad = False  # 冻结其他层

    def set_spectrum_freeze4(self):
        """根据spectrum配置冻结参数（仅保留指定层可训练）"""
        for name, param in self.student_model.named_parameters():
            if any(layer in name for layer in self.unfrozen_layers4):
                param.requires_grad = True  # 解冻指定层
            else:
                param.requires_grad = False  # 冻结其他层

    def on_epoch_begin(self, args, state, control,** kwargs):
        # state.epoch是从0开始的浮点数（如3.0表示第4个epoch开始）
        current_epoch = int(state.epoch)  # 转换为整数（0→第1个epoch，4→第5个epoch）
        
        if current_epoch < 1:  # 前4个epoch（0-3对应第1-4个epoch）：全参数训练
            #不冻结任何参数
            if int(state.epoch) == 0:  # 只在首次epoch开始时打印一次
                print("第1个epoch，不冻结参数")
        elif current_epoch == 1:  # 第2个epoch：启用20%解冻
            self.set_spectrum_freeze1()
            print(f"第{current_epoch+1}个epoch：启用spectrum冻结参数训练，解冻20%层")
        elif current_epoch == 2:  # 第3个epoch：启用40%解冻
            self.set_spectrum_freeze2()
            print(f"第{current_epoch+1}个epoch：启用spectrum冻结参数训练，解冻40%层")
        elif current_epoch == 3:  # 第4个epoch：启用60%解冻
            self.set_spectrum_freeze3()
            print(f"第{current_epoch+1}个epoch：启用spectrum冻结参数训练，解冻60%层")
        elif current_epoch == 4:  # 第5个epoch：启用80%解冻
            self.set_spectrum_freeze4()
            print(f"第{current_epoch+1}个epoch：启用spectrum冻结参数训练，解冻80%层")