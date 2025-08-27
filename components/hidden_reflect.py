import torch

#将学生模型的隐藏层适配到教师模型的隐藏层
class MultiLayerAdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, top_k=4, dtype=torch.bfloat16):
        super().__init__()
        self.projections = torch.nn.ModuleList([#创建多层线性投影
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers, top_k)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers, top_k):#具体映射层
        mapping = {}
        for i in range(num_student_layers - top_k, num_student_layers):
            j = num_teacher_layers - top_k + i - (num_student_layers - top_k)
            if j < num_teacher_layers:
                mapping[i] = j
        return mapping

    def forward(self, student_hidden_states):#创建适配层
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i not in self.layer_mapping:
                continue  # 跳过未映射的层
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states