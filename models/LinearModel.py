from __future__ import print_function

import torch.nn as nn
from torch.autograd import Function

class alex_net_complete(nn.Module):
    def __init__(self, image_embedding_model, classifier=None):
        super(alex_net_complete, self).__init__()
        self.image_embedding_model = image_embedding_model
        self.classifier = classifier

    def forward(self, input_image_variable, age_id=None):
        image_embedding = self.image_embedding_model(input_image_variable, age_id)
        if self.classifier is None:
            logit_res = image_embedding
        else:
            logit_res = self.classifier(image_embedding)

        return logit_res

class alex_net_mmse(nn.Module):
    def __init__(self, image_embedding_model, regressor):
        super(alex_net_mmse, self).__init__()
        self.image_embedding_model = image_embedding_model
        self.regressor = regressor

    def forward(self, input_image_variable):
        image_embedding = self.image_embedding_model(input_image_variable)
        output = self.regressor(image_embedding)

        return output

class alex_net_seg(nn.Module):
    def __init__(self, image_embedding_model, deconv_layers):
        super(alex_net_seg, self).__init__()
        self.image_embedding_model = image_embedding_model
        self.deconv_layers = deconv_layers

    def forward(self, input_image_variable):
        image_embedding = self.image_embedding_model(input_image_variable)
        output = self.deconv_layers(image_embedding)

        return output

"""
Gradient reversal layer from https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/6
"""
class GradReverse(Function):

    def __init__(self, lambd=-1.0):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

class adversarial_linear_model(nn.Module):
    def __init__(self, image_embedding_model, classifier,
            lambda_grl):
        super(adversarial_linear_model, self).__init__()
        self.image_embedding_model = image_embedding_model
        self.reversal_layer = GradReverse(lambda_grl)
        self.classifier = classifier

    def get_lambda(self):
        return(self.reversal_layer.lambd)

    def set_lambda(self, lambd):
        self.reversal_layer.lambd = lambd

    def forward(self, input_image_variable,age_id=None):
        image_embedding = self.image_embedding_model(input_image_variable, age_id)

        classifier_input = self.reversal_layer(image_embedding)
        logit_res = self.classifier(classifier_input)

        return logit_res
