
ResNet(
  (conv1): LoRAConv2d(
    (base_conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  )
  (bn1): FrozenBatchNorm2d(64, eps=1e-05)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): LoRAConv2d(
        (base_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn1): FrozenBatchNorm2d(64, eps=1e-05)
      (relu): ReLU(inplace=True)
      (conv2): LoRAConv2d(
        (base_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn2): FrozenBatchNorm2d(64, eps=1e-05)
    )
    (1): BasicBlock(
      (conv1): LoRAConv2d(
        (base_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn1): FrozenBatchNorm2d(64, eps=1e-05)
      (relu): ReLU(inplace=True)
      (conv2): LoRAConv2d(
        (base_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn2): FrozenBatchNorm2d(64, eps=1e-05)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): LoRAConv2d(
        (base_conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (bn1): FrozenBatchNorm2d(128, eps=1e-05)
      (relu): ReLU(inplace=True)
      (conv2): LoRAConv2d(
        (base_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn2): FrozenBatchNorm2d(128, eps=1e-05)
      (downsample): Sequential(
        (0): LoRAConv2d(
          (base_conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        )
        (1): FrozenBatchNorm2d(128, eps=1e-05)
      )
    )
    (1): BasicBlock(
      (conv1): LoRAConv2d(
        (base_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn1): FrozenBatchNorm2d(128, eps=1e-05)
      (relu): ReLU(inplace=True)
      (conv2): LoRAConv2d(
        (base_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn2): FrozenBatchNorm2d(128, eps=1e-05)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): LoRAConv2d(
        (base_conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (bn1): FrozenBatchNorm2d(256, eps=1e-05)
      (relu): ReLU(inplace=True)
      (conv2): LoRAConv2d(
        (base_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn2): FrozenBatchNorm2d(256, eps=1e-05)
      (downsample): Sequential(
        (0): LoRAConv2d(
          (base_conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        )
        (1): FrozenBatchNorm2d(256, eps=1e-05)
      )
    )
    (1): BasicBlock(
      (conv1): LoRAConv2d(
        (base_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn1): FrozenBatchNorm2d(256, eps=1e-05)
      (relu): ReLU(inplace=True)
      (conv2): LoRAConv2d(
        (base_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn2): FrozenBatchNorm2d(256, eps=1e-05)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): LoRAConv2d(
        (base_conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (bn1): FrozenBatchNorm2d(512, eps=1e-05)
      (relu): ReLU(inplace=True)
      (conv2): LoRAConv2d(
        (base_conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn2): FrozenBatchNorm2d(512, eps=1e-05)
      (downsample): Sequential(
        (0): LoRAConv2d(
          (base_conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        )
        (1): FrozenBatchNorm2d(512, eps=1e-05)
      )
    )
    (1): BasicBlock(
      (conv1): LoRAConv2d(
        (base_conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn1): FrozenBatchNorm2d(512, eps=1e-05)
      (relu): ReLU(inplace=True)
      (conv2): LoRAConv2d(
        (base_conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (bn2): FrozenBatchNorm2d(512, eps=1e-05)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=128, bias=True)
)