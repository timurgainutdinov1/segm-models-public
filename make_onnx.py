import argparse
import torch
import cv2
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_toolbelt.utils.torch_utils import image_to_tensor
from torchsummary import summary
from model import get_model
from torch import nn
import torch.onnx
import onnx


def read_image(filename):
    image = cv2.imread(filename)
    if image is None:
        raise IOError("Cannot read " + filename)
    return image


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def visualize(**images):
    """Функция для визуализации данных, располагает изображения в ряд"""
    n = len(images)
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Run model in evaluation mode')
    parser.add_argument("-c", type=str, help="checkpoint path", required=True)
    parser.add_argument("-d", "--data-dir", type=str, help="path where dataset is placed")
    parser.add_argument("-o", "--output-dir", type=str, help="path where to write output, if isn't provided")
    parser.add_argument("-t", "--test", type=bool, default=True, help="calculate accuracy and iou")
    parser.add_argument("-w", "--workers", type=int, default=6, help="default=6")
    parser.add_argument("--device", type=str, default="cpu", help="default=cuda, or cpu")
    parser.add_argument("--tile-size", default=512, help="default=512")
    args = parser.parse_args()
    print("Passed arguments: ", args)

    file_path = r'D:\Vector_data\Water\test_dataset\val\images\N-34-140-A-b-4-2_5.tif'
    output_path = r'D:\Vector_data\Water\test_dataset\val\prediction.tif'
    tile_size = args.tile_size
    device = args.device
    step = tile_size

    model_name = 'unet'
    encoder_name = 'efficientnet-b7'
    encoder_weights = 'imagenet'
    activation = 'sigmoid'
    model = get_model(model_name=model_name, encoder_name=encoder_name, encoder_weights=encoder_weights,
                      activation=activation)
    model.encoder.set_swish(memory_efficient=False)

    # for printing
    # model.to('cuda')
    # # print(model)
    # input = torch.tensor((3, 512, 512))
    # summary(model, (3, 512, 512))
    # exit()

    model = nn.DataParallel(model)
    checkpoint = torch.load(args.c)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module.to(device) # убираем nn.DataParallel т.к. с ним не считается на cpu
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, 512, 512, requires_grad=True)
    torch_out = model(x)
    #print(f'torch out {torch_out}')

    # Этот код работает:
    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "water.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load("water.onnx")
    onnx.checker.check_model(onnx_model)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession("onnx/water.onnx")

    def to_numpy(tensor):
        # return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        return tensor.detach().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"{torch_out.shape}")
    print(f"{ort_outs[0].shape}")
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # image = read_image(file_path)
    # image_copy = image
    # w, h, z = image.shape
    # print(f'image shape is{image.shape}')
    # res_shape = (math.ceil(float(w) / tile_size)*tile_size, math.ceil(float(h) / tile_size)*tile_size, z)
    # print(f'res shape is{res_shape}')
    # w_new = res_shape[0]
    # h_new = res_shape[1]
    # res = np.zeros(res_shape, dtype=np.float32)
    # res[0:w, 0:h, :] = image
    # image = res
    #
    # image = normalize(image)
    # tensor = image_to_tensor(image)
    # tensor = tensor.reshape((1,) + tensor.shape)
    # tensor.to(device)
    #
    # res = np.zeros((1, w_new, h_new), dtype=np.float32)
    # i = j = 0
    #
    # with tqdm(total=(w_new // step) * (h_new // step)) as pbar:
    #     while i + tile_size <= w_new:
    #         j = 0
    #         while j + tile_size <= h_new:
    #             # print(f'i is {i} j is {j}')
    #             frag = tensor[:, :, i:i+tile_size, j:j+tile_size]
    #             # print(frag.shape)
    #             out = model.forward(frag)
    #             res[:, i:i+tile_size, j:j+tile_size] = out.detach().numpy()
    #             j += step
    #             pbar.update(1)
    #         i += step
    #
    # res = res.reshape((w_new, h_new))
    # res = res[0:w, 0:h]
    #
    # max_val = np.max(res)
    # min_val = np.min(res)
    #
    # res = 255.0*(res - min_val)/(max_val-min_val)
    # print(f'result shape is {res.shape}')
    # # print(f'max val is {max} min val is {min}')
    # # img = Image.fromarray(np.uint8(res))
    # # img.save(output_path)
    # res = np.uint8(res)
    # cv2.imwrite(output_path, res)
    #
    # visualize(
    #     image=image_copy,
    #     # image=image.permute(1, 2, 0),
    #     predicted=res,
    # )


if __name__ == '__main__':
    main()
