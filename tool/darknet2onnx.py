import sys
import torch
from tool.darknet2pytorch import Darknet


def transform_to_onnx(cfgfile, weightfile, batch_size=1):
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    input_names = ["input"]
    output_names = ['boxes', 'confs']

    if dynamic:
        x = torch.randn((1, 3, model.height, model.width), requires_grad=True)
        onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(model.height, model.width)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, model.height, model.width)
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name


if __name__ == '__main__':
    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        transform_to_onnx(cfgfile, weightfile)
    elif len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        batch_size = int(sys.argv[3])
        transform_to_onnx(cfgfile, weightfile, batch_size)
    elif len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        batch_size = int(sys.argv[3])
        dynamic = True if sys.argv[4] == 'True' else False
        transform_to_onnx(cfgfile, weightfile, batch_size, dynamic)
    else:
        print('Please execute this script this way:\n')
        print('  python darknet2onnx.py <cfgFile> <weightFile>')
        print('or')
        print('  python darknet2onnx.py <cfgFile> <weightFile> <batchSize>')
