from django.http import HttpResponse, JsonResponse
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input
import tesis.settings as settings
import base64
from django.views.decorators.csrf import csrf_exempt
from tesis.settings import K
import matplotlib.pyplot as plt
import cv2


@csrf_exempt
def index(request):
    if request.method == 'POST':
        try:
            model_name = request.POST.get("model")
            image_base64 = request.POST.get("image")
            file = base64ToImage(image_base64)
            graph, model = settings.gModelObjs.get(model_name)
            pred = predict(file, model)
            makeHeatmap(file, model, graph)
            encoded_image = imageToBase64()
        except Exception as error:
            return JsonResponse({"message": str(error)}, status=500)
        return JsonResponse({
            "prediction": float(pred[0][0]),
            "heatmap": str(encoded_image)
        })
    else:
        return JsonResponse({
            "prediction": "hola"
        })


def predict(file, model):
    limpia = image.load_img(file, target_size=(320, 320))
    dataimg_limpia = np.float64(np.array(limpia))
    dataimg_limpia = np.reshape(dataimg_limpia, (1, 320, 320, 3))
    dataimg_limpia = preprocess_input(dataimg_limpia)
    return model.predict(dataimg_limpia)


def makeHeatmap(file, model, graph):
    with graph.as_default():
        img = image.load_img(file, target_size=(320, 320))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        sucia = model.output[:, 0]
        last_conv_layer = model.get_layer('block5_conv3')
        grads = K.gradients(sucia, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input],
                             [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        im = plt.imshow(heatmap)
        img = cv2.imread('imagenDecodificada.jpeg')
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        cv2.imwrite('heatmap.jpeg', superimposed_img)


def base64ToImage(text):
    imgdata = base64.b64decode(text)
    filename = 'imagenDecodificada.jpeg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

    return filename


def imageToBase64():
    filename = 'heatmap.jpeg'
    with open(filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    return encoded_string
