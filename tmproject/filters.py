import cv2
import numpy as np

# Descripció dels filtres puntuals
FILTERS_DESCRIPTION = {
    'binarization': 'Filtre puntual de binarització utilitzant el valor llindar indicat.',
    'negative': 'Filtre puntual negatiu sobre la imatge.',
    'hsv': 'Converteix la imatge a l\'espai de colors HSV (Hue, Saturation, Value).',
    'sepia': 'Aplica un efecte sèpia a la imatge, emulant un aspecte antic.',
    'contrast_stretching': 'Amplia el rang dinàmic dels valors de píxel per a millorar el contrast de la imatge.'
}

# Descripció dels filtres convolucionals
CONV_FILTERS_DESCRIPTION = {
    'averaging': 'Filtre convolucional d’averaging en zones de value x value.',
    'sobel': 'Operador Sobel per a la detecció de contorns en la imatge utilitzant la mida del kernel indicada.',
    'gaussian': 'Filtre convolucional de suavitzat gaussià per a la reducció del soroll i les irregularitats de la '
                'imatge utilitzant la mida del kernel indicada.',
    'sharpening': 'Filtre convolucional d’enfocament per a realçar els detalls de la imatge i augmentar el contrast '
                  'local utilitzant la mida del kernel indicada.',
    'laplacian': 'Filtre convolucional Laplacian per a ressaltar detalls i detectar contorns utilitzant la mida del '
                 'kernel indicada.',
}

def apply_filters(img_array, filters):
    """
        Aplica filtres puntuals a una imatge.
        Arguments: img_array (numpy.ndarray): La imatge a la qual s'aplicaran els filtres. filters (dict): Un
        diccionari que conté els noms dels filtres com a claus i els seus paràmetres, si en té, com a valors.
        Returns:
        numpy.ndarray: La imatge després d'aplicar els filtres.
        """
    for filter_name, filter_param in filters.items():
        if filter_name == 'binarization':
            threshold_value = int(filter_param) if filter_param is not None else 50
            img_array = np.where(img_array > threshold_value, 255, 0).astype(np.uint8)
        elif filter_name == 'negative':
            img_array = 255 - img_array
        elif filter_name == 'contrast_stretching':
            img_array = ((img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))) * 255
            img_array = img_array.astype(np.uint8)
        elif filter_name == 'hsv':
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        elif filter_name == 'sepia':
            sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]])
            img_array = img_array @ sepia_matrix.T
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return img_array


def apply_conv_filters(img_array, conv_filters):
    """
    Aplica una sèrie de filtres convolucionals a una imatge representada com una matriu de píxels.

    Args:
        img_array (numpy.ndarray): La matriu que representa la imatge.
        conv_filters (dict): Un diccionari que conté els noms dels filtres convolucionals com a claus i els valors
                             dels paràmetres dels filtres com a valors.

    Returns:
        numpy.ndarray: La matriu de la imatge després d'aplicar els filtres convolucionals.
    """
    for filter_name, filter_param in conv_filters.items():
        if filter_name == 'averaging':
            kernel_size = int(filter_param) if filter_param is not None else 3
            img_array = cv2.blur(img_array, (kernel_size, kernel_size))
        elif filter_name == 'sobel':
            kernel_size = int(filter_param) if filter_param is not None else 3
            grad_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=kernel_size)
            grad_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=kernel_size)
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            img_array = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif filter_name == 'sharpening':
            kernel_size = int(filter_param) if filter_param is not None else 3
            kernel = np.ones((kernel_size, kernel_size), np.float32) * -1
            kernel[kernel_size // 2, kernel_size // 2] = kernel_size * kernel_size
            img_array = cv2.filter2D(img_array, -1, kernel)
        elif filter_name == 'gaussian':
            kernel_size = int(filter_param) if filter_param is not None else 3
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        elif filter_name == 'laplacian':
            kernel_size = int(filter_param) if filter_param is not None else 1
            img_array = cv2.Laplacian(img_array, cv2.CV_64F, ksize=kernel_size)
            img_array = np.uint8(np.absolute(img_array))
    return img_array
