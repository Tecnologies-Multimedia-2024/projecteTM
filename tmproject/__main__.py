import threading
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import click
import cv2
import imageio
import numpy as np
from PIL import Image

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

# Mètode per aplicar els filtres puntuals a la imatge
def apply_filters(img_array, filters):
    """
        Aplica filtres puntuals a una imatge.

        Arguments: img_array (numpy.ndarray): La imatge a la qual s'aplicaran els filtres. filters (dict): Un
        diccionari que conté els noms dels filtres com a claus i els seus paràmetres, si en té, com a valors.

        Returns:
        numpy.ndarray: La imatge després d'aplicar els filtres.
        """
    for filter_name, filter_param in filters.items():
        # Aplica el filtre de binarització
        if filter_name == 'binarization':
            threshold_value = int(filter_param) if filter_param is not None else 50
            for i in range(3):  # Per cada canal R, G, B
                img_array[:, :, i] = np.where(img_array[:, :, i] > threshold_value, 255, 0)
            # Converteix l'array a uint8
            img_array = img_array.astype(np.uint8)
        # Aplica el filtre de negative
        elif filter_name == 'negative':
            img_array = 255 - img_array
        # Aplica el filtre de contrast_stretching
        elif filter_name == 'contrast_stretching':
            for i in range(3):  # Per cada canal R, G, B
                img_array[:, :, i] = ((img_array[:, :, i] - np.min(img_array[:, :, i])) / (np.max(img_array[:, :, i]) - np.min(img_array[:, :, i]))) * 255
            img_array = img_array.astype(np.uint8)
        # Aplica el filtre de hsv
        elif filter_name == 'hsv':
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HLS)
        # Aplica el filtre de sepia
        elif filter_name == 'sepia':
            sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]])
            img_array = img_array @ sepia_matrix.T
            img_array = np.clip(img_array, 0, 255)
            img_array = img_array.astype(np.uint8)
    return img_array


def apply_conv_filters(img_array, conv_filters):
    """
        Aplica filtres convolucionals a una imatge.

        Arguments:
        img_array (numpy.ndarray): La imatge d'entrada.
        conv_filters (dict): Un diccionari amb els noms dels filtres com a claus i la mida dels kernels com a valors.

        Returns:
        numpy.ndarray: La imatge després d'aplicar els filtres convolucionals.
    """
    for filter_name, filter_param in conv_filters.items():
        # Passar la imatge a escala de grisos si no ho està
        # Aplica el filtre d'averaging
        if filter_name == 'averaging':
            kernel_size = int(filter_param) if filter_param is not None else 3
            for i in range(3):
                img_array[:, :, i] = cv2.blur(img_array[:, :, i], (kernel_size, kernel_size))

            img_array = cv2.blur(img_array, (kernel_size, kernel_size))
        # Aplica el filtre de sobel
        elif filter_name == 'sobel':
            kernel_size = int(filter_param) if filter_param is not None else 3
            for i in range(3):
                grad_x = cv2.Sobel(img_array[:, :, i], cv2.CV_64F, 1, 0, ksize=kernel_size)
                grad_y = cv2.Sobel(img_array[:, :, i], cv2.CV_64F, 0, 1, ksize=kernel_size)
                magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
                img_array[:, :, i] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Aplica el filtre de sharpening
        elif filter_name == 'sharpening':
            kernel_size = int(filter_param) if filter_param is not None else 3
            kernel = np.ones((kernel_size, kernel_size), np.float32) * -1
            center_value = kernel_size * kernel_size
            kernel[int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)] = center_value
            for i in range(3):
                img_array[:, :, i] = cv2.filter2D(img_array[:, :, i], -1, kernel)
        # Aplica el filtre gaussià
        elif filter_name == 'gaussian':
            kernel_size = int(filter_param) if filter_param is not None else 3
            for i in range(3):
                img_array[:, :, i] = cv2.GaussianBlur(img_array[:, :, i], (kernel_size, kernel_size), 0)
        # Aplica el filtre Laplacià
        elif filter_name == 'laplacian':
            kernel_size = int(filter_param) if filter_param is not None else 1
            for i in range(3):
                img_array[:, :, i] = cv2.Laplacian(img_array[:, :, i], cv2.CV_64F, ksize=kernel_size)
                img_array[:, :, i] = np.uint8(np.absolute(img_array[:, :, i]))
    return img_array


def play_images(images_dir, fps):
    """
        Reprodueix una seqüència d'imatges a la velocitat especificada.

        Arguments:
        images_dir (Path): La ruta al directori que conté les imatges.
        fps (float): Els frames per segon per a la reproducció.

        Returns:
        None
    """
    while True:
        interval = 1.0 / fps
        for image_name in images_dir:
            # Llegeix la imatge i la mostra
            cv2.imshow('Image', image_name)
            # Espera a prèmer la tecla 'q' per parar el vídeo
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            # Espera abans de mostrar la següent imatge
            time.sleep(interval)

# Funció per llegir imatges contingudes en un ZIP
def read_images_from_zip(zip_file_path):
    images = []
    with ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            with zip_ref.open(file_info) as file:
                # Llegir imatge com a buffer de bytes
                image_bytes = file.read()
                # Convertir els bytes a imatge utilitzant Pillow
                image = Image.open(BytesIO(image_bytes))
                images.append(image)
    return images

# Funció per llegir imatges contingudes en un GIF
def read_images_from_gif(gif_file_path):
    gif = cv2.VideoCapture(gif_file_path)
    images = []
    while True:
        ret, frame = gif.read()
        if not ret:
            break
        images.append(frame)
    gif.release()
    return images

# Funció per llegir frames d'un vídeo
def read_frames_from_video(video_file_path):
    frames = []
    cap = cv2.VideoCapture(video_file_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convertir frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames
@click.command()
@click.option('-i', '--input', required=True, type=click.Path(exists=True), help='Fitxer d’entrada. Argument '
                                                                                 'obligatori.')
@click.option('-o', '--output', type=click.Path(),
              help='Nom del fitxer en format propi amb la seqüència d’imatges de sortida i la informació necessària '
                   'per la descodificació.')

@click.option('--fps', default=25, type=int, help='nombre d’imatges per segon amb les quals és reproduirà el vídeo.')
@click.option('--filters', help='''Lista de filtros puntuales separados por comas.
{}'''.format('\n'.join([f"{key}: {val}\n" for key, val in FILTERS_DESCRIPTION.items()])))
@click.option('--conv_filters', help='''Lista de filtros convolucionales separados por comas.
{}'''.format('\n'.join([f"{key}: {val}\n" for key, val in CONV_FILTERS_DESCRIPTION.items()])))
@click.option('--ntiles', type=int,
              help='nombre de tessel·les en la qual dividir la imatge. Es poden indicar diferents valors per l’eix '
                   'vertical i horitzontal, o bé especificar la mida de les tessel·les en píxels.')
@click.option('--seekRange', type=int, help='desplaçament màxim en la cerca de tessel·les coincidents.')
@click.option('--GOP', type=int, help='nombre d’imatges entre dos frames de referència.')
@click.option('--quality', type=float,
              help='factor de qualitat que determinarà quan dos tessel·les és consideren coincidents.')

def main(input, output, fps, filters, conv_filters, ntiles, seekrange, gop, quality):
    filter_dict = {}
    conv_filter_dict = {}

    # Processament dels filtres puntuals
    if filters:
        filter_dict = {
            filter_item.split('[')[0]: filter_item.split('[')[1].split(']')[0] if '[' in filter_item else None
            for filter_item in filters.split(',')
        }

    # Processament dels filtres convolucionals
    if conv_filters:
        conv_filter_dict = {
            filter_item.split('[')[0]: filter_item.split('[')[1].split(']')[0] if '[' in filter_item else None
            for filter_item in conv_filters.split(',')
        }

    if input.endswith('.zip'):
        images = read_images_from_zip(input)
    elif input.endswith('.gif'):
        images = read_images_from_gif(input)
    elif input.endswith(('.avi', '.mpeg', '.mp4')):
        images = read_frames_from_video(input)
    else:
        print("Format d'arxiu no suportat. Si us plau, proporciona un arxiu en format ZIP, GIF o vídeo.")
        return

    for i, img in enumerate(images):
        img_array = np.array(img)
        # Aplica els filtres puntuals que s'han especificat
        if filters:
            img_array = apply_filters(img_array, filter_dict)
        # Aplica els filtres convolucionals que s'han especificat
        if conv_filters:
            img_array = apply_conv_filters(img_array, conv_filter_dict)
            # Converteix l'array de numpy en imatge
        img = Image.fromarray(img_array)
        # Desa la imatge com a JPEG
        img.save(f'image_{i}.jpeg')

    # Si s'ha especificat la opció output, guardem les imatges en un arxiu ZIP
    if output:
        with ZipFile(output, 'w') as new_zip:
            for i in range(len(images)):
                new_zip.write(f'image_{i}.jpeg')

    # Inicia la reproducció de les imatges en un thread separat
    play_thread = threading.Thread(target=play_images, args=(images, fps))
    play_thread.start()
    play_thread.join()



if __name__ == '__main__':
    main()
