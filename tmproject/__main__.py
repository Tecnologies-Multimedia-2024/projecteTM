import threading
import time
from pathlib import Path
from zipfile import ZipFile

import click
import cv2
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
            # Passar la imatge a escala de grisos si no ho està
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            _, img_array = cv2.threshold(img_array, threshold_value, 255, cv2.THRESH_BINARY)
        # Aplica el filtre de negative
        elif filter_name == 'negative':
            # Passar la imatge a escala de grisos si no ho està
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_array = 255 - img_array
        # Aplica el filtre de contrast_stretching
        elif filter_name == 'contrast_stretching':
            # Passar la imatge a escala de grisos si no ho està
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img = ((img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))) * 255
            img_array = img.astype(np.uint8)
        # Aplica el filtre de hsv
        elif filter_name == 'hsv':
            # Passar la imatge a RGB si no ho està
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HLS)
        # Aplica el filtre de sepia
        elif filter_name == 'sepia':
            # Passar la imatge a RGB si no ho està
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]])
            img_array = cv2.transform(img_array, sepia_matrix)
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
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        # Aplica el filtre d'averaging
        if filter_name == 'averaging':
            kernel_size = int(filter_param) if filter_param is not None else 3
            img_array = cv2.blur(img_array, (kernel_size, kernel_size))
        # Aplica el filtre de sobel
        elif filter_name == 'sobel':
            kernel_size = int(filter_param) if filter_param is not None else 3
            grad_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=kernel_size)
            grad_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=kernel_size)
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            img_array = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Aplica el filtre de sharpening
        elif filter_name == 'sharpening':
            kernel_size = int(filter_param) if filter_param is not None else 3
            kernel = np.ones((kernel_size, kernel_size), np.float32) * -1
            center_value = kernel_size * kernel_size
            kernel[int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)] = center_value
            img_array = cv2.filter2D(img_array, -1, kernel)
        # Aplica el filtre gaussià
        elif filter_name == 'gaussian':
            kernel_size = int(filter_param) if filter_param is not None else 3
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        # Aplica el filtre Laplacià
        elif filter_name == 'laplacian':
            kernel_size = int(filter_param) if filter_param is not None else 1
            img_array = cv2.Laplacian(img_array, cv2.CV_64F, ksize=kernel_size)
            img_array = np.uint8(np.absolute(img_array))
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
    # Obté les imatges en JPEG
    image_paths = sorted(images_dir.glob('*.jpg'))
    while True:
        interval = 1.0 / fps
        for image_path in image_paths:
            # Llegeix la imatge i la mostra
            img = cv2.imread(str(image_path))
            cv2.imshow('Image', img)
            # Espera a prèmer la tecla 'q' per parar el vídeo
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # esborrar la carpeta d'imatges temporals
                for file_path in images_dir.iterdir():
                    file_path.unlink()
                images_dir.rmdir()
                return
            # Espera abans de mostrar la següent imatge
            time.sleep(interval)


@click.command()
@click.option('-i', '--input', required=True, type=click.Path(exists=True), help='Fitxer d’entrada. Argument '
                                                                                 'obligatori.')
@click.option('-o', '--output', type=click.Path(),
              help='Nom del fitxer en format propi amb la seqüència d’imatges de sortida i la informació necessària '
                   'per la descodificació.')
@click.option('-e', '--encode', is_flag=True,
              help='Argument que indica que s’haurà d’aplicar la codificació sobre el conjunt d’imatges d’input i '
                   'guardar el resultat al lloc indicat per output. En acabar, s’ha de procedir a reproduir el '
                   'conjunt d’imatges sense codificar (input).')
@click.option('-d', '--decode', is_flag=True,
              help='Argument que indica que s’haurà d’aplicar la descodificació sobre el conjunt d’imatges d’input '
                   'provinents d’un fitxer en format propi i reproduir el conjunt d’imatges descodificat (output). ')
@click.option('--fps', default=25, type=int, help='nombre d’imatges per segon amb les quals és reproduirà el vídeo.')
@click.option('--filters', default=" ", help='''Lista de filtros puntuales separados por comas.
{}'''.format('\n'.join([f"{key}: {val}\n" for key, val in FILTERS_DESCRIPTION.items()])))
@click.option('--conv_filters', default=" ", help='''Lista de filtros convolucionales separados por comas.
{}'''.format('\n'.join([f"{key}: {val}\n" for key, val in CONV_FILTERS_DESCRIPTION.items()])))
@click.option('--ntiles', type=int,
              help='nombre de tessel·les en la qual dividir la imatge. Es poden indicar diferents valors per l’eix '
                   'vertical i horitzontal, o bé especificar la mida de les tessel·les en píxels.')
@click.option('--seekRange', type=int, help='desplaçament màxim en la cerca de tessel·les coincidents.')
@click.option('--GOP', type=int, help='nombre d’imatges entre dos frames de referència.')
@click.option('--quality', type=float,
              help='factor de qualitat que determinarà quan dos tessel·les és consideren coincidents.')
@click.option('-b', '--batch', is_flag=True,
              help='en aquest mode no s’obrirà cap finestra del reproductor de vídeo. Ha de permetre executar el '
                   'còdec a través de Shell scripts per avaluar de forma automatitzada el rendiment de l’algorisme '
                   'implementat en funció dels diferents paràmetres de configuració.')
def main(input, output, encode, decode, fps, filters, conv_filters, ntiles, seekrange, gop, quality, batch):
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
    # Extreure imatges del ZIP
    with ZipFile(input, 'r') as zip_ref:
        zip_ref.extractall('temp_images')
    images_dir = Path('temp_images')
    # Llegeix les imatges de dintre el zip que estan en format JPEG, PNG BMP i GIF
    for file_path in images_dir.iterdir():
        if file_path.suffix in ['.jpg', '.png', '.bmp', '.gif']:
            img = Image.open(file_path)
            img_array = np.array(img)
            # Aplica els filtres puntuals que s'han especificat
            img_array = apply_filters(img_array, filter_dict)
            # Aplica els filtres convolucionals que s'han especificat
            img_array = apply_conv_filters(img_array, conv_filter_dict)
            # Guarda les imatges amb els filtres aplicats
            img_array = Image.fromarray(img_array)
            img_array.save(file_path)

    # Guarda les imatges en format JPEG
    for file_path in images_dir.iterdir():
        if file_path.suffix in ['.png', '.bmp', '.gif']:
            img = Image.open(file_path)
            img.save(images_dir.joinpath(file_path.stem + '.jpg'))
    # Si s'indica l'opció output, guarda les imatges en un ZIP
    if output:
        with ZipFile(output, 'w') as new_zip:
            for file_path in images_dir.iterdir():
                new_zip.write(file_path, file_path.name)

    play_thread = threading.Thread(target=play_images, args=(images_dir, fps))
    play_thread.start()
    play_thread.join()


if __name__ == '__main__':
    main()
