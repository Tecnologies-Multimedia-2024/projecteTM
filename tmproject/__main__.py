import sys
import time
from pathlib import Path
from zipfile import ZipFile

import click
import cv2
import numpy as np
from PIL import Image

FILTERS_DESCRIPTION = {
    'binarization': 'Filtre puntual de binarització utilitzant el valor llindar indicat.',
    'negative': 'Filtre puntual negatiu sobre la imatge.',
    'hsv': '',
    'sepia': '',
    'contrast_stretching': ''
}

CONV_FILTERS_DESCRIPTION = {
    'averaging': 'Filtre convolucional d’averaging en zones de value x value.',
    'sobel': '',
    'gaussian': 'Aplica un desenfoque gaussiano en la imagen.',
    'sharpening': 'Realiza un filtro de realce en la imagen.',
    'laplacian': 'Aplica un desenfoque laplacian en la imagen.',
}


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
{}'''.format('\n'.join([f"{key}: {val}" for key, val in FILTERS_DESCRIPTION.items()])))
@click.option('--conv_filters', default=" ", help='''Lista de filtros convolucionales separados por comas.
{}'''.format('\n'.join([f"{key}: {val}" for key, val in CONV_FILTERS_DESCRIPTION.items()])))
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
    binarization, negative, sepia, hsv, contrast_stretching = None, None, None, None, None
    conv_filter_dict = {}
    if filters:
        filter_dict = {
            filter_item.split('[')[0]: filter_item.split('[')[1].split(']')[0] if '[' in filter_item else None
            for filter_item in filters.split(',')
        }
        binarization = filter_dict.get('binarization', None)
        negative = 'negative' in filters
        sepia = 'sepia' in filters
        hsv = 'hsv' in filters
        contrast_stretching = 'contrast_stretching' in filters

    if conv_filters:
        conv_filter_dict = {
            filter_item.split('[')[0]: filter_item.split('[')[1].split(']')[0] if '[' in filter_item else None
            for filter_item in conv_filters.split(',')
        }

    averaging = conv_filter_dict.get('averaging', None)
    sobel = conv_filter_dict.get('sobel', None)
    sharpening = conv_filter_dict.get('sharpening', None)
    gaussian = conv_filter_dict.get('gaussian', None)
    laplacian = conv_filter_dict.get('laplacian', None)

    # Extreure imatges del ZIP
    with ZipFile(input, 'r') as zip_ref:
        zip_ref.extractall('temp_images')

    # Aplicar filtres
    images_dir = Path('temp_images')
    for file_path in images_dir.iterdir():
        if file_path.suffix in ['.png', '.bmp', '.gif']:
            img = Image.open(file_path)
            img_array = np.array(img)

            if binarization:
                im_gray = img.convert('L')
                img_array = np.array(im_gray)
                _, img_array = cv2.threshold(img_array, int(binarization), 255, cv2.THRESH_BINARY)

            if negative:
                im_gray = img.convert('L')
                img_array = np.array(im_gray)
                img_array = 255 - img_array

            if contrast_stretching:
                gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                img = ((gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image))) * 255
                img_array = img.astype(np.uint8)

            if hsv:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HLS)

            if sepia:
                sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                         [0.349, 0.686, 0.168],
                                         [0.272, 0.534, 0.131],
                                         ])

                img_array = cv2.transform(img_array, sepia_matrix)

            if averaging:
                img_array = cv2.blur(img_array, (int(averaging), int(averaging)))
            if sobel:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=int(sobel))
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=int(sobel))
                magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

                img_array = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if sharpening:
                kernel = np.ones((int(sharpening), int(sharpening)), np.float32) * -1
                center_value = int(sharpening) * int(sharpening)
                kernel[int((int(sharpening) - 1) / 2), int((int(sharpening) - 1) / 2)] = center_value
                img_array = cv2.filter2D(img_array, -1, kernel)
            if gaussian:
                img_array = cv2.GaussianBlur(img_array, (int(gaussian), int(gaussian)), 0)

            if laplacian:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                img_array = cv2.Laplacian(gray, cv2.CV_64F, ksize=int(laplacian))
                img_array = np.uint8(np.absolute(img_array))
            img_array = Image.fromarray(img_array)
            img_array.save(file_path)

    for file_path in images_dir.iterdir():
        if file_path.suffix in ['.png', '.bmp', '.gif']:
            img = Image.open(file_path)
            img.save(images_dir.joinpath(file_path.stem + '.jpg'))

    if output:
        with ZipFile(output, 'w') as new_zip:
            for file_path in images_dir.iterdir():
                new_zip.write(file_path, file_path.name)

    image_paths = sorted(images_dir.glob('*.jpg'))
    interval = 1.0 / fps
    while True:
        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # esborrar la carpeta d'imatges
                for file_path in images_dir.iterdir():
                    file_path.unlink()
                images_dir.rmdir()
                sys.exit(0)
            time.sleep(interval)




if __name__ == '__main__':
    main()
