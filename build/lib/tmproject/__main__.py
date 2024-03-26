import time
from pathlib import Path
from zipfile import ZipFile

import click
import cv2
import numpy as np
from PIL import Image


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
@click.option('--fps', type=int, help='nombre d’imatges per segon amb les quals és reproduirà el vídeo.')
@click.option('--binarization', type=int,
              help='Filtre puntual de binarització utilitzant el valor llindar indicat.')
@click.option('--negative', is_flag=True, help='Filtre puntual negatiu sobre la imatge.')
@click.option('--averaging', type=int,
              help='Filtre convolucional d’averaging en zones de value x value.')
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
def main(input, output, encode, decode, fps, binarization, negative, averaging, ntiles, seekrange, gop, quality, batch):
    click.echo(f'Input: {input}')
    click.echo(f'Output: {output}')
    click.echo(f'Encode: {encode}')
    click.echo(f'Decode: {decode}')
    click.echo(f'FPS: {fps}')
    click.echo(f'Binarization: {binarization}')
    click.echo(f'Negative: {negative}')
    click.echo(f'Averaging: {averaging}')
    click.echo(f'nTiles: {ntiles}')
    click.echo(f'seekRange: {seekrange}')
    click.echo(f'GOP: {gop}')
    click.echo(f'Quality: {quality}')
    click.echo(f'Batch: {batch}')

    # Extreure imatges del ZIP
    with ZipFile(input, 'r') as zip_ref:
        zip_ref.extractall('temp_images')

    # Aplicar filtres
    images_dir = Path('temp_images')
    for file_path in images_dir.iterdir():
        if file_path.suffix in ['.png', '.bmp', '.gif']:
            img = Image.open(file_path)
            img_array = np.array(img)

            if binarization is not None:
                im_gray = img.convert('L')
                img_array = np.array(im_gray)
                _, img_array = cv2.threshold(img_array, binarization, 255, cv2.THRESH_BINARY)

            if negative:
                im_gray = img.convert('L')
                img_array = np.array(im_gray)
                img_array = 255 - img_array

            if averaging is not None:
                im_gray = img.convert('L')
                img_array = np.array(im_gray)
                noise = np.random.random(img_array.shape)
                noise_95 = noise > 0.95
                im_noisy = img_array.copy()
                im_noisy[noise_95] = 255
                noise_05 = noise < 0.05
                im_noisy[noise_05] = 0

                kernel = np.ones((averaging, averaging), np.float32) / (averaging * averaging)
                img_array = cv2.filter2D(im_noisy, -1, kernel)
            img_array = Image.fromarray(img_array)
            img_array.save(file_path)

    for file_path in images_dir.iterdir():
        if file_path.suffix in ['.png', '.bmp', '.gif']:
            img = Image.open(file_path)
            img.save(images_dir.joinpath(file_path.stem + '.jpg'))

    image_paths = sorted(images_dir.glob('*.jpg'))
    interval = 1.0 / fps
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(interval)
    # esborrar la carpeta d'imatges
    for file_path in images_dir.iterdir():
        file_path.unlink()
    images_dir.rmdir()


if __name__ == '__main__':
    main()
