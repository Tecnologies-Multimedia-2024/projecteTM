import time
from pathlib import Path
from zipfile import ZipFile
import click
import cv2
import numpy as np
from PIL import Image, ImageSequence

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


def play_images(images_dir, fps):
    image_paths = sorted(images_dir.glob('*.jpeg'))
    interval = 1.0 / fps
    while True:
        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                for file_path in images_dir.iterdir():
                    file_path.unlink()
                images_dir.rmdir()
                return
            time.sleep(interval)


def read_images_from_gif(gif_file_path, images_dir, filters=None, conv_filters=None):
    gif = Image.open(gif_file_path)
    frame_count = 0
    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert('RGB')
        img_array = np.array(frame)
        if filters:
            img_array = apply_filters(img_array, filters)
        if conv_filters:
            img_array = apply_conv_filters(img_array, conv_filters)
        img = Image.fromarray(img_array)
        img.save(images_dir / f'frame_{frame_count}.jpeg')
        frame_count += 1


def read_images_from_zip(input_file, images_dir, filters=None, conv_filters=None):
    with ZipFile(input_file, 'r') as zip_ref:
        zip_ref.extractall(images_dir)
    for file_path in images_dir.iterdir():
        if file_path.suffix in ['.jpg', '.png', '.bmp']:
            img = Image.open(file_path)
            img_array = np.array(img)
            if filters:
                img_array = apply_filters(img_array, filters)
            if conv_filters:
                img_array = apply_conv_filters(img_array, conv_filters)
            img_array = Image.fromarray(img_array)
            img_array.save(file_path)
            if file_path.suffix in ['.png', '.bmp']:
                img = Image.open(file_path)
                img.save(file_path.with_suffix('.jpeg'))


def divide_into_tiles(image, num_tiles):
    """Subdivideix la imatge en el nombre de tessel·les especificat per l'usuari."""
    height, width = image.shape[:2]
    tile_height = height // num_tiles
    tile_width = width // num_tiles
    tiles = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            tile = image[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width]
            tiles.append(tile)
    return tiles

def compare_tiles(tile1, tile2, max_shift):
    """Compara dues tessel·les utilitzant correlació i retorna si són equivalents o no."""
    correlation = cv2.matchTemplate(tile1, tile2, cv2.TM_CCORR_NORMED)
    min_val, max_val, _, _ = cv2.minMaxLoc(correlation)
    return max_val > max_shift

def remove_equivalent_tiles(reference_image, image, num_tiles, max_shift):
    """Elimina les tessel·les equivalents de la imatge."""
    new_image = np.copy(image)
    reference_tiles = divide_into_tiles(reference_image, num_tiles)
    tiles = divide_into_tiles(image, num_tiles)
    for i, tile in enumerate(tiles):
        if compare_tiles(tile, reference_tiles[i], max_shift):
            new_image = cv2.rectangle(new_image, (i % num_tiles, i // num_tiles), ((i % num_tiles) + 1, (i // num_tiles) + 1), (0, 0, 0), -1)
    return new_image

def process_images(images, gop, num_tiles, max_shift):
    """Processa les imatges segons el GOP i els passos especificats."""
    processed_images = []
    for i in range(0, len(images), gop):
        reference_image = images[i]
        processed_images.append(reference_image)
        for image in images[i+1:i+gop]:
            new_image = remove_equivalent_tiles(reference_image, image, num_tiles, max_shift)
            processed_images.append(new_image)
    return processed_images
@click.command()
@click.option('-i', '--input', required=True, type=click.Path(exists=True),
              help='Fitxer d’entrada. Argument obligatori.')
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
    if filters:
        filter_dict = {
            filter_item.split('[')[0]: filter_item.split('[')[1].split(']')[0] if '[' in filter_item else None
            for filter_item in filters.split(',')
        }
    if conv_filters:
        conv_filter_dict = {
            filter_item.split('[')[0]: filter_item.split('[')[1].split(']')[0] if '[' in filter_item else None
            for filter_item in conv_filters.split(',')
        }
    images_dir = Path("temp_images")
    images_dir.mkdir(exist_ok=True)

    if input.endswith('.zip'):
        read_images_from_zip(input, images_dir, filter_dict, conv_filter_dict)
    elif input.endswith('.gif'):
        read_images_from_gif(input, images_dir, filter_dict, conv_filter_dict)
    else:
        print("Format d'arxiu invalid")
        return

    if not output:
        play_images(images_dir, fps)
    else:
        image_files=sorted(images_dir.glob('*.jpeg'))
        images = [cv2.imread(str(file)) for file in image_files]
        compressed_images=process_images(images, gop, ntiles, seekrange)
        with ZipFile(output, 'w') as new_zip:
            for i, image in enumerate(compressed_images):
                image_filename = f'image_{i}.jpg'
                # Escriu la imatge comprimida a l'arxiu ZIP amb el nom de fitxer corresponent
                cv2.imwrite(image_filename, image)
                new_zip.write(image_filename)
if __name__ == "__main__":
    main()
