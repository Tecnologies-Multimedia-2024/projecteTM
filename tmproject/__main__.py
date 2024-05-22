import json
import os
import time
from pathlib import Path
from zipfile import ZipFile
import click
import cv2
import numpy as np
from PIL import Image, ImageSequence
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

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


def divide_into_tiles(image, nTiles):
    # Dividim la imatge en tessel·les segons el nombre indicat
    tiles = []
    h, w = image.shape[:2]
    tile_h, tile_w = h // nTiles, w // nTiles

    for i in range(nTiles):
        for j in range(nTiles):
            tile = image[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
            tiles.append(tile)

    return tiles, tile_h, tile_w


def calculate_mean_color(tile):
    # Calcula el color mitjà de la tessel·la
    return np.mean(tile, axis=(0, 1))


def compare_tiles(tile1, tile2, quality):
    # Calcula la diferència entre dues tessel·les i determina si són equivalents segons el factor de qualitat
    difference = np.mean(np.abs(tile1 - tile2))
    return difference < quality


def process_image(image, ref_image, nTiles, seekRange, quality):
    # Processa una imatge comparant-la amb la imatge de referència
    tiles, tile_h, tile_w = divide_into_tiles(image, nTiles)
    ref_tiles, _, _ = divide_into_tiles(ref_image, nTiles)
    tiles_to_restore = []

    for i, tile in enumerate(tiles):
        y = (i // nTiles) * tile_h
        x = (i % nTiles) * tile_w
        for dy in range(-seekRange, seekRange + 1):
            for dx in range(-seekRange, seekRange + 1):
                ref_y = y + dy * tile_h
                ref_x = x + dx * tile_w
                if 0 <= ref_y < image.shape[0] - tile_h and 0 <= ref_x < image.shape[1] - tile_w:
                    ref_tile = ref_image[ref_y:ref_y + tile_h, ref_x:ref_x + tile_w]
                    if compare_tiles(tile, ref_tile, quality):
                        image[y:y + tile_h, x:x + tile_w] = calculate_mean_color(tile)
                        tiles_to_restore.append((y, x, tile))
                        break
    return image, tiles_to_restore


def reconstruir_tessella(imatge_referencia, info_tesselles, ntiles, seekrange):
    # Funció per reconstruir una tessel·la a partir de la informació proporcionada

    # Inicialitzar la matriu de la imatge reconstruïda
    reconstructed_image = imatge_referencia.copy()

    # Iterar sobre la informació de les tessel·les
    for tessella_info in info_tesselles:
        y, x, tile_data = tessella_info
        # Descomprimir la tessel·la
        tile_data = np.frombuffer(tile_data, dtype=np.uint8)
        tile_data = np.reshape(tile_data, (ntiles, ntiles, 3))

        # Reubicar la tessel·la a la posició correcta a la imatge reconstruïda
        start_y = y * ntiles
        start_x = x * ntiles
        end_y = start_y + ntiles
        end_x = start_x + ntiles
        reconstructed_image[start_y:end_y, start_x:end_x, :] = tile_data

    return reconstructed_image


def descodificar_zip(zip_path, gop, ntiles, seekrange):
    images = []
    tiles_info = []

    # Obrir el fitxer zip en mode de lectura
    with ZipFile(zip_path, 'r') as zip_ref:
        # Llegir i processar les imatges i la informació de les tessel·les
        for name in zip_ref.namelist():
            # Si el fitxer és una imatge JPEG
            if name.endswith('.jpeg'):
                # Llegir el contingut de l'arxiu
                with zip_ref.open(name) as image_file:
                    # Decodificar la imatge
                    image_data = image_file.read()
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    images.append(image)
            # Si el fitxer és el fitxer JSON amb la informació de les tessel·les
            elif name == 'all_data.json':
                # Llegir el contingut de l'arxiu
                with zip_ref.open(name) as json_file:
                    # Deserialitzar les dades JSON
                    tiles_info = json.load(json_file)

    # Reagrupar les imatges en el GOP
    gop_images = [images[i:i + gop] for i in range(0, len(images), gop)]

    # Iterar sobre els GOPs i reconstruir les imatges
    reconstructed_images = []
    for i, gop_group in enumerate(gop_images):
        ref_image = gop_group[0].copy()
        reconstructed_images.append(ref_image)

        # Iterar sobre les imatges del GOP
        for j in range(1, len(gop_group)):
            # Reconstruir la tessel·la per a cada imatge
            reconstructed_tile = reconstruir_tessella(ref_image, tiles_info[i * (gop - 1) + (j - 1)], ntiles,
                                                       seekrange)
            reconstructed_images.append(reconstructed_tile)

            # La imatge reconstruïda es converteix en la referència per la següent iteració
            ref_image = reconstructed_tile

    return reconstructed_images, tiles_info


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
        image_files = sorted(images_dir.glob('*.jpeg'))
        images = [cv2.imread(str(file)) for file in image_files]
        info = []
        with ZipFile(output, 'w') as zipf:
            ref_image = None

            for i, image in tqdm(enumerate(images), total=len(images), desc='Processing Images'):
                if i % gop == 0:
                    ref_image = image.copy()
                    processed_image = image
                else:
                    processed_image, tiles_to_restore = process_image(image, ref_image, ntiles, seekrange, quality)
                    tiles_to_restore = [(y, x, tile.tolist()) for y, x, tile in tiles_to_restore]

                    info.append(tiles_to_restore)

                _, buffer = cv2.imencode('.jpeg', processed_image)
                image_bytes = buffer.tobytes()

                # Save the processed image to the zip file
                zipf.writestr(f'frame_{i:04d}.jpeg', image_bytes)
            # Write the JSON data to the ZIP file
            zipf.writestr('all_data.json', json.dumps(info))
        descodificar_zip(output,gop,ntiles,seekrange)

if __name__ == "__main__":
    main()
