import json
import os
import time
from zipfile import ZipFile

import click
import cv2
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm
import math

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

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_compression_factor(original_size, compressed_size):
    return (1 - (compressed_size / original_size)) * 100


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


def play_images(image_array, fps):
    """
    Reproduïx una seqüència d'imatges amb una certa velocitat de quadres per segon (FPS).

    Args:
        image_array (list): Una llista de matrius que representen les imatges.
        fps (int): El nombre de quadres per segon amb els quals es reproduiran les imatges.

    Returns:
        None
    """
    interval = 1.0 / fps
    while True:
        for img in image_array:
            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
            time.sleep(interval)


def read_images_from_gif(gif_file_path, filters=None, conv_filters=None):
    """
    Llegeix les imatges d'un fitxer GIF i aplica opcionalment filtres.

    Args:
        gif_file_path (str): La ruta del fitxer GIF.
        filters (dict): Un diccionari de filtres puntuals i els seus paràmetres.
        conv_filters (dict): Un diccionari de filtres convolucionals i els seus paràmetres.

    Returns:
        list: Una llista de matrius que representen les imatges processades.
    """
    gif = Image.open(gif_file_path)
    processed_frames = []
    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert('RGB')
        img_array = np.array(frame)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        if filters:
            img_array = apply_filters(img_array, filters)
        if conv_filters:
            img_array = apply_conv_filters(img_array, conv_filters)
        processed_frames.append(img_array)
    return processed_frames


def read_images_from_zip(input_file, filters=None, conv_filters=None):
    """
    Llegeix imatges d'un fitxer comprimit ZIP i opcionalment aplica filtres a les imatges llegides.

    Args:
        input_file (str): La ruta del fitxer ZIP que conté les imatges.
        filters (dict): Un diccionari de filtres puntuals i els seus paràmetres.
        conv_filters (dict): Un diccionari de filtres convolucionals i els seus paràmetres.

    Returns:
        list: Una llista de matrius que representen les imatges processades.
    """
    processed_images = []
    with ZipFile(input_file, 'r') as zipf:
        for file_name in zipf.namelist():
            if file_name.endswith(('.jpg', '.png', '.bmp')):
                with zipf.open(file_name) as file:
                    img = Image.open(file)
                    img_array = np.array(img)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    if filters:
                        img_array = apply_filters(img_array, filters)
                    if conv_filters:
                        img_array = apply_conv_filters(img_array, conv_filters)
                    processed_images.append(img_array)
    return processed_images


def divide_into_tiles(image, ntiles):
    """
    Divideix una imatge en una llista de tessel·les.

    Args:
        image (numpy.ndarray): La matriu que representa la imatge.
        ntiles (int): El nombre de tessel·les en què es dividirà la imatge.

    Returns:
        tuple: Una tupla que conté la llista de tessel·les, l'alçada de cada tessel·la i l'amplada de cada tessel·la.
    """
    tiles = []
    h, w = image.shape[:2]
    tile_h, tile_w = h // ntiles, w // ntiles  # Calculem l'alçada i l'amplada de cada tessel·la.
    for i in range(ntiles):
        for j in range(ntiles):
            # Extraiem una tessel·la de la imatge i l'afegim a la llista de tessel·les.
            tile = image[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
            tiles.append(tile)
    return tiles, tile_h, tile_w  # Retornem les tessel·les, l'alçada i l'amplada de cada tessel·la.



def compare_tiles(tile1, tile2, quality):
    """
    Compara dues teselles i retorna si la similitud entre elles és superior a un cert nivell de qualitat.

    Args:
        tile1 (numpy.ndarray): La primera tesella.
        tile2 (numpy.ndarray): La segona tesella.
        quality (float): El nivell de qualitat mínim per considerar que les teselles són similars.

    Returns:
        bool: True si la similitud entre les teselles és superior al nivell de qualitat especificat, False altrament.
    """
    # Utilitzem la funció de coincidència de plantilles per comparar les teselles.
    result = cv2.matchTemplate(tile1, tile2, cv2.TM_CCOEFF_NORMED)
    # Obtenim el valor màxim de la coincidència.
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val > quality


def process_image(image, ref_image, ntiles, seekrange, quality):
    """
    Processa una imatge per a la reconstrucció de regions danificades utilitzant la cerca de teselles coincidents.

    Args:
        image (numpy.ndarray): La imatge a processar.
        ref_image (numpy.ndarray): La imatge de referència per a la comparació.
        ntiles (int): El nombre de teselles en què es dividirà la imatge.
        seekrange (int): El desplaçament màxim permès en la cerca de teselles coincidents.
        quality (float): El factor de qualitat que determina quan dues teselles es consideren coincidents.

    Returns:
        tuple: Una tupla que conté la imatge processada i una llista de les teselles a restaurar.
    """
    tiles, tile_h, tile_w = divide_into_tiles(image, ntiles)
    ref_tiles, _, _ = divide_into_tiles(ref_image, ntiles)
    tiles_to_restore = []

    for i, tile in enumerate(tiles):
        y = (i // ntiles) * tile_h
        x = (i % ntiles) * tile_w
        # Itera sobre el rang de cerca
        for dy in range(-seekrange, seekrange + 1):
            for dx in range(-seekrange, seekrange + 1):
                ref_y = y + dy * tile_h
                ref_x = x + dx * tile_w
                # Verifica si la tesella de referència està dins dels límits de la imatge
                if 0 <= ref_y < image.shape[0] - tile_h and 0 <= ref_x < image.shape[1] - tile_w:
                    # Obté la tesella de referència
                    ref_tile = ref_image[ref_y:ref_y + tile_h, ref_x:ref_x + tile_w]
                    if compare_tiles(tile, ref_tile, quality):
                        # Restaura la tesella de la imatge i l'afegeix a la llista de teselles a restaurar
                        image[y:y + tile_h, x:x + tile_w] = np.mean(tile, axis=(0, 1))
                        tiles_to_restore.append((y, x, tile))
                        break
    return image, tiles_to_restore


def decode_images(input_zip, gop, ntiles):
    """
    Descodifica imatges d'un fitxer ZIP utilitzant un esquema de group of pictures (GOP) i reconstrueix regions danificades.

    Args:
        input_zip (str): La ruta del fitxer ZIP que conté les imatges.
        gop (int): El nombre d'imatges entre dos frames de referència.
        ntiles (int): El nombre de tessel·les en què es dividiran les imatges.

    Returns:
        list: Una llista de matrius que representen les imatges descodificades.
    """
    with ZipFile(input_zip, 'r') as zipf:
        image_files = sorted([f for f in zipf.namelist() if f.endswith('.jpeg')])
        json_data = json.loads(zipf.read('all_data.json'))
        tile_h, tile_w = None, None
        decoded_images = []
        for i, image_file in tqdm(enumerate(image_files), total=len(image_files), desc='Decoding Images'):
            with zipf.open(image_file) as img_file:
                # Decodifiquem l'imatge
                image = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Si és un frame de referència
            if i % gop == 0:
                # Copiem l'imatge com a referència
                ref_image = image.copy()
                # Si encara no tenim l'alçada o l'amplada de les teselles, les dividim
                if tile_h is None or tile_w is None:
                    tiles, tile_h, tile_w = divide_into_tiles(ref_image, ntiles)
                # Afegim la imatge de referència a les imatges descodificades
                decoded_images.append(ref_image)

            # Si no és un frame de referència
            else:
                tiles_to_restore = json_data.pop(0)
                image = reconstruct_image(image, tiles_to_restore, tile_h, tile_w)
                decoded_images.append(image)
    return decoded_images


def reconstruct_image(image, tiles_to_restore, tile_h, tile_w):
    """
    Reconstrueix una imatge danificada utilitzant teselles prèviament emmagatzemades.

    Args:
        image (numpy.ndarray): La imatge a reconstruir.
        tiles_to_restore (list): Una llista de les teselles a restaurar.
        tile_h (int): L'alçada de cada tesella.
        tile_w (int): L'amplada de cada tesella.

    Returns:
        numpy.ndarray: La imatge reconstruïda.
    """
    for y, x, tile in tiles_to_restore:
        # Substituïm la regió  de la imatge amb la tesella emmagatzemada
        image[y:y + tile_h, x:x + tile_w] = np.array(tile, dtype=np.uint8)
    return image

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
    """
    Funció principal que processa les imatges d'entrada, aplica els filtres especificats i guarda les imatges processades en un fitxer ZIP si s'ha proporcionat un nom de fitxer de sortida, o bé les mostra en pantalla si no s'ha proporcionat cap sortida.

    Args:
        input (str): Ruta del fitxer d'entrada que pot ser un fitxer ZIP o un GIF.
        output (str): Nom del fitxer de sortida en format ZIP amb les imatges processades i la informació necessària per a la descodificació.
        fps (int): Nombre d'imatges per segon amb les quals es reproduirà el vídeo.
        filters (str): Llista de filtres puntuals separats per comes.
        conv_filters (str): Llista de filtres convolucionals separats per comes.
        ntiles (int): Nombre de tessel·les en què es dividirà la imatge per a la reconstrucció de regions danificades.
        seekrange (int): Desplaçament màxim en la cerca de tessel·les coincidents.
        gop (int): Nombre d'imatges entre dos frames de referència.
        quality (float): Factor de qualitat que determinarà quan dues tessel·les es consideren coincidents.
    """
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

    if input.endswith('.zip'):
        images = read_images_from_zip(input, filter_dict, conv_filter_dict)
    elif input.endswith('.gif'):
        images = read_images_from_gif(input, filter_dict, conv_filter_dict)
    else:
        print("Format d'arxiu invalid")
        return

    if not output:
        play_images(images, fps)
    else:
        info = []
        start_time = time.time()
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
                zipf.writestr(f'frame_{i:04d}.jpeg', image_bytes)

            zipf.writestr('all_data.json', json.dumps(info))

        compression_time = time.time() - start_time
        input_size = os.path.getsize(input)
        zip_size = os.path.getsize(output)
        compression_factor = (1 - (zip_size / input_size)) * 100

        print(f"Mida del arxiu d'entrada {input}: {input_size:,}")
        print(f"Mida del arxiu de sortida {output}: {zip_size:,}")
        print("Factor de compressió en %: {:.2f}%".format(compression_factor))
        print("Temps de compressió: {:.2f} segons".format(compression_time))

        start_time = time.time()
        decoded_images = decode_images(output, gop, ntiles)
        decoding_time = time.time() - start_time

        psnr_values = [calculate_psnr(images[i], decoded_images[i]) for i in range(len(images))]
        avg_psnr = np.mean(psnr_values)

        print("Temps de decodificació: {:.2f} segons".format(decoding_time))
        print("PSNR mitjà: {:.2f} dB".format(avg_psnr))

        play_images(decoded_images, fps)


if __name__ == "__main__":
    main()
