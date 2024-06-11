import os

import click

from tmproject.filters import *
from tmproject.image_processing import *
from tmproject.utils import *


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
        compression_factor = calculate_compression_factor(input_size, zip_size)

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
