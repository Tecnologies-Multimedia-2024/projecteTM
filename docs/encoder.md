# Encoder

## divide_into_tiles(image, ntiles)
Divideix una imatge en una llista de tessel·les.
- **Args:**
    - `image` (numpy.ndarray): La matriu que representa la imatge.
    - `ntiles` (int): El nombre de tessel·les en què es dividirà la imatge.
- **Returns:**
    - `tuple`: Una tupla que conté la llista de tessel·les, l'alçada de cada tessel·la i l'amplada de cada tessel·la.

## compare_tiles(tile1, tile2, quality)
Compara dues teselles i retorna si la similitud entre elles és superior a un cert nivell de qualitat.
- **Args:**
    - `tile1` (numpy.ndarray): La primera tesella.
    - `tile2` (numpy.ndarray): La segona tesella.
    - `quality` (float): El nivell de qualitat mínim per considerar que les teselles són similars.
- **Returns:**
    - `bool`: True si la similitud entre les teselles és superior al nivell de qualitat especificat, False altrament.

## process_image(image, ref_image, ntiles, seekrange, quality)
Processa una imatge per a la reconstrucció de regions danificades utilitzant la cerca de teselles coincidents.
- **Args:**
    - `image` (numpy.ndarray): La imatge a processar.
    - `ref_image` (numpy.ndarray): La imatge de referència per a la comparació.
    - `ntiles` (int): El nombre de teselles en què es dividirà la imatge.
    - `seekrange` (int): El desplaçament màxim permès en la cerca de teselles coincidents.
    - `quality` (float): El factor de qualitat que determina quan dues teselles es consideren coincidents.
- **Returns:**
    - `tuple`: Una tupla que conté la imatge processada i una llista de les teselles a restaurar.

## decode_images(input_zip, gop, ntiles)
Descodifica imatges d'un fitxer ZIP utilitzant un esquema de group of pictures (GOP) i reconstrueix regions danificades.
- **Args:**
    - `input_zip` (str): La ruta del fitxer ZIP que conté les imatges.
    - `gop` (int): El nombre d'imatges entre dos frames de referència.
    - `ntiles` (int): El nombre de tessel·les en què es dividiran les imatges.
- **Returns:**
    - `list`: Una llista de matrius que representen les imatges descodificades.

## reconstruct_image(image, tiles_to_restore, tile_h, tile_w)
Reconstrueix una imatge danificada utilitzant teselles prèviament emmagatzemades.
- **Args:**
    - `image` (numpy.ndarray): La imatge a reconstruir.
    - `tiles_to_restore` (list): Una llista de les teselles a restaurar.
    - `tile_h` (int): L'alçada de cada tesella.
    - `tile_w` (int): L'amplada de cada tesella.
- **Returns:**
    - `numpy.ndarray`: La imatge reconstruïda.
