# Documentació projecte Tecnologies Multimèdia

## Funcions Filtres

        def apply_filters(img_array, filters):

            """ Aplica una sèrie de filtres puntuals a una imatge representada com una matriu de píxels.
            Args:
                img_array (numpy.ndarray): La matriu que representa la imatge.
                filters (dict): Un diccionari que conté els noms dels filtres com a claus i els valors dels
                                paràmetres dels filtres com a valors.
            Returns:
                numpy.ndarray: La matriu de la imatge després d'aplicar els filtres puntuals."""

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


## Funcions Filtres convulacionals

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

## Reproduir seqüència d'imatges

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


## Llegir imatges d'un GIF
    
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

## Llegir imatges d'un ZIP

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

## Divideix imatge en tessel·les
    
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
            tile_h, tile_w = h // ntiles, w // ntiles
            for i in range(ntiles):
                for j in range(ntiles):
                    tile = image[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
                    tiles.append(tile)
            return tiles, tile_h, tile_w

## Comparar tessel·les d'una imatge

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
        result = cv2.matchTemplate(tile1, tile2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val > quality

## Processar imatge
    
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
            for dy in range(-seekrange, seekrange + 1):
                for dx in range(-seekrange, seekrange + 1):
                    ref_y = y + dy * tile_h
                    ref_x = x + dx * tile_w
                    if 0 <= ref_y < image.shape[0] - tile_h and 0 <= ref_x < image.shape[1] - tile_w:
                        ref_tile = ref_image[ref_y:ref_y + tile_h, ref_x:ref_x + tile_w]
                        if compare_tiles(tile, ref_tile, quality):
                            image[y:y + tile_h, x:x + tile_w] = np.mean(tile, axis=(0, 1))
                            tiles_to_restore.append((y, x, tile))
                            break
        return image, tiles_to_restore


## Decodificar imatges d'un fitxer ZIP

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
                    image = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
                if i % gop == 0:
                    ref_image = image.copy()
                    if tile_h is None or tile_w is None:
                        tiles, tile_h, tile_w = divide_into_tiles(ref_image, ntiles)
                    decoded_images.append(ref_image)
                else:
                    tiles_to_restore = json_data.pop(0)
                    image = reconstruct_image(image, tiles_to_restore, tile_h, tile_w)
                    decoded_images.append(image)
        return decoded_images

## Reconstruir imatge decodificada

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
            image[y:y + tile_h, x:x + tile_w] = np.array(tile, dtype=np.uint8)
        return image