# Filtres Normals i convulacionals

## apply_filters(img_array, filters)
Aplica una sèrie de filtres puntuals a una imatge representada com una matriu de píxels.

- **Args:**
    - `img_array` (numpy.ndarray): La matriu que representa la imatge.
    - `filters` (dict): Un diccionari que conté els noms dels filtres com a claus i els valors dels paràmetres dels filtres com a valors.
- **Returns:**
    - `numpy.ndarray`: La matriu de la imatge després d'aplicar els filtres puntuals.

## apply_conv_filters(img_array, conv_filters)
Aplica una sèrie de filtres convolucionals a una imatge representada com una matriu de píxels.

- **Args:**
    - `img_array` (numpy.ndarray): La matriu que representa la imatge.
    - `conv_filters` (dict): Un diccionari que conté els noms dels filtres convolucionals com a claus i els valors dels paràmetres dels filtres com a valors.
- **Returns:**
    - `numpy.ndarray`: La matriu de la imatge després d'aplicar els filtres convolucionals.

## play_images(image_array, fps)
Reproduïx una seqüència d'imatges amb una certa velocitat de quadres per segon (FPS).

- **Args:**
    - `image_array` (list): Una llista de matrius que representen les imatges.
    - `fps` (int): El nombre de quadres per segon amb els quals es reproduiran les imatges.
- **Returns:**
    - `None`

## read_images_from_gif(gif_file_path, filters=None, conv_filters=None)
Llegeix les imatges d'un fitxer GIF i aplica opcionalment filtres.

- **Args:**
    - `gif_file_path` (str): La ruta del fitxer GIF.
    - `filters` (dict): Un diccionari de filtres puntuals i els seus paràmetres.
    - `conv_filters` (dict): Un diccionari de filtres convolucionals i els seus paràmetres.
- **Returns:**
    - `list`: Una llista de matrius que representen les imatges processades.

## read_images_from_zip(input_file, filters=None, conv_filters=None)
Llegeix imatges d'un fitxer comprimit ZIP i opcionalment aplica filtres a les imatges llegides.

- **Args:**
    - `input_file` (str): La ruta del fitxer ZIP que conté les imatges.
    - `filters` (dict): Un diccionari de filtres puntuals i els seus paràmetres.
    - `conv_filters` (dict): Un diccionari de filtres convolucionals i els seus paràmetres.
- **Returns:**
    - `list`: Una llista de matrius que representen les imatges processades.
