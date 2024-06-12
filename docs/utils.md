## calculate_psnr(original, compressed)
Determina la qualitat de la imatge reconstruïda en comparació amb la imatge original calculant el PSNR

- **Args**:
    - `original` (np.ndarray): La matriu de la imatge original.
    - `compressed` (np.ndarray): La matriu de la imatge comprimida.
- **Returns**:
   - `float`: El valor de la PSNR en decibels (dB). Retorna infinit si MSE és zero.


## calculate_compression_factor(original_size, compressed_size)
 Calcula el factor de compressió donada la mida de les dades originals i les comprimides.
   
- **Args**:
   - `original_size` (int): La mida de les dades originals en bytes.
   - `compressed_size` (int): La mida de les dades comprimides en bytes.
 - **Returns**:
   - `float`: El factor de compressió en percentatge.