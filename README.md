# Projecte Tecnologies Múltimedia


Integrants de l'equip:
- Marta Bernadas Portas
- Núria Pallejà Algueró

Com executar el projecte?
1. Obrir la terminal.
2. Creem l'entorn virtual:
python -m venv myenv
3. Activem l'entorn virtual:
.\myenv\Scripts\activate
4. Instalem tots els paquets necessaris per a executar el projecte:
python setup.py install
5. Executem el projecte:
python -m tmproject --input Cubo.zip 


## PROVES A FER A LA DEMO
1. `python -m tmproject --help`
2. `python -m tmproject --input Cubo.zip`
3. `python -m tmproject --input Cubo.zip --fps 100`
4. `python -m tmproject --input pedro.gif`
5. `python -m tmproject --input Cubo.zip --filters "binarization"`
6. `python -m tmproject --input Cubo.zip --filters "negative"`
7. `python -m tmproject --input Cubo.zip --conv_filters "sobel"`
8. `python -m tmproject --input Cubo.zip --conv_filters "sharpening"`
9. `python -m tmproject --input Cubo.zip --ntiles 4 --output "prova.zip" --GOP 2 --seekRange 4 --quality 0.9`
10. `python -m tmproject --input Cubo.zip --ntiles 8 --output "prova.zip" --GOP 2 --seekRange 4 --quality 0.9`
11. `python -m tmproject --input Cubo.zip --ntiles 4 --output "prova.zip" --GOP 8 --seekRange 8 --quality 0.9`
