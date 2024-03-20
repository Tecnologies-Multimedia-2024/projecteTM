# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import click


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
@click.option('--nTiles', type=int,
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
def main(input, output, encode, decode, fps, binarization, negative, averaging, nTiles, seekRange, GOP, quality, batch):
    click.echo(f'Input: {input}')
    click.echo(f'Output: {output}')
    click.echo(f'Encode: {encode}')
    click.echo(f'Decode: {decode}')
    click.echo(f'FPS: {fps}')
    click.echo(f'Binarization: {binarization}')
    click.echo(f'Negative: {negative}')
    click.echo(f'Averaging: {averaging}')
    click.echo(f'nTiles: {nTiles}')
    click.echo(f'Seek Range: {seekRange}')
    click.echo(f'GOP: {GOP}')
    click.echo(f'Quality: {quality}')
    click.echo(f'Batch: {batch}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
