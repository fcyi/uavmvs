import colmap_read_write_model as crwm


if __name__ == '__main__':
    inputPath = ''
    outputPath = ''
    removeName = []

    crwm.remove_images_based_fileNames(inputPath, outputPath, removeName)