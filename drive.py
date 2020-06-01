def mount_drive():
	from google.colab import drive
	drive.mount('/content/gdrive')

	my_drive = '/content/gdrive/My Drive/'
	image_folder = my_drive + 'TestImages/'
	training_folder = my_drive + "Traning/"
	return my_drive, image_folder, training_folder