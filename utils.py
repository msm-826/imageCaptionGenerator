# Set these path according to project folder in you system
DATASET_TEXT = "Resources/Flickr_8k_text"
DATASET_IMAGES = "Resources/Flickr_8k_Dataset"

# Loading a text file into memory
def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text