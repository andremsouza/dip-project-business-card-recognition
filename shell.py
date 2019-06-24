import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
import corner_detection
import perspective_transform
import text_detection
import cv2
import re
import random


def showSamples(files,randomFiles,size=9):
    """Show to the user some samples business cards selected randomly.

        Args:
            files: A list of business card filenames.
            randomFiles: A list of indexes of business cards selected randomly.
            size: size of cards to be showed to the user. Fixed to 9 because of matplot lib limit.

        """
    plt.figure(figsize=(30, 30))
    line = col = int(size/3)
    j = 1
    for i in randomFiles:
        img = imageio.imread(files[i])
        position = int(f"{line}{col}{j}")
        j = j+1
        plt.subplot(position)
        plt.axis("off")
        plt.imshow(img)
    plt.show()

def selectRandomFiles(files):
    """Select randomly 9 business cards 
        to be showned to the user in order for him to choose which wants to process.

        Args:
            files: List of the business cards filenames.
        Returns:
            list: A list of the indexes of the bussiness cards selected randomly.
        """
    randomFiles = []
    while len(randomFiles) < 9:
        index = np.random.randint(0,len(files))
        if randomFiles.count(index) == 0:
            randomFiles.append(index)
    return randomFiles

#Process the card detecting corners, repairing perspective and with opencv detect text region and try to extract informations.
def processCard(selectedCard):

    img = imageio.imread(selectedCard)
    print("Detecting card corners....")
    img1_corners = corner_detection.CornerDetector(img).corner_detector()[0]
    plt.figure(figsize=(20, 20))
    plt.imshow(img1_corners)
    plt.show()

    print("Repairing card perspective....")
    corner_points = corner_detection.CornerDetector(img).find_corners4().astype(
    np.float32)
    corner_points[:, [0, 1]] = corner_points[:, [1, 0]]
    img1_t = perspective_transform.PerspectiveTransform(
    img, corner_points).four_point_transform()
    plt.figure(figsize=(20, 20))
    plt.imshow(img1_t)
    plt.show()

    print("Detecting text in card...")
    img1_t_cv = cv2.cvtColor(img1_t, cv2.COLOR_RGB2BGR)

    strs, bboxes1, _ = text_detection.TextDetector(img1_t_cv,(30, 10)).recognize_text()
    
    for box in bboxes1:
        x, y, w, h = box
        cv2.rectangle(img1_t_cv, (x, y), (x + w, y + h), (0, 0, 255), 3, 8, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img1_t_cv, cv2.COLOR_BGR2RGB))
    plt.show()

    for text in strs:
        if(len(text) != 0):
            print(text)

    print("Recognition completed.")

#Main program of the shell which read continously the user commands and process them.
def shell():
    selectedCard = 0
    usageMessage = """
        Usage:
        The shell supports the next commands:
            - exit: finish the program and go out from the shell
            - samples: show to the user 9 business cards selected randomly from the sample data set.
            - selectcard <number>: select a businness card giving a number between 1 and 9.
            - run: perform the process to analise and extract the business card informations.
            - help: show this message.
    """
    print(usageMessage)
    example_files = [
                './images/' + f for f in os.listdir('./images')
                if os.path.isfile(os.path.join('./images', f))
            ]
    randomFiles = []

    while(True):
        cmd  = str(input().rstrip())
        if re.fullmatch(r"exit",cmd):
            break
        elif re.fullmatch(r"samples",cmd):
            randomFiles = selectRandomFiles(example_files)
            showSamples(example_files,randomFiles)
        elif re.fullmatch(r"selectcard [1-9]",cmd):
            selectedCard = int(re.search(r"[1-9]",cmd).group(0))
            print(selectedCard)
            if selectedCard <= 0 or selectedCard > 9:
                print("Please select a card between 1 and 9")
                selectedCard = 0
        elif re.fullmatch(r"run",cmd):
            if selectedCard == 0:
                print("Please select first a card before running.")
            else:
                index = randomFiles[selectedCard-1]
                processCard(example_files[index])
        elif re.fullmatch(r"help", cmd):
            print(usa)
        else:
            print("Command not recognized. Please, try again.")
        
def main():
    shell()

if __name__ == "__main__":
    main()