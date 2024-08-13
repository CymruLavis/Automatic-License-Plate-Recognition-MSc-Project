# Automatic License Plate Recognition (ALPR)
## Overview

Automatic License Plate Recognition (ALPR) is a computer vision project that automatically identifies and reads license plates from images or video streams. This project is to research various object detection and optical character recognition models to create an ALPR pipeline to be depoloyed on a Raspberry Pi. YOLOv8n, SSD, Canny Edge Detection were the object detection models explored and EasyOCR, PyTesseract, TrOCR were the OCR models explored. The top performing models, YOLOv8n and TrOCR were chosen based on accuracy and speed and integrated into the ALPR pipeline that deployed on a Raspberry Pi 4

## Features of pipeline

- **Real-time License Plate Detection**: Efficiently detects license plates in real-time from video streams using YOLOv8n.
- **Optical Character Recognition (OCR)**: Uses TrOCR to accurately read license plate characters.
- **High Accuracy**: Optimized for high accuracy in various lighting and weather conditions.
- **Low Power**: Pipeline was deployed on a low power embedded system for smart home use.

## Demo

![image](https://github.com/user-attachments/assets/d354a2fc-dab0-41c9-81bd-147628dd3003) ![image](https://github.com/user-attachments/assets/c4a45f07-a631-4a5b-92f0-73cdc5c4babb)

## Prerequisites

- Python 3.7 or higher
- cv2
- NumPy
- torch
- torchvision
- ultralytics
- roboflow
- transformers
- sentencepiece  

## Findings
An Automatic Licence Plate Recognition (ALPR) system was developed to evaluate whether consumer-level microcontrollers, such as a Raspberry Pi could achieve performance comparable to industry-standard systems. The system aimed to achieve a minimum of 15 frames per second for license plate detection and character recognition in a smart home garage door opening environment. 
Three separate object detection and OCR methods were tested to determine the combination that offered the highest accuracy and fastest inference times for the final system. For object detection, deep learning models YOLOv8n, and SSD as well as traditional canny edge detection were fine tuned with a custom dataset to detect cars and license plates. The YOLO model achieved the highest accuracy with a mean-average-precision score of 0.91 at an Intersection-over-Union threshold of 0.5. It was a close second to the SSD model in inference time, averaging 17ms. Thus, the YOLO model was selected as the optimal object detection model for the ALPR system. 
The OCR models compared were EasyOCR, PyTesseract, and TrOCR. These were evaluated on cropped images from the object detection testing dataset, showing only license plates. TrOCR was the top performing model with a word accuracy of 47%, a character accuracy of 84%, and an average inference time of 120ms. Common errors such as confusing O and 0, S and 5, and 7 and Z caused a sharp decrease in word accuracy. 
Both models were tested in the PyTorch framework on a T4 GPU and optimized for mobile platforms like the Raspberry Pi. Despite optimizations, the Raspberry Pi failed to meet the speed goal of 15fps, achieving 0.5 fps which is not fast enough to be classified as similar performance to the systems used in industry. This was due to the lack of computational power of the microcontroller that not even the vast model optimizations could account for. A possible solution would be to utilize a different microcontroller, such as the NVIDIA Jetson Nano, a GPU powered microcontroller with 128 cores [86] capable of running the ALPR system at an acceptable framerate. 

## Future Considerations
There are several potential improvements for enhancing the project outcomes, such as model accuracy, inference time for real time performance, and system security. 
First, gathering a larger, more balanced dataset would benefit the object detection models. The current dataset was unbalanced, with more instances of cars than license plates, which significantly reduced the SSD model’s ability to detect license plates. The SSD model had a faster inference time than the YOLO model by 5ms. Thus, with better license plate detection, the SSD model would be an ideal choice for enabling the microcontroller to operate at the required speed.
In addition to this, incorporating images with a broader range of environmental conditions would improve the model. The current data set included only daytime images with either flat light or bright sun resulting in the clearest photo possible. Adding images of objects in varying weather conditions and lighting would improve the model’s ability to operate at all times of the day and in any weather.
One of the drawbacks of the OCR models was that they were trained on handwritten and typed text. While research indicated these models could accurately read license plate characters, practical testing proved otherwise. Fine-tuning the OCR models with license plate-specific data would have improved their ability to differentiate between the commonly confused characters such as O and 0. This would enhance the ALPR system’s accuracy in identifying the license plate string, reducing the likelihood of missing crucial character.
Secondly, security for a computer vision based smart garage is a significant concern, as individuals with malicious intent could steal a license plate and gain unauthorized access to a home. The current system includes a defense mechanism that confirms the license plate is within a car bounding box before allowing access, but this is not a fool-proof solution. Adding an extra layer of security by creating a custom classifier with the TensorFlow Cars196 dataset could be beneficial [87]. The Cars196 dataset contains over 16,000 images of 196 different classes of car, allowing the classifier to identify the brand, model, and year of the car. Training a classifier with this dataset would verify that the user’s car is present along with the correct license plate, thereby preventing unauthorized access.
Finally, future research should consider utilizing a microcontroller with greater computational capabilities to improve the system. While the Raspberry Pi is an excellent consumer level electronic that is affordable and low power, it is insufficient to run a real time ALPR system due to its computation constraints. Investing in a more powerful microcontroller, like the Jetson Nano, would allow the system to operate at the desired speeds with the applied optimizations. Although the Jetson Nano is three times the cost of a Raspberry Pi, its superior computational power makes it a worthwhile investment for the system’s performance and reliability.
By upgrading to a microcontroller like the jetson nano, adding custom classifiers for security and utilizing a more balanced dataset, the ALPR system would be better equipped to meet the demands of real-time processing, ultimately improving the overall systems efficiency and effectiveness. 
