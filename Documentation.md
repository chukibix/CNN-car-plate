# Computer Vision model to detect vehicule types and license plate



## Table of Contents

- [system architecture](#Architecture)

- [Detection model](#fastercnn)

- [Tracking algo](#deepsort)

- [Detection plate](#detection_plate_model)

- [Read Plate](#read_plate_model)

- [challenges](#overall_challenges)

- [usages](#usage)

- [improvement](#improvement)



## Architecture

        We will discuss our overall system, functionality and explain how each model interact with each other.

        The system consist of 4 main operation:
                - Detect vehcule with type.
                - Track each one of our detection between frame.
                - Detect possible visible plate for each vehicule.
                - Read the detected plate.

The first step, was to first detect in each frame of our video, the vehicule,coordination, and classify each detection: car,Motorcycle or truck. 
For this part, Faster R-CNN was the model choosen because according to research it s more suited for precised task amd detailed task, in comparison to yolo. however it comes with a drawback of taking more time.

After getting the detections and storing them with theirs informations: coordiantion scores, and labels.
they will be passed to our trcking algorithm, to pin an Id to each detected vehicule in future frames.
we went for Deepsort.

Then, for each tracks of  detected vehicule:
   - we extracted coordination of the vehicule.

   - cropped the respectif frame.to only focus on this specific vehicle

   - passed the frame to a model based on detecting a possible license plate.
      - case 'no plate' : returns none.
      - case 'found plate' : get the coordination of the plate FROM THE CAR FRAME CROPPED, and pass the frame 
                            to our OCR .




## fastercnn
for our first pretrained-model, we chose faster cnn over yolo for the reason stated above:tradind time for better accuracy. freindly to use and set up.
faster cnn can detect multiple object, inside coco_classes. So according to our need we set the indexes wanted: car, truck amd motocycles.
for each detection we got: boxes , scores of detection and it labels.


how ever i ran into a problem, where certain vehicle, like sem-truck would sometimes be counted as truck and car at the same time.and mess up our tracking.
So i added nms threshold, to minimize overlapping of same vehicle. this solved the problem.

another weakness appeared when passing videos of cars with their shadows, it would take for a couples of frames the shadows as a part of the car frame, or even count it as a different car.
nms minimize this situation, but wasnt succesfull as the semi truck problem.



## DeepSort
for tracking, i used the result of the detections stored in a specific format needed from deepSort.
updated the traker with new detection, for the cureent frame, and passing in paramets its informations.

i would like to point that deep sort was hard to set, most documentation did not help me due to different version , including chatgpt,  they all worked on 'install deepsort' which, in my case collab was giving error of 'not found. i had to use 'install deep-sort-realtime' which does not have the same function and attributes.

So i had to search,trial and error a lot, my main issues where : 
    - tracking was altering coordinations for "prediction" reason,messing up completly my detections result.
    - unable to get informations related to track (label and score).

At the end, i solved by printing debugging and online documentation :
    - https://pypi.org/project/deep-sort-realtime/#storing-supplementary-info-of-original-detection
    - added parameters to tracking org=true  to save original coordination.
    - get_det_supplementary() to get all additional info.

as for the parameters used :
    max_cosine_distance=1.0,             # Maximum cosine distance for association
    nms_max_overlap=1.0,                 # Maximum overlap for Non-Maximum Suppression
    max_iou_distance=0.7,                # Maximum IoU distance for association

the goals of the firt three was to solve an issue, that sometimes, the same objcet would be tracked twice or more.it narrowed this issue a bit, but still present.

    max_age=35,      
     Maximum age before a track is considered lost
    n_init=5,                 
     Minimum number of frames to confirm a track
    nn_budget=300         
     the number of factor to take in consideration when acomparing track.

so after tracking we pass the tracked frame to our detecting and reading plate.(we will see them later)

and then with the result, draw a box on the main frame for each detection, with it s imformation.



## detection_plate_model

As stated above,after cropping the frame to have only the detetion, we want to detect a possible present license plate.So we decided to build our own model.
we used Yolo

I needed a dataset that had good data,different angle and shape, plus for time management reason not very big.

https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data

the hard part was the anootations format , xml.files. i was new to this , as i was new to yolo training(training yolo to our need).
So online demos and examples did help a lot to get throught these issue, how to extract informations from xml files, how to store my dataset in traing/test/eval set to yolo format.

parameters : 

    epochs=110,           
       Number of training epochs
    batch=16,             
       Batch size
    
    imgsz=320,            
       Image size (width and height) for training
       used for better details

    cache=True             
       Cache images for faster training 

A detection(plate frame cropped) would be then passed to our reading_plate model.
in case of no detection, it will return none, and a score result of 0.0


     Results to note after training:

- Precision for bounding box predictions, is 0.902 . This indicates how many of the detected objects are true positives.
-  Recall, which is 0.835 . This measures how many of the actual objects in the validation set are correctly detected by the model.

The model appears to perform well, with good precision and recall, showing that it correctly detects and classifies license plate.



## read_plate_model

So the model before, focus on cropping the plate frame only.

For reading at first, i used a pretrained model from Microsoft's TrOCR .
however the result were not good, and later with more research found out that is not well designed for this task.So again we opted to make our own model, whole task to read license plate.

we needed dataset designed only on plates(no need for whole car), and we decided on the characters A-Z  0-9.

https://www.kaggle.com/code/infistforever/vehicle-plate-ocr/input
from this dataset i used "Car License Plate Detection" file,"License Plate Characters - Detection" and"tags_licenseplate".

for the "License Plate Characters - Detection" dataset, indian license plate. Include close picture of only the license plate.with different angle.


0  188    1  139    2  128    3  112    4  124    5  116
6  118    7   83    8   76    9  103    A   67    B   51
C   54    D   30    E   34    F   13    G   16    H  108
I    4    J   16    K   60    L   59    M   77    N   41
O    6    P   24    Q    2    R   43    S   18    T   59
U   15    V    9    W   14    X    7    Y    8    Z    4

each caracter count,cover most characters, however more balance can be helpful.

and the second dataset, "Car License Plate Detection" was also used to read license plate, the difference is that this one, the cars are included, so zoom in will happen.Simulating the whole project.

same to detection license plate we had to work on xml format, and merge both dataset.then train our model.
we  had in total around 500 plates to work on, mostly indians.

parameters for model:

-processor.tokenizer.sep_token_id
ensures that the model knows when to stop generating text. In OCR tasks, this helps to stop generating characters once the license plate text or any text in the image has been fully recognized

-model.config.max_length = 64
reduce useless computution. license plate are much shorter. 64 is enough.

-model.config.early_stopping = True
  save time computation.

-model.config.no_repeat_ngram_size = 3
  help avoid repetion and useless text.

-model.config.length_penalty = 2.0
 helps with accuracy. it wont generate a lot of long sequences.

-model.config.num_beams = 4
a higher chance of finding the most optimal sequence.

the metric set is "cer" Character Error Rate . the goal is to produce text that matches our plate.

finalyy we saved our model, and the result were a lot better than our first testing.

  Results to note after training:
- CER (Character Error Rate) - 0.127660:  which 12% of characters are incorect, which is good, but can be better.



## overall_challenges

One of the chalenges faced, was that between frame, license plate result can be diffferent,and the last result will always overide the one before.So how to keep the best one?

this is where we implemented a dictionary, representing  a key and an array value having 4 values.
the key is the track id, representing the same vehicle.
and the array values : scores of label[0], and labels[1].   vehicle type and the probabilities of the result
                       scores of plates[2], and plate[3].   plate number and it s probability,

So each loop, we will compare scores gotten, and store the better value.
this proved to be, effcient and solved this concern.
however there was still some weakness, for example :

A plate,let s say A1B2C3, at a side angle comming showing half of the plate, only getting A1B,getting a high score of 0.85, in future frames showing the whole plate, This score will be hard to surpass due to more incertainty.even tho it read the whole plate correctly but with a 0.80 score.


Another small challenges was handling the images/frame passed between model, reformating or resizing each frame to accomodate models.
for example fastern cnn returnes detetctions was not compatible with the deeptracking function, it needed a specific format.
this was solved by print debugging, and reading source code.



## usage

even if Our project license plate was trained on indian plate, after testing it did work on lebanese license plate. So it can work in countries having a similar styles of writing.
it did good in night time, but streetlight presence is preferable.

camera height does impact the reading result, the higher it is, the harder it is to read.
As for the angle,to treat the weakness spoken in overall_challenges, it should be straight to the nose/back of the car. catch the whole plate characters at the same time.
A side angle is harder to get the number of the plate.



## improvement

for future improvements, i would like more stability on the detections side, maybe more precise labeling for diverse vehicle, and handle detections anomalies, like car shadows caused by the sun or even reflection, counting as different car(i had a rare occurence of my model detecting a second car from the side miror of the main car).

improving detections will impact tracking, since they pass results,but we can definatly improve tracking alone.
even with parameters, trading time for better result, it can be a bit shaky when it comes tracking new vehicles entering the video, it give sometimes two id.Or a bad frames can throw off the tracking into thinking it's a different vehicle.

i would improve my detectionn plate model to know when the whole plate is in the frame,choose the best result better.and improve reading model more to lower CER
improvise results on these situation: further distance, very fast cars, broken plate.


Also,priotizing accurate result/perfomance did cone with slower time for a response.
maybe knowing in advance inforamtions about the cameras used,position, can help us cut corners in our architecture and reduce computation. 

And overall ,add more informations, speed of the car, time of entering/leaving.

althought it is not in our field of work, hardware improvement can be very benificial. 
Upgrading to high-quality cameras enhances computer vision systems. Higher resolution captures more detail, crucial for accurate license plate recognition. Improved low-light performance and dynamic range ensure better detection in varied lighting conditions, making the system more effective.