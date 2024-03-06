# Task description

Implement two image checkers:
- if there is a face in the image,
- if there is a child in the image.

Assumptions:
- there is only one person in the uploaded photos,
- uploaded photos are in portrait format.


# Thought Process

Colab notebook solution: [here](https://colab.research.google.com/drive/11DQIadiFSDEzShFCabw_indxmXo-AB1Y?usp=sharing)


### Solution 1
In order to build a solution for this task, I thought of it as a pipeline of two steps:

1. Detect the face (if any) in the image
    - From the 1st constraint, there will always be one person in the image.
    - So, there will always be atleast one face in the image.
2. Predict the age of the person from the detected face
    - From the 2nd constraint, age prediction models (CNN arch) will perform better.
    - Due to the photos being in potrait format. No other distractions.


From a quick research I found 3 options for the first step:

1. [OpenCV](https://opencv.org) (for face detection)
2. [YOLOv9](https://arxiv.org/abs/2402.13616) (for object detection, overkill)
3. [MTCNN](https://arxiv.org/abs/1604.02878) (face detection, fast and robust)

I decided to go with MTCNN, due to it's accuracy and faster performance speed.

For age prediction, I came across `Age Net` model which is based on AlexNet architecture for predicting/classifying the age range of a particular human face.

Considering that a person with age 0-14 is a child, I check this condition in the code.


> Note: predicting age is really a subjective task. A person may "look" a certain age in a photo, but they might not actually be that age in real life. However, the input space reduces when the task is about predicting only "child" or "not child". There will still be some outliers though, which is a limitation.

Time taken: 2 hours

### Solution 2
Essentially, for this problem I thought of another solution to use a Vision Language Model (VLM).

VLMs are trained on a large amount of text and image datasets. They have both, a text encoder and an image encoder. On a lower level, they have transformers architecture (using attention mechanisms).

The benefit of a VLM is they can really "see" the image since they calculate self-attention between image features, and also cross-attention between text and image tokens. CNNs can see "local" regions, but VLMs can process long range image tokens due to attention.

I decided to use [moondreamv2](https://huggingface.co/vikhyatk/moondream2) which is a tiny VLM based on SigLip and Phi1.5

> Note: transformers suffer a lot from hallucinations which is a limitation. However, due to the given constraints, the sample space for image tokens reduces. Also, I implicitly ask the VLM to generate at most 3-5 tokens. This reduces hallucinations to some degree.

Time taken: 1 hour 30 minutes


### Combining both

It might be possible that `AgeNet` predicts the face to be of a child, while `moondreamv2` predicts it to be not a child, and vice versa.

One way to overcome this is to use a confidence score for AgeNet. Let's say $c$.

Let the boolean output from `moondreamv2` be $M$. Let the boolean output from `AgeNet` be $A$.

We can simply combine both outputs to predict if the person is a child:

$$ 
\text{child} = 
     \begin{cases}
       A, &\quad c \ge threshold\\
       M, &\quad\text{otherwise.} \\ 
     \end{cases}
$$

Here $threshold$ can be higher float value representing the confidence threshold, $80-90$%.

> We give higher priority to `moondreamv2` for lower AgeNet confidence, due to the success of the VLM architecture.

Time taken: 30 minutes