# Project Write-Up: AI Plant Disease Detection

This is a write-up of the AI Plant Disease Detection project, which I worked on as a team member. It covers what the system does, the data we used, the two models we trained, how we prepared the images, how the model is served to users, and the results.

## What the project does, in simple words

The idea is simple: a person takes a photo of a leaf, and the system tells them whether the plant looks healthy or which illness it most likely has. A farmer or gardener doesn't need to be an expert. They upload a picture, wait under a second, and get an answer on a screen. Catching a sick crop early can save a harvest, so speed and accuracy both matter here.

Under the hood, this is an image classification problem. The computer looks at the colours, spots, edges and textures on a leaf and decides which category the picture belongs to.

## The data

We trained on the PlantVillage dataset, using more than 54,000 leaf images spread across 38 distinct classes. Each class is a combination of a plant and a condition, so the model has to tell apart not just "sick versus healthy" but exactly which disease is showing on which plant. With 38 classes, guessing randomly would be right less than 3% of the time, so the model really has to learn the visual patterns.

## Two models: a custom CNN and MobileNetV2

We built the models with TensorFlow and Keras, and we tried two approaches.

**1. A custom convolutional neural network (CNN).** A CNN learns small filters that slide across an image and respond to patterns like edges, blotches and colour changes. Early layers pick up simple shapes, and deeper layers combine them into more meaningful features, such as the ring-shaped spots of a particular leaf disease. Building our own CNN from scratch was a good way to understand what each layer contributes.

**2. Transfer learning with MobileNetV2.** MobileNetV2 is a network that has already been trained on a huge general image collection, so it already "knows" how to see edges, textures and shapes. Instead of starting from zero, we fine-tuned this pre-trained network on our leaf images. It's like hiring someone who already knows how to read and only teaching them a new subject. MobileNetV2 is also designed to be lightweight, which keeps prediction fast.

For training we used the Adam optimizer and categorical cross-entropy loss. Adam adjusts the learning rate for each weight automatically, which usually makes training stable without much tuning. Categorical cross-entropy is the standard loss for problems where each image belongs to exactly one of many classes: it punishes the model more when it is confidently wrong.

## Results

Transfer learning clearly won. The MobileNetV2 model reached **94.45% validation accuracy**. Even more striking, it hit **91% accuracy after the very first epoch**. That fast convergence is the whole point of transfer learning: the network already understood images in general, so it only needed a little nudging to understand leaves. A model trained from scratch has to learn everything, including what an edge looks like, before it can be useful.

## Preparing the images

Raw photos come in all sizes and lighting conditions, so we built a preprocessing and augmentation pipeline:

- **Resizing to 224x224 pixels**, the input size MobileNetV2 expects.
- **Normalization**, scaling pixel values into a small consistent range so training is numerically stable.
- **Rotation, flipping and shearing**, which create slightly altered copies of training images.

Augmentation matters because a model can otherwise memorise the exact training photos instead of learning the general pattern, which is called overfitting. If the model sees the same leaf rotated, mirrored and slightly skewed, it learns that the disease is the spots themselves, not the angle the photo was taken from. This makes it more reliable on new photos taken by real users.

## From model to a usable app

A model sitting in a notebook doesn't help anyone, so we built a full-stack deployment. The trained model is saved as an .h5 file and loaded by a **Flask API** on the backend. The frontend is a **React dashboard** where the user uploads an image. The dashboard sends the image to the Flask API, the API runs it through the model, and the prediction comes back to the screen. Real-time inference takes about **0.8 seconds** per image, which feels instant for the user.

## What this project shows

This project combines machine learning with turning a model into a working product. Technically, the main takeaway is that a strong pre-trained model like MobileNetV2 can reach high accuracy very quickly, instead of reinventing the wheel with a model built from scratch. It also fits the advice I give juniors: build projects that solve significant modern problems, and crop health is one of them.

## Plain summary

In short, our team made a program that looks at pictures of plant leaves and recognises 38 different health conditions. We taught it using tens of thousands of example photos, reused a ready-made vision network to learn faster, got it right about 94 out of 100 times on unseen photos, and wrapped it in a website that gives an answer in under a second.
