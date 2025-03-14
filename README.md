# A Study on the Effectiveness and Quality of DCGAN-based Adversarial Attacks

---

<p align="center">
    <img src="Assets/CIIC-FCT.png" width="75%"/>
</p>

---

## Description
Deep neural networks (DNNs), while widely used for classification and recognition tasks in computer vision, are vulnerable to adversarial attacks. These attacks craft imperceptible perturbations that can easily mislead DNN models across various real-world scenarios, potentially leading to severe consequences. 

This project explores the use of deep convolutional generative adversarial networks (DCGANs) with an additional encoder to generate adversarial images that can deceive DNN models. We trained the DCGAN using images from four different adversarial attacks with varying perturbation levels and tested them on five DNN models. Our experiments demonstrate that the generated adversarial images achieved a high fooling rate (FR) of up to 91.21%. 

However, we also assessed image quality using the Fréchet Inception Distance (FID) and Learned Perceptual Image Patch Similarity (LPIPS) metrics. Our results indicate that while achieving a high FR is feasible, maintaining image quality is equally important - _yet more challenging_ - for generating effective adversarial examples.

## Repository Structure
```
adversarial-dcgan/
│
├── 🎨 Assets/                  # Logos and other visual assets
├── ⚔️ Attacks/                 # Adversarial attack implementations
├── 🧠 DCGAN/                   # DCGAN and encoder source code
├── 📓 Notebooks/               # Jupyter notebooks with pre-trained models
├── 🧪 Testing/                 # Test scripts and sample evaluations
├── 🙈 .gitignore               # Git ignore file
├── 🛠️ DCGAN-Training.sh        # Shell script for training DCGAN
├── 🛠️ Encoder-Training.sh      # Shell script for training the encoder
├── 📜 README.md                # Project documentation
├── 🚀 Testing.sh               # Test script for validating the implementation
```

## Usage

Reproducing this work is simple. Just follow these steps:

**Prepare the Attack**

- Ensure you have the necessary attack — either the code to generate the perturbation or the perturbation itself in any format.

**Train the DCGAN**

- Run `DCGAN-Training.sh` to generate adversarial images and train the DCGAN on them.
- Want to tweak settings? You can modify the script to change the model, attack type, number of epochs, or delta values.

**Train the Encoder**

- Run `Encoder-Training.sh` to train the encoder using the best checkpoint from the DCGAN training.
- Make sure to specify the correct checkpoint within the script.

**Test & Evaluate**

- Run `Testing.sh` to test and evaluate the generated images.
- Results will be saved in a JSON file for further analysis.

And that's it. The adversarial DCGAN pipeline is ready to go.

## Acknowledgements
This work is funded by [Fundação para a Ciência e a Tecnologia](https://www.fct.pt/) through project UIDB/04524/2020.
