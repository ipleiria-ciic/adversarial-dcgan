## Fooling Rate and Perceptual Similarity: A Study on the Effectiveness and Quality of DCGAN-based Adversarial Attacks

---

<p align="center">
    <img src="Assets/CIIC-FCT.png" width="75%"/>
</p>

---

### Description
Deep neural networks (DNNs), while widely used for classification and recognition tasks in computer vision, are vulnerable to adversarial attacks. These attacks craft imperceptible perturbations that can easily mislead DNN models across various real-world scenarios, potentially leading to severe consequences. This paper explores the use of deep convolutional generative adversarial networks (DCGANs) with an additional encoder to generate adversarial images that can deceive DNN models. We trained the DCGAN using images from four different adversarial attacks with varying perturbation levels and tested them on five DNN models. Our experiments demonstrate that the generated adversarial images achieved a high fooling rate (FR) of up to 91.21%. However, we also assessed image quality using the Fréchet Inception Distance (FID) and Learned Perceptual Image Patch Similarity (LPIPS) metrics. Our results indicate that while achieving a high FR is feasible, maintaining image quality is equally important - _yet more challenging_ - for generating effective adversarial examples.

### Acknowledgements
This work is funded by FCT - Fundação para a Ciência e a Tecnologia, I.P., through project with reference UIDB/04524/2020.