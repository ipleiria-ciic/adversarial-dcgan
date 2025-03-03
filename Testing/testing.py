import os
import time
import json
import torch
import utils
import datetime

if os.path.exists("Evaluation_Results.json"):
    with open("Evaluation_Results.json", "r") as f:
        results = json.load(f)
else:
    results = []

device = utils.use_device()

for iteration in range(0, 10001, 1000):
    start_time = time.time()

    print(f"[ INFO ] Starting iteration {iteration}...")

    # Loads the checkpoint in a given iteration.
    checkpoint = utils.fetch_checkpoint(f"../DCGAN/FGSM/Models/Checkpoint-FGSM-Epoch-{iteration}-64.pth", device)

    # Creates the Generator class and loads it with the dictionary from the checkpoint.
    generator_network = utils.Generator(1, 64, 3, 100).to(device)
    generator_network.load_state_dict(checkpoint['netG_state_dict'])
    generator_network.eval()

    # Creates the Dataloader with the images from the DCGAN-based FGSM attack.
    dataloader = utils.custom_dataloader("FGSM")

    # Generate the images from the dataloader and the Generator class.
    utils.generate_images(dataloader, iteration, generator_network, device)

    # Load the model to evaluate the images.
    model = utils.load_model("ResNet152", device)

    # Load both original and adversarial dataloaders.
    original_loader = utils.load_dataset("../Dataset/Imagewoof/train")
    adversarial_loader = utils.load_dataset(f"Generated-Images-{iteration}")

    # Compute original predictions.
    orig_preds = utils.classify_images(model, original_loader, device, title="Classifing the real images")
    correctly_classified = {k: v[0] for k, v in orig_preds.items() if v[0] == v[1]}

    # Compute adversarial predictions.
    adv_preds = utils.classify_images(model, adversarial_loader, device, title="Classifing the adversarial images")
    fooling_count = sum(1 for k, v in correctly_classified.items() if k in adv_preds and adv_preds[k][0] != v)
    fooling_rate = fooling_count / len(correctly_classified) if correctly_classified else 0

    # Compute the LPIPS metric.
    lpips_score = utils.calculate_lpips(original_loader, adversarial_loader, device)

    # Compute the FID metric.
    fid_score = utils.fid(real_dataset_path="../Dataset/Imagewoof/train", generated_dataset_path=f"Generated-Images-{iteration}", device=device)
    
    print(f"[ \033[92mRESULTS\033[0m ] Correctly classified original images: {len(correctly_classified)}")
    print(f"[ \033[92mRESULTS\033[0m ] Fooling Rate (FR): {fooling_rate:.2f} ({fooling_count})")
    print(f"[ \033[92mRESULTS\033[0m ] FID Score: {fid_score:.2f}")
    print(f"[ \033[92mRESULTS\033[0m ] LPIPS Score: {lpips_score:.2f}")

    # Save results to dictionary.
    iteration_results = {
        "iteration": iteration,
        "correctly_classified": len(correctly_classified),
        "fooling_rate": fooling_rate,
        "fooling_count": fooling_count,
        "fid_score": fid_score,
        "lpips_score": lpips_score
    }

    # Append results and save to JSON.
    results.append(iteration_results)
    with open("Evaluation_Results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Deletes every cache that was used.
    del checkpoint, generator_network, dataloader, model, original_loader, adversarial_loader, orig_preds, adv_preds
    torch.cuda.empty_cache() 
    torch.cuda.synchronize()
    
    elapsed = time.time() - start_time

    print(f"[ INFO ] All the tasks completed in {str(datetime.timedelta(seconds=int(elapsed)))}.")
    print("---")
