import os
import time
import json
import torch
import datetime
import argparse

import Utils

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str)
parser.add_argument("--checkpoint", type=int)
parser.add_argument("--delta", type=float)
args = parser.parse_args()

batch = 128
attack = args.attack
best_epoch = args.checkpoint
delta = f"{args.delta:.02f}"

models = ['AlexNet', 'ResNet18', 'ResNet152', 'VGG16', 'VGG19']

results_file = f"Testing/Results/Evaluation-Results-{attack}.json"
os.makedirs(os.path.dirname(results_file), exist_ok=True)

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
else:
    results = []

device = Utils.use_device()

for model_name in models:
    start_time = time.time()

    print(f"[ INFO ] Starting testing for {attack} with delta {delta} against model {model_name}...")

    # Original and adversarial paths.
    original_inference_path = "Dataset/Imagewoof/train"
    adversarial_inference_path = f"Images/Generated-Images-{attack}-Batch-{batch}-Delta-{delta}"

    # Loads the G checkpoint in a given iteration.
    checkpoint_name_g = f"DCGAN/{attack}/Models/Delta-{delta}/Best-Checkpoint-{attack}-Delta-{delta}-Epoch-{best_epoch}-128.pth"
    checkpoint_g = Utils.fetch_checkpoint(checkpoint_name_g, device)

    # Creates the Generator class and loads it with the dictionary from the checkpoint.
    generator_network = Utils.Generator(1, 64, 3, 100).to(device)
    generator_network.load_state_dict(checkpoint_g['netG_state_dict'])
    generator_network.eval()

    # Loads the E checkpoint in a given iteration.
    checkpoint_name_e = f"DCGAN/{attack}/Models/Encoder-Delta-{delta}/Encoder.pth"
    checkpoint_e = Utils.fetch_checkpoint(checkpoint_name_e, device)

    # Creates the Encoder class and loads it with the dictionary from the best checkpoint.
    encoder_network = Utils.Encoder(64, 3).to(device)
    generator_network.load_state_dict(checkpoint_g['netG_state_dict'])
    generator_network.eval()

    # Creates the Dataloader with the images from the DCGAN-based.
    dataloader = Utils.custom_dataloader(original_inference_path)

    # Generate the images from the dataloader and the Generator class.
    Utils.generate_images(dataloader, attack, batch, delta, generator_network, device)

    # Load the model to evaluate the images.
    model = Utils.load_model(model_name, device)

    # Load both original and adversarial dataloaders.
    original_loader = Utils.load_dataset(original_inference_path)
    adversarial_loader = Utils.load_dataset(adversarial_inference_path)

    # Compute original predictions.
    orig_preds = Utils.classify_images(model, original_loader, device, title="Classifing the real images")
    correctly_classified = {k: v[0] for k, v in orig_preds.items() if v[0] == v[1]}

    # Compute adversarial predictions.
    adv_preds = Utils.classify_images(model, adversarial_loader, device, title="Classifing the adversarial images")
    fooling_count = sum(1 for k, v in correctly_classified.items() if k in adv_preds and adv_preds[k][0] != v)
    fooling_rate = fooling_count / len(correctly_classified) if correctly_classified else 0

    # Compute the LPIPS metric.
    lpips_score = Utils.calculate_lpips(original_loader, adversarial_loader, device, attack, delta, model_name)

    # Compute the FID metric.
    fid_score = Utils.fid(real_dataset_path=original_inference_path, generated_dataset_path=adversarial_inference_path, device=device)
    
    print(f"[ \033[92mRESULTS\033[0m ] Correctly classified original images: {len(correctly_classified)}")
    print(f"[ \033[92mRESULTS\033[0m ] Fooling Rate (FR): {fooling_rate:.2f} ({fooling_count})")
    print(f"[ \033[92mRESULTS\033[0m ] FID Score: {fid_score:.2f}")
    print(f"[ \033[92mRESULTS\033[0m ] LPIPS Score: {lpips_score:.2f}")

    # Save results to dictionary.
    iteration_results = {
        "model": model_name,
        "delta": delta,
        "correctly_classified": len(correctly_classified),
        "fooling_rate": fooling_rate,
        "fooling_count": fooling_count,
        "fid_score": fid_score,
        "lpips_score": lpips_score
    }

    # Append results and save to JSON.
    results.append(iteration_results)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    # Deletes every cache that was used.
    del checkpoint_g, checkpoint_e, generator_network, dataloader, model, original_loader, adversarial_loader, orig_preds, adv_preds
    torch.cuda.empty_cache() 
    torch.cuda.synchronize()
    
    elapsed = time.time() - start_time

    print(f"[ INFO ] All the tasks completed in {str(datetime.timedelta(seconds=int(elapsed)))}.")