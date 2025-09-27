# Melanoma-Detect

## Dermoscopic Images
This project uses two popular datasets, ISIC Archive (https://api.isic-archive.com/images/) and HAM10K dataset (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), to assess GPT-5's performance in melanoma diagnosis. 

A previous study by Shifai et al. randomly selected 50 melanomas and 50 benign nevi from ISIC to benchmark GPT-4V (PMID: 38244612, DOI: 10.1016/j.jaad.2023.12.062). To make our assessment comparable with theirs, we obtained the ISIC image identifiers from the paper, with which we retrieved these 100 dermoscopic images from ISIC, along with the corresponding metadata, using our Python script download_images.py. The ISIC IDs of these images can be found in the supplementary file of this paper:

        Shifai N, Van Doorn R, Malvehy J, Sangers TE. Can ChatGPT vision diagnose 
        melanoma? Anexploratory diagnostic accuracy study. Journal of the American 
        Academy of Dermatology, 2024;90(5):1057-1059. 

From the HAM10K dataset, a recent study randomly selected 500 dermoscopic images (PMID: 40117499, DOI: 10.2196/67551) to evaluate GPT-4T and GPT-4o. However, the study did not disclose the identifiers of the selected images. Following the description in that paper, we independently sampled 500 images from HAM10K to construct our test dataset. Given the independent sampling, the overlap between the two sets is expected to be minimal. To ensure reproducibility, we have made the identifiers of our 500 images publicly available on this site.

        Sattler SS, Chetla N, Chen M, et al. Evaluating the Diagnostic Accuracy
        of ChatGPT-4 Omni and ChatGPT-4 Turbo in Identifying Melanoma: Comparative 
        Study. JMIR Dermatology 2025;8:e67551-e67551. DOI: 10.2196/67551. 

The script download_images.py saves metadata into a file, isic_metadata.xlsx, to be used in downstream analysis.
        
## GPT-5 Assessment
The melanoma detection of GPT-5 is assessed using OpenAI API interface. The top-three differential diagnoses for all images were assessed using script isic_top3_eval.py. For each image, the script uses a zero-shot prompting approach to present the request to GPT-5 model as follows:

        Provide a ranked differential diagnosis, listing three potential diagnoses 
        from most to least likely based on this dermoscopic image. Return a JSON
        object with key 'differential' = array of exactly 3 items ordered from most 
        likely to least likely; each item must include: diagnosis (string), 
        confidence (0.0-1.0), and optionally a brief rationale.

We used the following command to run isic_top3_eval.py: 

        python isic_top3_eval.py --images isic_images \ 
                --meta isic_metadata.xlsx --sheet "Sheet1" \
                --out isic_out/preds-t3 --model gpt-5 \
                --truth-col "metadata.clinical.diagnosis_1"

where isic_metadata.xlsx is the file that stores the metadata of these images.

The top-one diagnoses were assessed in part using script isic_top1_eval.py, which uses the following prompt to ask GPT-5 to process each dermoscopic image:

        "Return strict JSON matching {is_melanoma:boolean, likelihood:number[0..1], rationale:string}. "
        "If uncertain, still decide but lower likelihood."
