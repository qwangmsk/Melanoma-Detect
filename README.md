# Melanoma-Detect

This project uses 50 melanomas and 50 benign nevi from the ISIC dataset to assess GPT-5's performance for melanoma identification. 

These 100 dermoscopic images, along with their metadata, were downloaded using the script download_isic.py. They were originally used by Shifai et al. to evaluate GPT-4V. The ISIC IDs corresponding to these images, which we used for downloading, are provided in the supplementary file of this study.

        Shifai N, Van Doorn R, Malvehy J, Sangers TE. Can ChatGPT vision diagnose melanoma? An exploratory diagnostic accuracy study. 
        Journal of the American Academy of Dermatology 2024;90(5):1057-1059. DOI: 10.1016/j.jaad.2023.12.062.

The melanoma detection of GPT-5 is assessed using OpenAI API interface. The top-three differential diagnoses for all images were assessed using script isic_top3_eval.py. For each image, the script uses a zero-shot prompting approach to present the request to GPT-5 model as follows:

        "Provide the TOP-3 differential diagnoses for this dermoscopic image, "
        "focusing distinguishing between melanoma and benign nevi. Rank them by likelihood."
        "Return a JSON object with key 'differential' = array of exactly 3 items ordered from most likely to least likely; "
        "each item must include: diagnosis (string), confidence (0.0-1.0), and optionally a brief rationale."

The top-one diagnoses were assessed in part using script isic_top1_eval.py, which uses the following prompt to ask GPT-5 to process each dermoscopic image:

        "Return strict JSON matching {is_melanoma:boolean, likelihood:number[0..1], rationale:string}. "
        "If uncertain, still decide but lower likelihood."
