## GPT-5.2 melanoma diagnosis performance across skin tones

<!--The objective of this project is to comprehensively evaluate the performance of the newly released GPT-5 (in Section 1) and GPT-5.2 (in Section 2) for melanoma detection.
-->
## System setup

To run the code under this folder, a valid OpenAI API account and an API key are required. You can follow the following steps to set up your running environment:

1. Sign up at the OpenAI API platform.
2. Set up your payment method.
3. Generate an API key at https://platform.openai.com/api-keys, if you don't have it yet. 
4. Save your key as a global environment variable, OPENAI_API_KEY, so you can access across various applications and scripts on your system without hardcoding it.

### Data sources
The ISIC Archive and HAM10K dataset, although widely used, predominantly contain images from light-skinned individuals and lacks standardized skin tone annotations, limiting their suitability for assessing ChatGPT's robustness across diverse populations. 

After surveying dermatology image datasets, we identified <strong>Milk10K</strong> as a suitable resource for evaluating GPT diagnostic performance across skin tones. We were unable to obtain access to the Diverse Dermatology Images (DDI) dataset during the project period. All dermoscopic images, clinical close-up, and metadata of Milk10K are publically available through the ISIC Archive, Kaggle, and can be obtained directly from https://api.isic-archive.com/doi/milk10k/. 

From the Milk10K dataset, we randomly selected 92 lesions per skin tone class to construct a balanced subset for evaluating GPT-5.2. This subset comprises 460 unique lesions (92 per skin tone group) and 920 images in total. To ensure reproducibility, we provided the identifiers of the selected images in file, milk10k-460-image-ids.csv, which were used consistently across all experiments.

### Prompting

The GPT-5.2 model was used in this assessment. Because our earlier results on the ISIC and HAM100K datasets indicated that GPT-5 was not well suited for top-1 diagnosis, the present evaluation focused on two clinically relevant diagnostic tasks: (1) malignancy discrimination, and (2) top-three differential diagnoses. 

(1) For each skin lesion, we used the zero-shot prompting approach to submit requests to the GPT-5.2 model via the OpenAI API interface. A standardized and formal prompt format was applied to ensure consistency across evaluations. The prompt used for malignancy discrimination in the dermoscopy-only scenarios is provided below:

<!--* Dermoscopy only-->

       Task: classify the lesion as Malignant or Benign based on this dermoscopic image.
       Return ONLY valid JSON with keys:
         pred: 'Malignant' or 'Benign'
         confidence: number from 0 to 1
       No extra keys. No prose.
<!--* Dermoscopy plus clinical close-up

       Task: classify the lesion as Malignant or Benign based on this dermoscopic image
             and the clinical close-up.
       Return ONLY valid JSON with keys:
         pred: 'Malignant' or 'Benign'
         confidence: number from 0 to 1
       No extra keys. No prose.-->

(2) We tested different prompts and found minor variations in prompt wording did not materially affect the outcomes. So we used a single standardized prompt to generate the top-3 differential diagnoses for both scenarios to maintain simplicity and consistency, as follows:

       You are evaluating a skin lesion based on a dermoscopic image 
             (along with clinical close-up if provided).
       Task: Provide an ordered Top-3 differential diagnosis list 
           (most to least likely) for the lesion shown.

       Return ONLY valid JSON with exactly this key:
         differential: [
           {"diagnosis": "...", "confidence": 0.0},
           {"diagnosis": "...", "confidence": 0.0},
           {"diagnosis": "...", "confidence": 0.0}
         ]
       Rules:
       - Provide exactly 3 items.
       - 'confidence' must be a number in [0,1] and non-increasing.
       - Strict JSON only (double quotes). No extra keys. No prose. No code fences.

### Assessment

The malignancy discrimination were assessed using script milk10k_malignancy_eval.py. In current version, the image folder and metadata file are hardcoded. After setting correct file paths, run the following command and GPT diagnosis results will be automatically collected, processed, and stored in two seperate files, gpt52_milk10k_derm_only_predictions.csv and gpt52_milk10k_derm_plus_clin_predictions.csv.

        python milk10k_malignancy_eval.py

The top-three differential diagnoses of GPT-5.2 were conducted using script milk10k_top3_eval.py. Below is the command used:

        python milk10k_top3_eval.py

### Top-3 differential diagnostic performance 
A summary of GPT-5.2 performance in top-3 differential diagnosis accross skin tones:   

![Figure](../images/Figure_5.png)

### Publication

Our results suggest that GPT-5.2 exhibits stable melanoma-related diagnostic performance across diverse skin tones
on Milk10K. For a comprehensive analysis of GPT-5.2 and its performance, we refer readers to our recent publication:

    Frederickson, KL; Adunyah, SE; Wang, Q 
    Evaluation of GPT-5.2 for Melanoma Detection Across Skin Tones. 
    Frontiers in Medicine - Dermatology, 2026. 13:1816102.  
    https://doi.org/10.3389/fmed.2026.1816102
