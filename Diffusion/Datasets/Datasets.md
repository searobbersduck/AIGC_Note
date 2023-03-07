
## TODO

- [ ] [Semantic search with embeddings: index anything](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c)
- [ ] [clip-retrieval](https://github.com/rom1504/clip-retrieval)
- [ ] [watermark-detection](https://github.com/LAION-AI/watermark-detection)
- [ ] [LAION-5B-WatermarkDetection](https://github.com/LAION-AI/LAION-5B-WatermarkDetection/releases/tag/1.0)

## Public Datasets

### LAION

#### [OPEN LARGE-SCALE MULTI-MODAL DATASETS](https://laion.ai/blog/laion-5b/)
  *  a dataset of 5.85 billion CLIP-filtered image-text pairs, 14x bigger than LAION-400M
     *  2.3B contain English language
     *  2.2B samples from 100+ other languages
     *  1B samples have texts that do not allow a certain language assignment (e.g. names )
     *  LAION 5B, a CLIP-filtered dataset of 5,85 billion high-quality image-text pairs, their CLIP ViT-L/14 embeddings, kNN-indices, a web interface for exploration & subset-creation and NSFW- and watermark-detection scores and tools.
  *  Keep in mind that the uncurated nature of the dataset means that collected links may lead to strongly discomforting and disturbing content for a human viewer. 
     *  It is possible to extract a “safe” subset by filtering out samples based on the safety tags (using a customized trained NSFW classifier that we built). 
  *  **<font color='red'>Download the data</font>** 
     *  [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en) 2.32 billion of these contain texts in the English language
     *  [laion2B-multi](https://huggingface.co/datasets/laion/laion2B-multi) 2.26 billion contain texts from 100+ other languages
     *  [laion1B-nolang](https://huggingface.co/datasets/laion/laion1B-nolang) 1.27 billion have texts where a particular language couldn’t be clearly detected.
     *  The data can comfortably be downloaded with [img2dataset](https://github.com/rom1504/img2dataset) (240TB in 384, 80TB in 224)
  *  **<font color='red'>Acquisition pipeline</font>** 
     *  Distributed processing of Common Crawl
        *  language detection on text with three possible outputs: English language with confidence, another language with confidence, no language which contains “no detection” and “detection under the confidence threshold”. The “no language” set often contains short texts, mostly with names of people and places. 
     *  Distributed downloading of the images
     *  CLIP inference at the post-processing stage
        *  The data pipeline continued with GPU nodes doing inference on the collected image-text pairs, and calculating the similarity of the embeddings for the image and the text. After the similarity score was established we removed the pairs under the threshold we decided to use, i.e 0.28 for the English dataset ( with CLIP ViT B/32 ) and 0.26 for the rest (with mCLIP). As an estimation, we removed about 90% of the samples, trimming the 50+ billion of candidates to just below 6 billion.
     *  Filtering out unsuitable image-text pairs
  * **<font color='red'>Dataset preparation pipeline</font>**
    * Pipeline
      1. Downloading the data as webdataset with distributed img2datase
      2. Computing Vit-L/14 embeddings with distributed clip-inference
      3. Computing a KNN index from these embeddings using autofaiss
      4. Computing additional tags (NSFW and watermark) using clip embeddings
    * Distributed img2dataset
      * For LAION-5B we introduced a distributed mode for this tool, allowing to downloading the 5,85B samples in a week using 10 nodes.
    * Distributed clip inference
      *  clip retrieval
      *  ViT-L/14 embeddings
      *  a week using 32 A100
      *  a speed of 312 sample/s per GPU, compared to 1800 sample/s for ViT-B/32
    * Distributed indexing
      * We then used these 9 TB of image embeddings to build a large PQ128 knn index using the [autofaiss](https://github.com/criteo/autofaiss) tool. To make this run faster, a [distributed mode](https://github.com/criteo/autofaiss/blob/master/docs/distributed/distributed_autofaiss.md) is available.
    * Integration in the search UI
    * Watermark and safety inference
    * Watermarks
      * The training dataset is 90000 samples (45222 watermarks, 44778 clear).
      * The majority of the watermarked images have been extracted from the LAION-400M KNN index through the use of several text prompts like “clip art watermark”, “cat watermark” or “landscape watermark”.
      * The creation of high-quality, openly accessible watermark detection test sets with clear and plausible definitions of what should be considered a watermark and what not, remains a challenge for future projects. 
      * Nevertheless we are convinced that removing images with a high confidence score for containing a watermark based on our model will significantly reduce the percentage of images that would be considered as obvious watermarks.
    * Safety
  * **<font color='red'>Using LAION datasets</font>**
    * CLIP
      * [open_clip](https://github.com/mlfoundations/open_clip)
    * BLIP inference tuning
    * GLIDE
    * Semantic search and subset extraction
    * CLOOB

## 常见术语

NSFW: Not Safe/Suitable For Work