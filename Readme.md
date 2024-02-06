# ManiNeg

## Abstract

Breast cancer poses a significant health threat worldwide.
Contrastive learning has emerged as an effective method to extract critical lesion features from mammograms, thereby
offering a potent tool for breast cancer screening and analysis.
A crucial aspect of contrastive learning involves negative sampling,
where the selection of appropriate hard negative samples is essential for driving representations to retain more
detailed information about lesions.
In large-scale contrastive learning applied to natural images, it is often assumed that extracted features can
sufficiently capture semantic content,
and that each mini-batch inherently includes ideal hard negative samples.
However, the unique characteristics of breast lumps challenge these assumptions when dealing with mammographic data.
In response, we introduce ManiNeg, a novel approach that leverages manifestations as proxies to select hard negative
samples.
Manifestations, which refer to the observable symptoms or signs of a disease, provide a robust basis for choosing hard
negative samples.
This approach benefits from its invariance to model optimization, facilitating efficient sampling.
We tested ManiNeg on the task of distinguishing between benign and malignant breast lumps.
Our results demonstrate that ManiNeg not only improves representation in both unimodal and multimodal contexts but also
offers benefits that extend to datasets beyond the initial pretraining phase.
To support ManiNeg and future research endeavors, we have developed the MVKL mammographic dataset.
This dataset includes multi-view mammograms, corresponding reports, meticulously annotated manifestations,
and pathologically confirmed benign-malignant outcomes for each case.
The MVKL dataset and our codes are publicly available to foster further research within the community.

## Code and Dataset

The repository currently contains unarranged code of ManiNeg. Detailed instruction will be provided after
paper publication.
Meanwhile, our MVKL dataset will be publicly available after paper publication. Please stay tuned. 