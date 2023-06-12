# SYNCS: Synthetic Data and Contrastive Self-Supervised Training for Central Sulcus Segmentation

*Master thesis project at Danish Research Centre for Magnetic Resonance (DRCMR) focusing on the development of a self-supervised learning method based on synthetic data for central sulcus segmentation in MRI brain scans.*
<p align="center"> Author: Vladyslav Zalevskyi. Supervisor: Krisoffer H. Madsen.

</p>
<p align="center"> See the full report <a href="https://github.com/Vivikar/central-sulcus-analysis/blob/a53ca666f475db1689074267182ca0b124e70fb5/SYNCS__Synthetic_Data_and_Contrastive_Self_Supervised_Training_for_Central_Sulcus_Segmentation.pdf">here</a>.

</p>

### **Abstract**
<blockquote>
Bipolar disorder (BD) and schizophrenia (SZ) are severe mental disorders that have a significant impact on individuals
and society. Early identification of risk markers for these diseases is crucial for understanding their progression and
enabling preventive interventions. The Danish High Risk and Resilience Study (VIA) is a longitudinal cohort study
that aims to gain insights into the early disease processes of SZ and BD, particularly in children with familial high risk
(FHR). Understanding structural brain changes associated with these diseases during early stages is essential for ef-
fective interventions. The central sulcus (CS) is a prominent brain landmark related to brain regions involved in motor
and sensory processing. Analyzing CS morphology can provide valuable insights into neurodevelopmental abnormal-
ities in the FHR group. However, CS segmentation presents challenges due to its high morphological variability and
complex shape, which are especially apparent in the adolescent cohort. This study explores two novel approaches for
training robust and adaptable CS segmentation models that address these challenges. Firstly, we utilize synthetic data
generation to model the morphological variability of the CS, adapting SynthSeg’s generative model to our problem.
Secondly, we employ self-supervised pre-training and multi-task learning to adjust the segmentation models to new
subject cohorts by learning relevant feature representations of the cortex shape. These approaches aim to overcome
limited data availability and enable reliable CS segmentation performance on diverse populations, removing the need
for extensive and error-prone post- and pre-processing steps. By leveraging synthetic data and self-supervised learn-
ing, this research demonstrates how recent advancements in training robust and generalizable deep learning models
can help overcome problems hindering the deployment of DL medical imaging solutions. Although our evaluation
showed only a moderate improvement in performance metrics, we emphasize the significant potential of the methods
explored to advance CS segmentation and their importance in facilitating early detection and intervention strategies
for SZ and BD.
</blockquote>

## Repository structure
```
./src - source code containing the implementation of the proposed methods
    /models - implementation of the proposed models
    /data - implementation of the data loading and preprocessing pipeline
    /utils - implementation of the utility functions
./scripts - scripts used for training and data generation
./notebooks - notebooks used for code development, data analysis and visualization
./config - configuration files for training and data generation
```

