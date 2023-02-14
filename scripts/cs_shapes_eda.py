"""Script to perform exploratory data analysis on the CS shapes dataset.
Creates a csv file with the results of the analysis.
"""
from tqdm import tqdm
from data.cs_image_loader import CS_Images
from data.mesh_properties import MeshAnalyzer
from data.segmentation_properties import SegmAnalyzer
import pandas as pd

if __name__ == '__main__':
    dataset = CS_Images(segmentation='brainvisa',
                        mesh=True,
                        preload=False)
    mesh_analyzer = MeshAnalyzer()
    seg_analyzer = SegmAnalyzer()
    analysis_results = []
    for i in tqdm(range(len(dataset))):
        img = dataset[i]
        caseid = img['caseid']
        centre = img['centre']
        
        bvisa_segm = img['bvisa']
        
        bvisa_mesh_lscl = img['bvisa_mesh_lscl']
        bvisa_mesh_rscl = img['bvisa_mesh_rscl']
        
        corrected_mesh_lscl = img['corrected_mesh_lscl']
        corrected_mesh_rscl = img['corrected_mesh_rscl']
        
        case_results = {'caseid': caseid,
                        'centre': centre}
        
        if bvisa_mesh_lscl is not None:
            res = mesh_analyzer.process(bvisa_mesh_lscl)
            case_results = case_results | {f'bvisaL_{k}': v for k, v in res.items()}
        if bvisa_mesh_rscl is not None:
            res = mesh_analyzer.process(bvisa_mesh_rscl)
            case_results = case_results | {f'bvisaR_{k}': v for k, v in res.items()}
        
        if corrected_mesh_lscl is not None:
            res = mesh_analyzer.process(corrected_mesh_lscl)
            case_results = case_results | {f'correcL_{k}': v for k, v in res.items()}
        
        if corrected_mesh_rscl is not None:
            res = mesh_analyzer.process(corrected_mesh_rscl)
            case_results = case_results | {f'correcR_{k}': v for k, v in res.items()}
        
        segm_properties = seg_analyzer.process(bvisa_segm, caseid)
        case_results = case_results | segm_properties
        
        analysis_results.append(case_results)
    
    res_df = pd.DataFrame(analysis_results)
    res_df.to_csv('results/CS_shape_analysis_bvisa.csv', index=False)
    res_df.to_pickle('results/CS_shape_analysis_bvisa.pkl')