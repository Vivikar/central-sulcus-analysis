from tqdm import tqdm
from data.image import CS_Images
from data.mesh import MeshAnalyzer
import pandas as pd

if __name__ == '__main__':
    dataset = CS_Images(segmentation='brainvisa',
                        mesh=True,
                        preload=False)
    mesh_analyzer = MeshAnalyzer()
    analysis_results = []
    for i in tqdm(range(len(dataset))):
        img = dataset[i]
        caseid = img['caseid']
        centre = img['centre']
        
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

        analysis_results.append(case_results)
    
    res_df = pd.DataFrame(analysis_results)
    res_df.to_csv('results/CS_shape_analysis_bvisa.csv', index=False)