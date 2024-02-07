def main():
    # Importing the modules
    from modnet.hyper_opt import FitGenetic
    from pathlib import Path, PosixPath
    
    import modwrap as mdw
    import os
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    #n_jobs = int(0.7*int(os.environ["SLURM_CPUS_PER_TASK"]))
    n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])
    
    # Trick of PP to prevent explosion of the threads
    def setup_threading():
    	import os
    	os.environ['OPENBLAS_NUM_THREADS'] = '1'
    	os.environ['MKL_NUM_THREADS'] = '1'
    	os.environ["OMP_NUM_THREADS"] = "1"
    	os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    	os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    setup_threading()
    
    
    path_re2f = Path('/globalscratch/ucl/modl/vtrinque/NLO/HT/ref_idx/re2fractive')
    
    # Load the featurized MODData
    md_featselec = (path_re2f / 'humanguided' / 'v0' / 'mod.data_refeatselec_v0_v2')
    
    model_params={
        'size_pop':20, # dflt 20
        'num_generations':10, # dflt 10
        'nested':5, # dflt = 5
        'n_jobs':n_jobs,
        'early_stopping':4, # dflt 4
        'refit':0, # dflt = 5
        'fast':False,
        }
    
    mdw.actilearn(
        structures=None,
        ids=None,
        X=None,
        Y=None,
        md_feat=None,
        md_featselec=md_featselec,
        start_frac=None,
        start_n=200,
        start_set=None,
        start_state=42,
        ncycles=38,
        accuracy=None,
        accuracy_type=None,
        end_set=None,
        model_type=FitGenetic,
        model_params=model_params,
        cv_k=5,
        cv_state=42,
        acquisition=None,
        acquisition_kwargs=None,
        acquisition_n=100,
        acquisition_frac=None,
        featurize_cycle=None,
        featurize_cv=None,
        featselec_cycle=None,
        featselec_cv=None,
    )


if __name__ == "__main__":
	main()
