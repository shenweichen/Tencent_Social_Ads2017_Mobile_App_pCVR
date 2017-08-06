import subprocess,multiprocessing
import os,time
import pandas as pd
import numpy as np

class FFM:
    """libffm-1.21 Python Wrapper with libffm format data

    :Args:
        - reg_lambda: float, default: 2e-5
            regularization parameter
        - factor: int,default: 4
            number of latent factors
        - iteration: int, default: 15
        - learning_rate: float, default: 0.2 
        - n_jobs: int, default: 1
            Number of parallel threads
        - verbose: int, default: 1
        - norm: bool, default: True
            instance-wise normalization
    """
    def __init__(self,reg_lambda=0.00002,factor=4,iteration=15,learning_rate=0.2,n_jobs=1,
                verbose=1,norm=True,):
        if n_jobs <=0 or n_jobs > multiprocessing.cpu_count():
            raise ValueError('n_jobs must be 1~{0}'.format(multiprocessing.cpu_count()))

        self.reg_lambda = reg_lambda
        self.factor = factor
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.norm = norm
 

        self.cmd = ''

        self.output_name = 'ffm_result'+str(int(time.time()))# temp predict result file
       
        
        
   
    def fit(self,train_ffm_path,valid_ffm_path=None,model_path=None,auto_stop=False,):
        """ Train the FFM model with ffm-format data,

        :Args:
            - train_ffm_path: str
            - valid_ffm_path: str, default: None
            - model_path: str, default: None
            - auto_stop: bool, default: False 
                stop at the iteration that achieves the best validation loss
        """
        
        if not os.path.exists(train_ffm_path):
            raise FileNotFoundError("file '{0}' not exists".format(train_ffm_path))
        self.train_ffm_path = train_ffm_path
        self.valid_ffm_path = valid_ffm_path
        self.model_path = None
        self.auto_stop = auto_stop


        cmd = 'ffm-train -l {l} -k {k} -t {t} -r {r} -s {s}'\
        .format(l=self.reg_lambda,k=self.factor,t=self.iteration,r=self.learning_rate,s=self.n_jobs)
        if self.valid_ffm_path is not None:
            cmd +=' -p {p}'.format(p=self.valid_ffm_path)
            
        if self.verbose == 0:
            cmd += ' --quiet'
        if not self.norm:
            cmd += ' --no-norm'

        if self.auto_stop:
            if self.valid_ffm_path is None:
                raise ValueError('Must specify valid_ffm_path when auto_stop = True')
            cmd += ' --auto-stop'
        cmd += ' {p}'.format(p=self.train_ffm_path)
        if not model_path is  None:
            cmd +=' {p}'.format(p=model_path)
            self.model_path = model_path
        self.cmd = cmd
        print('Sending command...')
        popen = subprocess.Popen(cmd, stdout = subprocess.PIPE,shell=True)
        while True:
            output = str(popen.stdout.readline(),encoding='utf-8').strip('\n')
            if output.strip()=='':
                print('FFM training done')
                break
            print(output)

    def predict(self,test_ffm_path,model_path=None):
        """ Predict and return the probability of positive class.

        :Args:
            - test_ffm_path: str
            - model_path: str, default: None       
        :returns: 
            - pred_prob: np.array
        """

        cmd = "ffm-predict {t}".format(t=test_ffm_path)
        if model_path is None and self.model_path is None:
            raise ValueError('Must specify model_path')
        elif model_path is not None:
            self.model_path = model_path
        cmd +=" {0} {1}".format(self.model_path,self.output_name)
        self.cmd = cmd
        print('Sending command...')
        popen = subprocess.Popen(cmd, stdout = subprocess.PIPE,shell=True)
        while True:
            output = str(popen.stdout.readline(),encoding='utf-8').strip('\n')
            if output.strip()=='':
                print('FFM predicting done')
                break
            print(output)
    
        ans = pd.read_csv(self.output_name,names=['prob'])
        os.remove(self.output_name)
        return ans.prob.values