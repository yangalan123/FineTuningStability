# Fine-Tuning Stability
Codebase for the EMNLP 2022 paper "Improving Stability of Fine-Tuning Pretrained Language Models via Component-Wise Gradient Norm Clipping" [[paper](https://arxiv.org/abs/2210.10325)]

Author: Chenghao Yang (yangalan1996@gmail.com) (University of Chicago)

Project Supervisor: Xuezhe Ma (xuezhema@isi.edu) (University of Southern California)

## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

```
@inproceedings{yang-2022-stability,
  author =  "Yang, Chenghao and Ma, Xuezhe",
  title =  "Improving Stability of Fine-Tuning Pretrained Language Models via Component-Wise Gradient Norm Clipping",
  booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
  year =  "2022"
}
```

## Dependency Installation
```bash
conda create -p ./env python=3.7
conda activate ./env # the environment position is optional, you can choose whatever places you like to save dependencies. Here I choose ./env for example.
pip install -r requirements.txt
cd ./CustomToolBox
pip install -e .
# you also need to download the dataset (e.g., anli) and model checkpoints, see below
```

## Dataset and Model Downloading
```bash
python download_model_and_data_HF.py --model_name_or_path bert-base-uncased --task_name sst2 --cache_dir ./cache
```
The supported models list can be found [here](https://huggingface.co/models). Here we use `bert-base-uncased` as an example. 

The supported GLUE tasks list is [here](https://huggingface.co/datasets/glue). (We use `sst2` as an example.)

## Run Experiments
```bash
bash run.sh
```
You can play with `run.sh` to experiment with different settings. 

Once you have run the experiments, you can compute the standard deviation of the performance and draw the visualization used in the paper by running: 
```bash
pip install plotly==5.10.0
python compute_variance.py 
```

## Detailed Experiment Results
We will upload the experiment results as well as baseline replication results soon. 
Currently doing some works to move the data from the institutional server. Stay tuned!

## Special Acknowledgement
Our work is based on the finding of Marius Mosbach's [ICLR 2021 paper](https://openreview.net/pdf?id=nzpLWnVAyah) on fine-tuning stability for pretrained language models. 

We would like to thank Marius Mosbach for his help on baseline replication. 

Also check out their wonderful docker-supported [codebase](https://github.com/uds-lsv/bert-stable-fine-tuning)!