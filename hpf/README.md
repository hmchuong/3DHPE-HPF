<div align="center">
<h1>Domain adversarial STS-GCN</h1>
</div>

The code has been built on top of [STS-GCN](https://github.com/FraLuca/STSGCN)

 ### Install dependencies:
```
 $ pip install -r requirements.txt
```

 ### Human 3.6M Dataset

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).
 
Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```

```
Put the all downloaded datasets in ../datasets directory.
```

 ### Train
The arguments for running the code are defined in [parser.py](utils/parser.py). The following is the command for training the network,on different datasets and 3D body pose representations:
 
```bash
 python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 
 ```

 ### Test
 To test on the pretrained model, the following command is used:
 ```bash
 python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 --mode test --model_path ./checkpoints/CKPT_3D_H36M
  ```


### Visualization
The command for visualizing the trained model is as below:
 ```bash
  python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 --mode viz --model_path ./checkpoints/CKPT_3D_H36M --n_viz 5
 ```



