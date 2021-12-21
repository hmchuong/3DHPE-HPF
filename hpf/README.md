<div align="center">
<h1>Domain adversarial STS-GCN</h1>
</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

</div>

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


<div align="center"> <h2> Domain Adverserial STS-GCN </h2>  </div>
<div align="justify">

</div>

 ### Train
The arguments for running the code are defined in [parser.py](utils/parser.py). We have used the following commands for training the network,on different datasets and body pose representations(3D and euler angles):
 
```bash
 python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 
 ```
```bash
 python main_h36_ang.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 16 
  ```
 
 ### Test
 To test on the pretrained model, we have used the following commands:
 ```bash
 python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 --mode test --model_path ./checkpoints/CKPT_3D_H36M
  ```


### Visualization
 For visualizing from a pretrained model, we have used the following commands:
 ```bash
  python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 --mode viz --model_path ./checkpoints/CKPT_3D_H36M --n_viz 5
 ```



