# node-gen

## Description

`node-gen` is a command line tool to generate json representations of nodes

Manifests can then be edited by a human to match deployment needs or passed directly to kubectl create -f

## Usage
```
$ node-gen [--labels=KEY1=VAL1,KEY2=VAL2] NODE [NODES...]
```

### Options
- `labels`: labesl applied to all nodes in key=value format seperated by a comma

### Examples
```
$ node-gen machine1.example.com
$ node-gen --labels=rack=12,geo=Raleigh machine1.example.com machine2.example.com
```
