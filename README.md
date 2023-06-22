# MasterArbeit Trainer Contact Grasp Model Fusion / Contact Grasp Model

# How to run the program

Run in terminal following command:

```python
PYTHON .\run_training.py -c {config_file} -n {network} 
```

## Arguments
- Configuration File: There are two json files in the folder configuration, which contains all the important parameters for the training of the network to predict contact grasp).

```python
-c .\config_files\config_contact.json
```
- Network: Network to train 

    - 0: contact grasp fusion.
    - 1: contact grasp.

```python
-n 1
```

## Example 

```python
PYTHON run_training.py -c config/config_contact.json -n 1
```
