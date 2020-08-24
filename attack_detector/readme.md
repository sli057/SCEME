# generate perturbations

1. digital miscategorization attack
```
python digital_attack.py --attack_type 'miscls'
```
2. digital hiding attack
```
python digital_attack.py --attack_type 'hiding'
```
3. digital appearing attack
```
python digital_attack.py --attack_type 'appear'
```
4. physical miscategorization attack
```
python physical_attack.py --attack_type 'miscls'
```
5. physical hiding attack
```
python physical_attack.py --attack_type 'hiding'
```
6. physical appearing attack
```
python physical_attack.py --attack_type 'appear'
```

# collect the generated perturbations
```
cd script_extract_files
python extract_attack.py
```
# visulize the generated perturbations & according detection results inside visualization folder

# the physical attacks are re-implemented from https://github.com/evtimovi/robust_physical_perturbations.git
