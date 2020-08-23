1. digital miscategorization attack
```
python digital_attack.py --attack_type 'miscls'
```
# digital hiding attack
python digital_attack.py --attack_type 'hiding'
# digital appearing attack
python digital_attack.py --attack_type 'appear'
# physical miscategorization attack
python physical_attack.py --attack_type 'miscls'
# physical hiding attack
python physical_attack.py --attack_type 'hiding'
# physical appearing attack
python physical_attack.py --attack_type 'appear'

# collect the generated perturbations
cd script_extract_files
python extract_attack.py

# visulize the generated perturbations & according detection results inside visualization folder

# the physical attacks are re-implemented from https://github.com/evtimovi/robust_physical_perturbations.git
