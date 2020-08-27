import subprocess, argparse
from config import voc_classes


train_template = "python train_AutoEncoders.py --cls_id {:d} --version 0.0"

test_templates = [
	"python test_AutoEncoders.py --cls_id {:d} --version 0.0 --set_name 'test_benign' --resume_epoch 4",
	"python test_AutoEncoders.py --cls_id {:d} --version 0.0 --set_name 'test_digi_ifgsm_hiding' --resume_epoch 4",
	"python test_AutoEncoders.py --cls_id {:d} --version 0.0 --set_name 'test_digi_ifgsm_miscls' --resume_epoch 4",
	"python test_AutoEncoders.py --cls_id {:d} --version 0.0 --set_name 'test_digi_ifgsm_appear' --resume_epoch 4",
	"python test_AutoEncoders.py --cls_id {:d} --version 0.0 --set_name 'test_physical_hiding' --resume_epoch 4",
	"python test_AutoEncoders.py --cls_id {:d} --version 0.0 --set_name 'test_physical_miscls' --resume_epoch 4",
	"python test_AutoEncoders.py --cls_id {:d} --version 0.0 --set_name 'test_physical_appear' --resume_epoch 4"
	]


def test_all_recon_error():
	for cls_id in range(len(voc_classes)):
		for template in test_templates:
			print(template.format(cls_id))
			subprocess.call(template.format(cls_id), shell=True)


def train_all():
	for cls_id in range(len(voc_classes)):
		template = train_template.format(cls_id)
		print(template.format(cls_id))
		subprocess.call(template.format(cls_id), shell=True)


def parse_inputs(args=None):
	parser = argparse.ArgumentParser(description='Training or testing.')
	parser.add_argument('--mode', choices=['train','test'])
	return parser.parse_args(args)


if __name__ == '__main__':
	parser = parse_inputs()
	if parser.mode == 'train':
		train_all()
	else:
		test_all_recon_error()

