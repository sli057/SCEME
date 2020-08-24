import os
import glob, os


def extract_list(root_path, file_name, background=False):
	file_list = glob.glob(os.path.join(root_path,'*.p'))
	with open(file_name, 'w') as f:
		for filename in file_list:
			f.write(filename+'\n')
	print("{:d} entries in {:s}".format(len(file_list),file_name))
	return 

if __name__ == "__main__":
	
	root_path = '../IFGSM_p_miscls'
	file_name = 'digital_miscls.txt'
	extract_list(root_path, file_name)

	root_path = '../IFGSM_p_hiding'
	file_name = 'digital_hiding.txt'
	extract_list(root_path, file_name)

	root_path = '../IFGSM_p_appear'
	file_name = 'digital_appaer.txt'
	extract_list(root_path, file_name)

	root_path = '../Physical_p_miscls'
	file_name = 'physical_miscls.txt'
	extract_list(root_path, file_name)

	root_path = '../Physical_p_hiding'
	file_name = 'physical_hiding.txt'
	extract_list(root_path, file_name)
	
	root_path = '../Physical_p_appear'
	file_name = 'physical_appear.txt'
	extract_list(root_path, file_name)
	
	
