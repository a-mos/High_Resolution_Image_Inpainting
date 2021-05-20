if [ ! -d "gimpenv" ]; then

	echo "\n-----------Installing GIMP-ML-----------\n"
	
	if [ "$(uname)" == "Linux" ]; then
		if [[ $(lsb_release -rs) == "18.04" ]]; then #for ubuntu 18.04
			sudo apt install python-minimal
			alias python='python2'
		       
		elif [[ $(lsb_release -rs) == "20.04" ]]; then #for ubuntu 20.04
			sudo apt install python2-minimal
			wget https://bootstrap.pypa.io/get-pip.py 
			alias python='python2'
			python get-pip.py	

		elif [[ $(lsb_release -rs) == "10" ]]; then #for debian 10
			wget https://bootstrap.pypa.io/get-pip.py 
			python get-pip.py
		fi
	fi

	python -m pip install --user "virtualenv<=20.0.20"
	python -m virtualenv gimpenv
	source gimpenv/bin/activate
	python -m pip install torchvision
	python -m pip install "opencv-python<=4.3"
	python -m pip install numpy
	python -m pip install future
	python -m pip install torch
	python -m pip install scipy
	deactivate

	echo "\n-----------Installed GIMP-ML------------\n"

else

	echo "Environment already setup!"

fi
