## A set of scripts to install OpenCV 3 with contrib (non-free) modules

* download-opencv.sh

	Downloads OpenCV 3 and the non-free modules, and unzips them to the current directory using **wget** and **tar**.

* cmake-opencv.sh

	Runs **cmake** to enable CUDA, contrib modules and maximize compatibility with other components.

* build-opencv.sh

	Runs **make** and **make install** to install OpenCV to the install directory specified in the previous file.

* delete-opencv.sh

	Deletes the entire CV directory.


