#! /bin/bash
./build
sudo cp ./bin/flanneld /usr/bin/flanneld
sudo chmod -c 777 /usr/bin/flanneld
