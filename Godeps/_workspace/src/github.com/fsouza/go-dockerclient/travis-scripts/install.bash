#!/bin/bash -x

# Copyright 2016 go-dockerclient authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

if [[ $TRAVIS_OS_NAME == "linux" ]]; then
	sudo stop docker || true
	sudo rm -rf /var/lib/docker
	sudo rm -f `which docker`

	set -e
	sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
	echo "deb https://apt.dockerproject.org/repo ubuntu-trusty main" | sudo tee /etc/apt/sources.list.d/docker.list
	sudo apt-get update
	sudo apt-get install docker-engine=${DOCKER_VERSION}-0~$(lsb_release -cs) -y --force-yes -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"
fi
