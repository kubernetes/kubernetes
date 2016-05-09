# this file describes how to build tsuru python image
# to run it:
# 1- install docker
# 2- run: $ docker build -t tsuru/python https://raw.github.com/tsuru/basebuilder/master/python/Dockerfile

from	base:ubuntu-quantal
run	apt-get install wget -y --force-yes
run	wget http://github.com/tsuru/basebuilder/tarball/master -O basebuilder.tar.gz --no-check-certificate
run	mkdir /var/lib/tsuru
run	tar -xvf basebuilder.tar.gz -C /var/lib/tsuru --strip 1
run	cp /var/lib/tsuru/python/deploy /var/lib/tsuru
run	cp /var/lib/tsuru/base/restart /var/lib/tsuru
run	cp /var/lib/tsuru/base/start /var/lib/tsuru
run	/var/lib/tsuru/base/install
run	/var/lib/tsuru/base/setup
