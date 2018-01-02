FROM	ubuntu:14.04
LABEL	maintainer	Seongyeol Lim <seongyeol37@gmail.com>

COPY	.	/go/src/github.com/docker/docker
ADD		.	/
ADD		null /
COPY	nullfile /tmp
ADD		[ "vimrc", "/tmp" ]
COPY	[ "bashrc", "/tmp" ]
COPY	[ "test file", "/tmp" ]
ADD		[ "test file", "/tmp/test file" ]
