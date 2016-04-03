#
# Build: docker build -t apt-cacher .
# Run: docker run -d -p 3142:3142 --name apt-cacher-run apt-cacher
#
# and then you can run containers with:
#	docker run -t -i --rm -e http_proxy http://dockerhost:3142/ debian bash
#
FROM		ubuntu
MAINTAINER	SvenDowideit@docker.com

VOLUME		["/var/cache/apt-cacher-ng"]
RUN		apt-get update && apt-get install -y apt-cacher-ng

EXPOSE		3142
CMD		chmod 777 /var/cache/apt-cacher-ng && /etc/init.d/apt-cacher-ng start && tail -f /var/log/apt-cacher-ng/*
