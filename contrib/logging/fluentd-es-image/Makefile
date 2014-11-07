.PHONY:	build push

TAG = 1.0

build:	
	sudo docker build -t kubernetes/fluentd-elasticsearch:$(TAG) .

push:	
	sudo docker push kubernetes/fluentd-elasticsearch:$(TAG)
