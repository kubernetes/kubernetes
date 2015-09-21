.PHONY:	build push

TAG = latest

build:	
	sudo docker build -t kubernetes/fluentd-elasticsearch:$(TAG) .

push:	
	sudo docker push kubernetes/fluentd-elasticsearch:$(TAG)
