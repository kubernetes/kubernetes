.PHONY:	build push

TAG = 1.1

build:	
	docker build -t kubernetes/kibana:$(TAG) .

push:	
	docker push kubernetes/kibana:$(TAG)
