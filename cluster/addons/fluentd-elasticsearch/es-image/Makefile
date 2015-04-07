.PHONY:	build push

TAG = 1.0

build:	
	docker build -t kubernetes/elasticsearch:$(TAG) .

push:	
	docker push kubernetes/elasticsearch:$(TAG)
