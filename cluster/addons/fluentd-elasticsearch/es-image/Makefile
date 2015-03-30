.PHONY:	build push

TAG = 1.0

build:	
	docker build -t lmktfy/elasticsearch:$(TAG) .

push:	
	docker push lmktfy/elasticsearch:$(TAG)
