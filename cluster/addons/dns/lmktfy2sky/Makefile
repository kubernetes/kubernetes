all: lmktfy2sky

lmktfy2sky: lmktfy2sky.go
	CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' ./lmktfy2sky.go

container: lmktfy2sky
	sudo docker build -t lmktfy/lmktfy2sky .

push:
	sudo docker push lmktfy/lmktfy2sky

clean:
	rm -f lmktfy2sky
