all:

TAG = 1.0
PREFIX = bprashanth/nginxhttps
KEY = /tmp/nginx.key
CERT = /tmp/nginx.crt
SECRET = /tmp/secret.json

keys:
	# The CName used here is specific to the service specified in nginx-app.yaml.
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout $(KEY) -out $(CERT) -subj "/CN=nginxsvc/O=nginxsvc"

secret:
	CGO_ENABLED=0 GOOS=linux go run -a -installsuffix cgo -ldflags '-w' make_secret.go -crt $(CERT) -key $(KEY) > $(SECRET)

container:
	docker build -t $(PREFIX):$(TAG) .

push: container
	docker push $(PREFIX):$(TAG)

clean:
	rm $(KEY)
	rm $(CERT)
