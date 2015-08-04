all: serve_hostname

TAG = v1.2
PREFIX = gcr.io/google_containers
TEST_PREFIX = gcr.io/_b_k8s_authenticated_test

serve_hostname: serve_hostname.go
	CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' ./serve_hostname.go

container: serve_hostname
	docker build -t $(PREFIX)/serve_hostname:$(TAG) .
	if [ -n "$(TEST_PREFIX)" ]; then \
	    docker tag -f $(PREFIX)/serve_hostname:$(TAG) $(TEST_PREFIX)/serve_hostname:$(TAG); \
	fi

push:
	gcloud docker push $(PREFIX)/serve_hostname:$(TAG)
	if [ -n "$(TEST_PREFIX)" ]; then \
	    gcloud docker push $(TEST_PREFIX)/serve_hostname:$(TAG); \
	fi

clean:
	rm -f serve_hostname
