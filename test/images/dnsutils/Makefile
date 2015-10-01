all:
	@echo "try 'make image' or 'make push'"

image:
	docker build -t beta.gcr.io/google_containers/dnsutils .

push:
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/dnsutils
