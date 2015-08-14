all:
	@echo "try 'make image' or 'make push'"

image:
	docker build -t gcr.io/google_containers/dnsutils .

push:
	gcloud docker push gcr.io/google_containers/dnsutils
