all: push

# To bump the Spark version, bump the version in base/Dockerfile, bump
# this tag and reset to v1. You should also double check the native
# Hadoop libs at that point (we grab the 2.6.1 libs, which are
# appropriate for 1.5.1-with-2.6).
TAG = 1.5.1_v1

containers:
	docker build -t gcr.io/google_containers/spark-base base
	docker tag gcr.io/google_containers/spark-base gcr.io/google_containers/spark-base:$(TAG)
	docker build -t gcr.io/google_containers/spark-worker worker
	docker tag gcr.io/google_containers/spark-worker gcr.io/google_containers/spark-worker:$(TAG)
	docker build -t gcr.io/google_containers/spark-master master
	docker tag gcr.io/google_containers/spark-master gcr.io/google_containers/spark-master:$(TAG)
	docker build -t gcr.io/google_containers/spark-driver driver
	docker tag gcr.io/google_containers/spark-driver gcr.io/google_containers/spark-driver:$(TAG)

push: containers
	gcloud docker push gcr.io/google_containers/spark-base
	gcloud docker push gcr.io/google_containers/spark-base:$(TAG)
	gcloud docker push gcr.io/google_containers/spark-worker
	gcloud docker push gcr.io/google_containers/spark-worker:$(TAG)
	gcloud docker push gcr.io/google_containers/spark-master
	gcloud docker push gcr.io/google_containers/spark-master:$(TAG)
	gcloud docker push gcr.io/google_containers/spark-driver
	gcloud docker push gcr.io/google_containers/spark-driver:$(TAG)

clean:
