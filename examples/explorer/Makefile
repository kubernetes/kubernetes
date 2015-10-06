all: push

# Keep this one version ahead, so no one accidentally blows away the latest published version.
TAG = 1.1

explorer: explorer.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./explorer.go

container: explorer
	docker build -t beta.gcr.io/google_containers/explorer:$(TAG) .

push: container
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/explorer:$(TAG)

clean:
	rm -f explorer
