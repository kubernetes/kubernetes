# git-blog-demo

This demo shows how to use the `git-sync` sidekick container along side `volumes` and `volumeMounts` to create a markdown powered blog.

## How it works

The pod is composed of 3 containers that share directories using 2 volumes:

- The `git-sync` container clones a git repo into the `markdown` volume
- The `hugo` container read from the `markdown` volume and render it into the `html` volume.
- The `nginx` container serve the content from the `html` volume.

## Usage

Build the demo containers, and push them to a registry

```
docker build -t <some-registry>/git-sync ..
docker build -t <some-registry>/hugo hugo/
docker push <some-registry>/hugo <some-registry>/git-sync
```

Create the pod and the service for the blog

```
kubectl pods create config/pod.html
kubectl services create config/pod.html
```

Open the service external ip in your browser
