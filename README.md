# Kubernetes

Kubernetes is an open source reference implementation of container cluster management.

## Getting started on Google Compute Engine

### Prerequisites

1. You need a Google Cloud Platform account with billing enabled.  Visit http://cloud.google.com/console for more details
2. You must have Go installed: [www.golang.org](http://www.golang.org)
3. Ensure that your `gcloud` components are up-to-date by running `gcloud components update`.
4. Get the Kubernetes source:  `git clone https://github.com/GoogleCloudPlatform/kubernetes.git`

### Setup
```
cd kubernetes
./src/scripts/dev-build-and-up.sh
```

### Running a container (simple version)
```
cd kubernetes
./src/scripts/build-go.sh
./src/scripts/cloudcfg.sh -p 8080:80 run dockerfile/nginx 2 myNginx
```

This will spin up two containers running Nginx mapping port 80 to 8080.

To stop the container:
```
./src/scripts/cloudcfg.sh stop myNginx
```

To delete the container:
```
./src/scripts/cloudcfg.sh rm myNginx
```

### Running a container (more complete version)
```
cd kubernetes
./src/scripts/cloudcfg.sh -c examples/task.json create /tasks
```

Where task.json contains something like:
```
{
  "ID": "nginx",
  "desiredState": {
    "image": "dockerfile/nginx",
    "networkPorts": [{
      "containerPort": 80,
      "hostPort": 8080
    }]
  },
  "labels": {
    "name": "foo"
  }
}
```

Look in the ```examples/``` for more examples

### Tearing down the cluster
```
cd kubernetes
./src/scripts/kube-down.sh
```

## Development

### Hooks
```
# Before committing any changes, please link/copy these hooks into your .git
# directory. This will keep you from accidentally committing non-gofmt'd
# go code.
cd kubernetes
ln -s "../../hooks/prepare-commit-msg" .git/hooks/prepare-commit-msg
ln -s "../../hooks/commit-msg" .git/hooks/commit-msg
```

### Unit tests
```
cd kubernetes
./src/scripts/test-go.sh
```

### Coverage
```
cd kubernetes
go tool cover -html=target/c.out
```

### Integration tests
```
# You need an etcd somewhere in your path.
# To get from head:
go get github.com/coreos/etcd
go install github.com/coreos/etcd
sudo ln -s "$GOPATH/bin/etcd" /usr/bin/etcd
# Or just use the packaged one:
sudo ln -s "$REPO_ROOT/target/bin/etcd" /usr/bin/etcd
```

```
cd kubernetes
./src/scripts/integration-test.sh
```

### Keeping your development fork in sync
One time after cloning your forked repo:
```
git remote add upstream https://github.com/GoogleCloudPlatform/kubernetes.git
```

Then each time you want to sync to upstream:
```
git fetch upstream
git rebase upstream/master
```

### Regenerating the documentation
Install [nodejs](http://nodejs.org/download/), [npm](https://www.npmjs.org/), and
[raml2html](https://github.com/kevinrenskers/raml2html), then run:
```
cd kubernetes/api
raml2html kubernetes.raml > kubernetes.html
```
