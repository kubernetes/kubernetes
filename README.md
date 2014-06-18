# Kubernetes
Kubernetes is an open source implementation of container cluster management.

[Kubernetes Design Document](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/DESIGN.md)

[![GoDoc](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes?status.png)](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes)

## Kubernetes can run anywhere!
However, initial development was done on GCE and so our instructions and scripts are built around that.  If you make it work on other infrastructure please let us know and contribute instructions/code.

### Contents
* [Getting started on Google Compute Engine](#getting-started-on-google-compute-engine)
* [Running a local cluster](#running-locally)
* [Discussion and Community Support](#community-discussion-and-support)
* [Hacking on Kubernetes](#development)

## Getting started on Google Compute Engine

### Prerequisites

1. You need a Google Cloud Platform account with billing enabled. Visit
   [http://cloud.google.com/console](http://cloud.google.com/console) for more details.
2. Make sure you can start up a GCE VM.  At least make sure you can do the [Create an instance](https://developers.google.com/compute/docs/quickstart#addvm) part of the GCE Quickstart.
3. You must have Go (version 1.2 or later) installed: [www.golang.org](http://www.golang.org).
4. Ensure that your `gcloud` components are up-to-date by running `gcloud components update`.
5. Get the Kubernetes source:

        git clone https://github.com/GoogleCloudPlatform/kubernetes.git

6. Setting up a cluster requires the `htpasswd` tool in order to hash a randomly generated password for accessing the API server.  This is already installed on recent version of Mac OS X but on Linux you need to install it yourself.  On Debian/Ubuntu you can do this with:

        sudo apt-get update
        sudo apt-get install apache2-utils

### Setup

The setup script builds Kubernetes, then creates Google Compute Engine instances, firewall rules, and routes:

```
cd kubernetes
hack/dev-build-and-up.sh
```

### Running a container (simple version)

Once you have your instances up and running, the `build-go.sh` script sets up
your Go workspace and builds the Go components.

The `cloudcfg.sh` script spins up two containers, running [Nginx](http://nginx.org/en/) and with port 80 mapped to 8080:

```
cd kubernetes
hack/build-go.sh
cluster/cloudcfg.sh -p 8080:80 run dockerfile/nginx 2 myNginx
```

To stop the containers:
```
cluster/cloudcfg.sh stop myNginx
```

To delete the containers:
```
cluster/cloudcfg.sh rm myNginx
```

### Running a container (more complete version)


Assuming you've run `hack/dev-build-and-up.sh` and `hack/build-go.sh`:


```
cd kubernetes
cluster/cloudcfg.sh -c api/examples/pod.json create /pods
```

Where pod.json contains something like:

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

Look in `api/examples/` for more examples

### Tearing down the cluster
```
cd kubernetes
cluster/kube-down.sh
```

## Running locally
In a separate tab of your terminal, run:

```
cd kubernetes
hack/local-up.sh
```

This will build and start a lightweight local cluster, consisting of a master and a single minion. Type Control-C to shut it down. While it's running, you can use `hack/localcfg.sh` in place of `cluster/cloudcfg.sh` to talk to it.

## Where to go next?
[Detailed example application](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/guestbook/guestbook.md)

Or fork and start hacking!

## Community, discussion and support

If you have questions or want to start contributing please reach out.  We don't bite!

The Kubernetes team is hanging out on IRC on the [#google-containers room on freenode.net](http://webchat.freenode.net/?channels=google-containers).  We also have the [google-containers Google Groups mailing list](https://groups.google.com/forum/#!forum/google-containers).

If you are a company and are looking for a more formal engagement with Google around Kubernetes and containers at Google as a whole, please fill out [this form](https://docs.google.com/a/google.com/forms/d/1_RfwC8LZU4CKe4vKq32x5xpEJI5QZ-j0ShGmZVv9cm4/viewform). and we'll be in touch.

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
hack/test-go.sh
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
hack/integration-test.sh
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
```
cd kubernetes/api
sudo docker build -t kubernetes/raml2html .
sudo docker run --name="docgen" kubernetes/raml2html
sudo docker cp docgen:/data/kubernetes.html .
```

