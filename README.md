# Kubernetes
Kubernetes is an open source implementation of container cluster management.

[Kubernetes Design Document](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/DESIGN.md) - [Kubernetes @ Google I/O 2014](http://youtu.be/tsk0pWf4ipw)

[![GoDoc](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes?status.png)](https://godoc.org/github.com/GoogleCloudPlatform/kubernetes)
[![Travis](https://travis-ci.org/GoogleCloudPlatform/kubernetes.svg?branch=master)](https://travis-ci.org/GoogleCloudPlatform/kubernetes)


## Kubernetes can run anywhere!
However, initial development was done on GCE and so our instructions and scripts are built around that.  If you make it work on other infrastructure please let us know and contribute instructions/code.

## Kubernetes is in pre-production beta!
While the concepts and architecture in Kubernetes represent years of experience designing and building large scale cluster manager at Google, the Kubernetes project is still under heavy development.  Expect bugs, design and API changes as we bring it to a stable, production product over the coming year.

### Contents
* [Getting started on Google Compute Engine](#getting-started-on-google-compute-engine)
* [Getting started with a Vagrant cluster on your host](#getting-started-with-a-vagrant-cluster-on-your-host)
* [Running a local cluster on your host](#running-locally)
* [Running on CoreOS](#running-on-coreos)
* [Discussion and Community Support](#community-discussion-and-support)
* [Hacking on Kubernetes](#development)

## Getting started on Google Compute Engine

### Prerequisites

1. You need a Google Cloud Platform account with billing enabled. Visit
   [http://cloud.google.com/console](http://cloud.google.com/console) for more details.
2. Make sure you can start up a GCE VM.  At least make sure you can do the [Create an instance](https://developers.google.com/compute/docs/quickstart#addvm) part of the GCE Quickstart.
3. You need to have the Google Storage API, and the Google Storage JSON API enabled.
4. You must have Go (version 1.2 or later) installed: [www.golang.org](http://www.golang.org).
5. You must have the [`gcloud` components](https://developers.google.com/cloud/sdk/) installed.
6. Ensure that your `gcloud` components are up-to-date by running `gcloud components update`.
7. Get the Kubernetes source:

        git clone https://github.com/GoogleCloudPlatform/kubernetes.git

### Setup

The setup script builds Kubernetes, then creates Google Compute Engine instances, firewall rules, and routes:

```
cd kubernetes
hack/dev-build-and-up.sh
```

The script above relies on Google Storage to deploy the software to instances running in GCE. It uses the Google Storage APIs so the "Google Cloud Storage JSON API" setting must be enabled for the project in the Google Developers Console (https://cloud.google.com/console#/project).

The instances must also be able to connect to each other using their private IP. The script uses the "default" network which should have a firewall rule called "default-allow-internal" which allows traffic on any port on the private IPs.
If this rule is missing from the default network or if you change the network being used in `cluster/config-default.sh` create a new rule with the following field values:
* Source Ranges: 10.0.0.0/8
* Allowed Protocols or Port: tcp:1-65535;udp:1-65535;icmp

### Running a container (simple version)

Once you have your instances up and running, the `build-go.sh` script sets up
your Go workspace and builds the Go components.

The `kubecfg.sh` script spins up two containers, running [Nginx](http://nginx.org/en/) and with port 80 mapped to 8080:

```
cd kubernetes
hack/build-go.sh
cluster/kubecfg.sh -p 8080:80 run dockerfile/nginx 2 myNginx
```

To stop the containers:
```
cluster/kubecfg.sh stop myNginx
```

To delete the containers:
```
cluster/kubecfg.sh rm myNginx
```

### Running a container (more complete version)


Assuming you've run `hack/dev-build-and-up.sh` and `hack/build-go.sh`:


```
cd kubernetes
cluster/kubecfg.sh -c api/examples/pod.json create /pods
```

Where pod.json contains something like:

```
{
  "id": "php",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "php",
      "containers": [{
        "name": "nginx",
        "image": "dockerfile/nginx",
        "ports": [{
          "containerPort": 80,
          "hostPort": 8080
        }],
        "livenessProbe": {
          "enabled": true,
          "type": "http",
          "initialDelaySeconds": 30,
          "httpGet": {
            "path": "/index.html",
            "port": "8080"
          }
        }
      }]
    }
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

## Getting started with a Vagrant cluster on your host

### Prerequisites
1. Install latest version >= 1.6.2 of vagrant from http://www.vagrantup.com/downloads.html
2. Install latest version of Virtual Box from https://www.virtualbox.org/wiki/Downloads
3. Get the Kubernetes source:

```
git clone https://github.com/GoogleCloudPlatform/kubernetes.git
```

### Setup

By default, the Vagrant setup will create a single kubernetes-master and 3 kubernetes-minions.  You can control the number of minions that are instantiated via an environment variable on your host machine.  If you plan to work with replicas, we strongly encourage you to work with enough minions to satisfy your largest intended replica size.  If you do not plan to work with replicas, you can save some system resources by running with a single minion.

```
export KUBERNETES_NUM_MINIONS=3
```

To start your local cluster, open a terminal window and run:

```
cd kubernetes
vagrant up
```

Vagrant will provision each machine in the cluster with all the necessary components to build and run Kubernetes.  The initial setup can take a few minutes to complete on each machine.

By default, each VM in the cluster is running Fedora, and all of the Kubernetes services are installed into systemd.

To access the master or any minion:

```
vagrant ssh master
vagrant ssh minion-1
vagrant ssh minion-2
vagrant ssh minion-3
```

To view the service status and/or logs on the kubernetes-master:
```
vagrant ssh master
[vagrant@kubernetes-master ~] $ sudo systemctl status apiserver
[vagrant@kubernetes-master ~] $ sudo journalctl -r -u apiserver

[vagrant@kubernetes-master ~] $ sudo systemctl status controller-manager
[vagrant@kubernetes-master ~] $ sudo journalctl -r -u controller-manager

[vagrant@kubernetes-master ~] $ sudo systemctl status etcd
[vagrant@kubernetes-master ~] $ sudo systemctl status nginx
```

To view the services on any of the kubernetes-minion(s):
```
vagrant ssh minion-1
[vagrant@kubernetes-minion-1] $ sudo systemctl status docker
[vagrant@kubernetes-minion-1] $ sudo journalctl -r -u docker
[vagrant@kubernetes-minion-1] $ sudo systemctl status kubelet
[vagrant@kubernetes-minion-1] $ sudo journalctl -r -u kubelet
```

To push updates to new Kubernetes code after making source changes:
```
vagrant provision
```

To shutdown and then restart the cluster:
```
vagrant halt
vagrant up
```

To destroy the cluster:
```
vagrant destroy -f
```

You can also use the cluster/kube-*.sh scripts to interact with vagrant based providers just like any other hosting platform for kubernetes.

```
cd kubernetes
modify cluster/kube-env.sh:
  KUBERNETES_PROVIDER="vagrant"

cluster/kube-up.sh => brings up a vagrant cluster
cluster/kube-down.sh => destroys a vagrant cluster
cluster/kube-push.sh => updates a vagrant cluster
cluster/kubecfg.sh => interact with the cluster
```


### Running a container

Your cluster is running, and you want to start running containers!

You can now use any of the cluster/kube-*.sh commands to interact with your VM machines.
```
cluster/kubecfg.sh list /pods
cluster/kubecfg.sh list /services
cluster/kubecfg.sh list /replicationControllers
cluster/kubecfg.sh -p 8080:80 run dockerfile/nginx 3 myNginx

## begin wait for provision to complete, you can monitor the minions by doing
  vagrant ssh minion-1
  sudo docker images
  ## you should see it pulling the dockerfile/nginx image, once the above command returns it
  sudo docker ps
  ## you should see your container running!
  exit
## end wait

## back on the host, introspect kubernetes!
cluster/kubecfg.sh list /pods
cluster/kubecfg.sh list /services
cluster/kubecfg.sh list /replicationControllers
```

Congratulations!

### Testing

The following will run all of the end-to-end testing scenarios assuming you set your environment in cluster/kube-env.sh

```
hack/e2e-test.sh
```


### Troubleshooting

#### I just created the cluster, but I do not see my container running!

If this is your first time creating the cluster, the kubelet on each minion schedules a number of docker pull requests to fetch prerequisite images.  This can take some time and as a result may delay your initial pod getting provisioned.

#### I changed Kubernetes code, but its not running!

Are you sure there was no build error?  After running $ vagrant provison , scroll up and ensure that each Salt state was completed successfully on each box in the cluster.  Its very likely you see a build error due to an error in your source files!

## Running locally
In a separate tab of your terminal, run:

```
cd kubernetes
hack/local-up-cluster.sh
```

This will build and start a lightweight local cluster, consisting of a master and a single minion. Type Control-C to shut it down.

If you are running both a remote kubernetes cluster and the local cluster, you can determine which you talk to using the ```KUBERNETES_MASTER``` environment variable.


## Running on [CoreOS](http://coreos.com)
The folks at [CoreOS](http://coreos.com) have [instructions for running Kubernetes on CoreOS](http://coreos.com/blog/running-kubernetes-example-on-CoreOS-part-1/)

## Where to go next?
[Detailed example application](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/guestbook/README.md)

[Example of dynamic updates](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/update-demo/README.md)

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
#
# NOTE: The "../.." part seems odd but is correct, since the newly created
# links will be 2 levels down the tree.
cd kubernetes
ln -s ../../hooks/prepare-commit-msg .git/hooks/prepare-commit-msg
ln -s ../../hooks/commit-msg .git/hooks/commit-msg
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

### End-to-End tests
With a GCE account set up for running `cluster/kube-up.sh` (see Setup above):
```
cd kubernetes
hack/e2e-test.sh
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
