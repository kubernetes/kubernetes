## Getting started locally

### Requirements 

#### Linux

Not running Linux? Consider running Linux in a local virtual machine with [Vagrant](vagrant.md), or on a cloud provider like [Google Compute Engine](gce.md)

#### Docker

At least [Docker](https://docs.docker.com/installation/#installation) 1.0.0+. Ensure the Docker daemon is running and can be contacted by the user you plan to run as (try `docker ps`).

#### etcd

You need an [etcd](https://github.com/coreos/etcd) somewhere in your path. Get the [latest release](https://github.com/coreos/etcd/releases/) and place it in `/usr/bin`.


### Starting the cluster

In a separate tab of your terminal, run:

```
cd kubernetes
hack/local-up-cluster.sh
```

This will build and start a lightweight local cluster, consisting of a master and a single minion. Type Control-C to shut it down.

You can use the cluster/kubecfg.sh script to interact with the local cluster.

```
cd kubernetes
modify cluster/kube-env.sh:
  KUBERNETES_PROVIDER="local"

cluster/kubecfg.sh => interact with the local cluster
```

### Running a container

Your cluster is running, and you want to start running containers!

You can now use any of the cluster/kubecfg.sh commands to interact with your local setup.
```
cluster/kubecfg.sh list /pods
cluster/kubecfg.sh list /services
cluster/kubecfg.sh list /replicationControllers
cluster/kubecfg.sh -p 8080:80 run dockerfile/nginx 1 myNginx


## begin wait for provision to complete, you can monitor the docker pull by opening a new terminal
  sudo docker images
  ## you should see it pulling the dockerfile/nginx image, once the above command returns it
  sudo docker ps
  ## you should see your container running!
  exit
## end wait

## introspect kubernetes!
cluster/kubecfg.sh list /pods
cluster/kubecfg.sh list /services
cluster/kubecfg.sh list /replicationControllers
```

Congratulations!

### Troubleshooting

#### I cannot create a replication controller with replica size greater than 1!  What gives?

You are running a single minion setup.  This has the limitation of only supporting a single replica of a given pod.  If you are interested in running with larger replica sizes, we encourage you to try the local vagrant setup or one of the cloud providers.

#### I changed Kubernetes code, how do I run it?

```
cd kubernetes
hack/build-go.sh
hack/local-up-cluster.sh
```
