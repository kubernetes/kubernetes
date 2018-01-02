# Overview
Kubernetes templates for Heketi and Gluster. The following documentation is setup
to deploy the containers in Kubernetes.  It is not a full setup.  For full
documentation, please visit the Heketi wiki page.

# Usage

## Deploy Gluster

* Get node name by running:

```
$ kubectl get nodes
```

* Deploy the GlusterFS DaemonSet

```
$ kubectl create -f gluster-daemonset.json
```

* Deploy gluster container onto specified node by setting the label
`storagenode=glusterfs` on that node:

```
$ kubectl label node <...node...> storagenode=glusterfs
```

Repeat as needed.

## Deploy Heketi

First you will need to deploy the bootstrap Heketi container:

```
$ kubectl create -f deploy-heketi-deployment.json
```

This will deploy the a Heketi container used to bootstrap the Heketi
database.  Please refer to the wiki Kubernetes Deployment page for
more information

