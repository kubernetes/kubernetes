<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes User Guide: Managing Applications: Quick start

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->
- [Kubernetes User Guide: Managing Applications: Quick start](#kubernetes-user-guide:-managing-applications:-quick-start)
  - [Launching a simple application](#launching-a-simple-application)
  - [Exposing your application to the Internet](#exposing-your-application-to-the-internet)
  - [Killing the application](#killing-the-application)
  - [What's next?](#what's-next?)

<!-- END MUNGE: GENERATED_TOC -->

This guide will help you get oriented to Kubernetes and running your first containers on the cluster.

## Launching a simple application

Once your application is packaged into a container and pushed to an image registry, you’re ready to deploy it to Kubernetes.

For example, [nginx](http://wiki.nginx.org/Main) is a popular HTTP server, with a [pre-built container on Docker hub](https://registry.hub.docker.com/_/nginx/). The [`kubectl run`](kubectl/kubectl_run.md) command below will create two nginx replicas, listening on port 80.

```bash
$ kubectl run my-nginx --image=nginx --replicas=2 --port=80
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR       REPLICAS
my-nginx     my-nginx       nginx      run=my-nginx   2
```

You can see that they are running by:
```bash
$ kubectl get po
NAME             READY     STATUS    RESTARTS   AGE
my-nginx-l8n3i   1/1       Running   0          29m
my-nginx-q7jo3   1/1       Running   0          29m
```

Kubernetes will ensure that your application keeps running, by automatically restarting containers that fail, spreading containers across nodes, and recreating containers on new nodes when nodes fail.
## Exposing your application to the Internet
Through integration with some cloud providers (for example Google Compute Engine and AWS EC2), Kubernetes enables you to request that it provision a public IP address for your application. To do this run:

```bash
$ kubectl expose rc my-nginx --port=80 --type=LoadBalancer
NAME       LABELS         SELECTOR       IP(S)     PORT(S)
my-nginx   run=my-nginx   run=my-nginx             80/TCP
```

To find the public IP address assigned to your application, execute:

```bash
$ kubectl get svc my-nginx -o json | grep \"ip\"
                   "ip": "130.111.122.213"
```

In order to access your nginx landing page, you also have to make sure that traffic from external IPs is allowed. Do this by opening a [firewall to allow traffic on port 80](../../docs/services-firewalls.md).

## Killing the application

To kill the application and delete its containers and public IP address, do:

```bash
$ kubectl delete rc my-nginx
replicationcontrollers/my-nginx
$ kubectl delete svc my-nginx
services/my-nginx
```

## What's next?

[Learn about how to configure common container parameters, such as commands and environment variables.](configuring-containers.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/quick-start.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
