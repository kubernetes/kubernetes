<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/examples/porting-steps/k8s/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

Build and push to a registry
----------------------------
For normal development, you would need to do this step. For this
example, the sample app has been built and pushed to [Google Container
Registry](https://cloud.google.com/tools/container-registry/) at
`gcr.io/google-samples/steps-twotier:k8s`. Therefore, you may skip to the
next step if you wish

Install and configure [Docker](https://docs.docker.com/installation/).

You may use any Docker registry you wish, these steps demonstrate
pushing to
[GCR](https://cloud.google.com/tools/container-registry/). Follow the
linked steps to setup the gcloud tool, then build and push:

```
docker build -t gcr.io/<project-id>/twotier .
gcloud docker push gcr.io/<project-id>/twotier
```

Edit [`twotier.yaml`](twotier.yaml) and change the `image:` line to
point to the container you pushed to your registry.

Prep the Persistent Disk
-------
Here we are using Google Compute Engine persistent disks for the mysql
database storage, see the [volumes documentation](https://github.com/docs/volumes.md)
for other options. Check the `volumes:` section in
[`mysql.yaml`](mysql.yaml) for how this is configured.

Create the physical disk:

```
gcloud compute disks create --size=200GB mysql-disk
```

Start up your pods on Kubernetes
------------
Have a kuberenetes cluster running, with working kubectl. [Getting
Started](/docs/getting-started-guides)

```
kubectl create -f ./mysql.yaml
kubectl create -f ./twotier.yaml
```

Open port 80 in your Kubernetes environment. On GCE you may run:

```
gcloud compute firewall-rules create k8s-80 --allow=tcp:80 --target-tags kubernetes-minion
```

Check it out
------------

```
kubectl describe service twotier
```

Look for `LoadBalancer Ingress` for the external IP of your
service. If your environment does not support external load balancers,
you will have to find the external IPs of your Kubernetes nodes.

In your browser, go to `http://<ip>`




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/porting-steps/k8s/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
