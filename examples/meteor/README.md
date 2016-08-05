<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/examples/meteor/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Meteor on Kubernetes
====================

This example shows you how to package and run a
[Meteor](https://www.meteor.com/) app on Kubernetes.

Get started on Google Compute Engine
------------------------------------

Meteor uses MongoDB, and we will use the `GCEPersistentDisk` type of
volume for persistent storage. Therefore, this example is only
applicable to [Google Compute
Engine](https://cloud.google.com/compute/). Take a look at the
[volumes documentation](../../docs/user-guide/volumes.md) for other options.

First, if you have not already done so:

1. [Create](https://cloud.google.com/compute/docs/quickstart) a
[Google Cloud Platform](https://cloud.google.com/) project.
2. [Enable
billing](https://developers.google.com/console/help/new/#billing).
3. Install the [gcloud SDK](https://cloud.google.com/sdk/).

Authenticate with gcloud and set the gcloud default project name to
point to the project you want to use for your Kubernetes cluster:

```sh
gcloud auth login
gcloud config set project <project-name>
```

Next, start up a Kubernetes cluster:

```sh
wget -q -O - https://get.k8s.io | bash
```

Please see the [Google Compute Engine getting started
guide](../../docs/getting-started-guides/gce.md) for full
details and other options for starting a cluster.

Build a container for your Meteor app
-------------------------------------

To be able to run your Meteor app on Kubernetes you need to build a
Docker container for it first. To do that you need to install
[Docker](https://www.docker.com) Once you have that you need to add 2
files to your existing Meteor project `Dockerfile` and
`.dockerignore`.

`Dockerfile` should contain the below lines. You should replace the
`ROOT_URL` with the actual hostname of your app.

```
FROM chees/meteor-kubernetes
ENV ROOT_URL http://myawesomeapp.com
```

The `.dockerignore` file should contain the below lines. This tells
Docker to ignore the files on those directories when it's building
your container.

```
.meteor/local
packages/*/.build*
```

You can see an example meteor project already set up at:
[meteor-gke-example](https://github.com/Q42/meteor-gke-example). Feel
free to use this app for this example.

> Note: The next step will not work if you have added mobile platforms
> to your meteor project. Check with `meteor list-platforms`

Now you can build your container by running this in
your Meteor project directory:

```
docker build -t my-meteor .
```

Pushing to a registry
---------------------

For the [Docker Hub](https://hub.docker.com/), tag your app image with
your username and push to the Hub with the below commands. Replace
`<username>` with your Hub username.

```
docker tag my-meteor <username>/my-meteor
docker push <username>/my-meteor
```

For [Google Container
Registry](https://cloud.google.com/tools/container-registry/), tag
your app image with your project ID, and push to GCR. Replace
`<project>` with your project ID.

```
docker tag my-meteor gcr.io/<project>/my-meteor
gcloud docker push gcr.io/<project>/my-meteor
```

Running
-------

Now that you have containerized your Meteor app it's time to set up
your cluster. Edit [`meteor-controller.json`](meteor-controller.json)
and make sure the `image:` points to the container you just pushed to
the Docker Hub or GCR.

We will need to provide MongoDB a persistent Kubernetes volume to
store its data. See the [volumes documentation](../../docs/user-guide/volumes.md) for
options. We're going to use Google Compute Engine persistent
disks. Create the MongoDB disk by running:

```
gcloud compute disks create --size=200GB mongo-disk
```

Now you can start Mongo using that disk:

```
kubectl create -f examples/meteor/mongo-pod.json
kubectl create -f examples/meteor/mongo-service.json
```

Wait until Mongo is started completely and then start up your Meteor app:

```
kubectl create -f examples/meteor/meteor-service.json
kubectl create -f examples/meteor/meteor-controller.json
```

Note that [`meteor-service.json`](meteor-service.json) creates a load balancer, so
your app should be available through the IP of that load balancer once
the Meteor pods are started. We also created the service before creating the rc to
aid the scheduler in placing pods, as the scheduler ranks pod placement according to
service anti-affinity (among other things). You can find the IP of your load balancer
by running:

```
kubectl get service meteor --template="{{range .status.loadBalancer.ingress}} {{.ip}} {{end}}"
```

You will have to open up port 80 if it's not open yet in your
environment. On Google Compute Engine, you may run the below command.

```
gcloud compute firewall-rules create meteor-80 --allow=tcp:80 --target-tags kubernetes-minion
```

What is going on?
-----------------

Firstly, the `FROM chees/meteor-kubernetes` line in your `Dockerfile`
specifies the base image for your Meteor app. The code for that image
is located in the `dockerbase/` subdirectory. Open up the `Dockerfile`
to get an insight of what happens during the `docker build` step. The
image is based on the Node.js official image. It then installs Meteor
and copies in your apps' code. The last line specifies what happens
when your app container is run.

```sh
ENTRYPOINT MONGO_URL=mongodb://$MONGO_SERVICE_HOST:$MONGO_SERVICE_PORT /usr/local/bin/node main.js
```

Here we can see the MongoDB host and port information being passed
into the Meteor app. The `MONGO_SERVICE...` environment variables are
set by Kubernetes, and point to the service named `mongo` specified in
[`mongo-service.json`](mongo-service.json). See the [environment
documentation](../../docs/user-guide/container-environment.md) for more details.

As you may know, Meteor uses long lasting connections, and requires
_sticky sessions_. With Kubernetes you can scale out your app easily
with session affinity. The
[`meteor-service.json`](meteor-service.json) file contains
`"sessionAffinity": "ClientIP"`, which provides this for us. See the
[service
documentation](../../docs/user-guide/services.md#virtual-ips-and-service-proxies) for
more information.

As mentioned above, the mongo container uses a volume which is mapped
to a persistent disk by Kubernetes. In [`mongo-pod.json`](mongo-pod.json) the container
section specifies the volume:

```json
{
        "volumeMounts": [
          {
            "name": "mongo-disk",
            "mountPath": "/data/db"
          }
```

The name `mongo-disk` refers to the volume specified outside the
container section:

```json
{
    "volumes": [
      {
        "name": "mongo-disk",
        "gcePersistentDisk": {
          "pdName": "mongo-disk",
          "fsType": "ext4"
        }
      }
    ],
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/meteor/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
