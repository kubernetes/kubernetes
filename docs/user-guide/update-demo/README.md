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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/user-guide/update-demo/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
<!--
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->

# Rolling update example

This example demonstrates the usage of Kubernetes to perform a [rolling update](../kubectl/kubectl_rolling-update.md) on a running group of [pods](../../../docs/user-guide/pods.md). See [here](../managing-deployments.md#updating-your-application-without-a-service-outage) to understand why you need a rolling update. Also check [rolling update design document](../../design/simple-rolling-update.md) for more information.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](../../../docs/getting-started-guides/):

```console
$ cd kubernetes
$ ./cluster/kube-up.sh
```

### Step One: Turn up the UX for the demo

You can use bash job control to run this in the background (note that you must use the default port -- 8001 -- for the following demonstration to work properly).
This can sometimes spew to the output so you could also run it in a different terminal. You have to run `kubectl proxy` in the root of the
Kubernetes repository. Otherwise you will get "404 page not found" errors as the paths will not match. You can find more information about `kubectl proxy`
[here](../../../docs/user-guide/kubectl/kubectl_proxy.md).

```console
$ kubectl proxy --www=docs/user-guide/update-demo/local/ &
I0218 15:18:31.623279   67480 proxy.go:36] Starting to serve on localhost:8001
```

Now visit the the [demo website](http://localhost:8001/static).  You won't see anything much quite yet.

### Step Two: Run the replication controller

Now we will turn up two replicas of an [image](../images.md).  They all serve on internal port 80.

```console
$ kubectl create -f docs/user-guide/update-demo/nautilus-rc.yaml
```

After pulling the image from the Docker Hub to your worker nodes (which may take a minute or so) you'll see a couple of squares in the UI detailing the pods that are running along with the image that they are serving up.  A cute little nautilus.

### Step Three: Try scaling the replication controller

Now we will increase the number of replicas from two to four:

```console
$ kubectl scale rc update-demo-nautilus --replicas=4
```

If you go back to the [demo website](http://localhost:8001/static/index.html) you should eventually see four boxes, one for each pod.

### Step Four: Update the docker image

We will now update the docker image to serve a different image by doing a rolling update to a new Docker image.

```console
$ kubectl rolling-update update-demo-nautilus --update-period=10s -f docs/user-guide/update-demo/kitten-rc.yaml
```

The rolling-update command in kubectl will do 2 things:

1. Create a new [replication controller](../../../docs/user-guide/replication-controller.md) with a pod template that uses the new image (`gcr.io/google_containers/update-demo:kitten`)
2. Scale the old and new replication controllers until the new controller replaces the old. This will kill the current pods one at a time, spinning up new ones to replace them.

Watch the [demo website](http://localhost:8001/static/index.html), it will update one pod every 10 seconds until all of the pods have the new image.
Note that the new replication controller definition does not include the replica count, so the current replica count of the old replication controller is preserved.
But if the replica count had been specified, the final replica count of the new replication controller will be equal this number.

### Step Five: Bring down the pods

```console
$ kubectl delete rc update-demo-kitten
```

This first stops the replication controller by turning the target number of replicas to 0 and then deletes the controller.

### Step Six: Cleanup

To turn down a Kubernetes cluster:

```console
$ ./cluster/kube-down.sh
```

Kill the proxy running in the background:
After you are done running this demo make sure to kill it:

```console
$ jobs
[1]+  Running                 ./kubectl proxy --www=local/ &
$ kill %1
[1]+  Terminated: 15          ./kubectl proxy --www=local/
```

### Updating the Docker images

If you want to build your own docker images, you can set `$DOCKER_HUB_USER` to your Docker user id and run the included shell script. It can take a few minutes to download/upload stuff.

```console
$ export DOCKER_HUB_USER=my-docker-id
$ ./docs/user-guide/update-demo/build-images.sh
```

To use your custom docker image in the above examples, you will need to change the image name in `docs/user-guide/update-demo/nautilus-rc.yaml` and `docs/user-guide/update-demo/kitten-rc.yaml`.

### Image Copyright

Note that the images included here are public domain.

* [kitten](http://commons.wikimedia.org/wiki/File:Kitten-stare.jpg)
* [nautilus](http://commons.wikimedia.org/wiki/File:Nautilus_pompilius.jpg)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/update-demo/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
