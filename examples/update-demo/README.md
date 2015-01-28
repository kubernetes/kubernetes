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
# Live update example
This example demonstrates the usage of Kubernetes to perform a live update on a running group of pods.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](https://github.com/GoogleCloudPlatform/kubernetes-new#contents):

```bash
$ cd kubernetes
$ hack/dev-build-and-up.sh
```

This example also assumes that you have [Docker](http://docker.io) installed on your local machine.

It also assumes that `$DOCKER_HUB_USER` is set to your Docker user id.  We use this to upload the docker images that are used in the demo.
```bash
$ export DOCKER_HUB_USER=my-docker-id
```

You may need to open the firewall for port 8080 using the [console][cloud-console] or the `gcloud` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion`:

```bash
$ gcloud compute firewall-rules create \
  --allow tcp:8080 --target-tags=kubernetes-minion \
  --zone=us-central1-a kubernetes-minion-8080
```

### Step Zero: Build the Docker images

This can take a few minutes to download/upload stuff.

```bash
$ cd examples/update-demo
$ ./0-build-images.sh
```

### Step One: Turn up the UX for the demo

You can use bash job control to run this in the background.  This can sometimes spew to the output so you could also run it in a different terminal.

```
$ ./1-run-web-proxy.sh &
Running local proxy to Kubernetes API Server.  Run this in a
separate terminal or run it in the background.

    http://localhost:8001/static/

+ ../../cluster/kubectl.sh proxy --www=local/
I0115 16:50:15.959551   19790 proxy.go:34] Starting to serve on localhost:8001
```

Now visit the the [demo website](http://localhost:8001/static).  You won't see anything much quite yet.

### Step Two: Run the controller
Now we will turn up two replicas of an image.  They all serve on port 8080, mapped to internal port 80

```bash
$ ./2-create-replication-controller.sh
```

After pulling the image from the Docker Hub to your worker nodes (which may take a minute or so) you'll see a couple of squares in the UI detailing the pods that are running along with the image that they are serving up.  A cute little nautilus.

### Step Three: Try resizing the controller

Now we will increase the number of replicas from two to four:

```bash
$ ./3-scale.sh
```

If you go back to the [demo website](http://localhost:8001/static/index.html) you should eventually see four boxes, one for each pod.

### Step Four: Update the docker image
We will now update the docker image to serve a different image by doing a rolling update to a new Docker image.

```bash
$ ./4-rolling-update.sh
```
The rollingUpdate command in kubectl will do 2 things:

1. Create a new replication controller with a pod template that uses the new image (`$DOCKER_HUB_USER/update-demo:kitten`)
2. Resize the old and new replication controllers until the new controller replaces the old. This will kill the current pods one at a time, spinnning up new ones to replace them.

Watch the [demo website](http://localhost:8001/static/index.html), it will update one pod every 10 seconds until all of the pods have the new image.

### Step Five: Bring down the pods

```bash
$ ./5-down.sh
```

This will first 'stop' the replication controller by turning the target number of replicas to 0.  It'll then delete that controller.

[cloud-console]: https://console.developer.google.com

### Step Six: Cleanup

To turn down a Kubernetes cluster:

```bash
$ cd ../..  # Up to kubernetes.
$ cluster/kube-down.sh
```

Kill the proxy running in the background:
After you are done running this demo make sure to kill it:

```bash
$ jobs
[1]+  Running                 ./1-run-web-proxy.sh &
$ kill %1
[1]+  Terminated: 15          ./1-run-web-proxy.sh
```


### Image Copyright

Note that he images included here are public domain.

* [kitten](http://commons.wikimedia.org/wiki/File:Kitten-stare.jpg)
* [nautilus](http://commons.wikimedia.org/wiki/File:Nautilus_pompilius.jpg)
