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

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](https://github.com/GoogleCloudPlatform/kubernetes-new#setup):

    $ cd kubernetes
    $ hack/dev-build-and-up.sh

This example also assumes that you have [Docker](http://docker.io) installed on your local machine.

It also assumes that `$DOCKER_HUB_USER` is set to your Docker user id.  We use this to upload the docker images that are used in the demo.

You may need to open the firewall for port 8080 using the [console][cloud-console] or the `gcutil` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion`:

```shell
$ gcutil addfirewall --allowed=tcp:8080 --target_tags=kubernetes-minion kubernetes-minion-8080
```

### Step One: Build the image

```shell
$ cd kubernetes/examples/update-demo
$ images/build-images.sh
```

### Step Two: Turn up the UX for the demo

```shell
$ ./0-run-web-proxy.sh &
```

This can sometimes spew to the output so you could also run it in a different terminal.

Now visit the the [demo website](http://localhost:8001/static).  You won't see anything much quite yet.

### Step Three: Run the controller
Now we will turn up two replicas of an image.  They all serve on port 8080, mapped to internal port 80

```shell
$ ./1-create-replication-controller.sh
```

After these pull the image (which may take a minute or so) you'll see a couple of squares in the UI detailing the pods that are running along with the image that they are serving up.  A cute little nautilus.

### Step Four: Try resizing the controller

Now we will increase the number of replicas from two to four:

```shell
$ ./2-scale.sh
```

If you go back to the [demo website](http://localhost:8001/static/index.html) you should eventually see four boxes, one for each pod.

### Step Five: Update the docker image
We will now update the docker image to serve a different image by doing a rolling update to a new Docker image.

```shell
$ ./3-rolling-update
```
The rollingUpdate command in kubecfg will do 2 things:

1. Update the template in the replication controller to the new image (`$DOCKER_HUB_USER/update-demo:kitten`)
2. Kill each of the pods one by one.  It'll let the replication controller create new pods to replace those that were killed.

Watch the UX, it will update one pod every 10 seconds until all of the pods have the new image.

### Step Five: Bring down the pods

```shell
$ ./4-down.sh
```

This will first 'stop' the replication controller by turning the target number of replicas to 0.  It'll then delete that controller.

[cloud-console]: https://console.developer.google.com

### Image Copyright

Note that he images included here are public domain.

* [kitten](http://commons.wikimedia.org/wiki/File:Kitten-stare.jpg)
* [nautilus](http://commons.wikimedia.org/wiki/File:Nautilus_pompilius.jpg)
