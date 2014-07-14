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
    $ hack/build-go.sh

This example also assumes that you have [Docker](http://docker.io) installed on your local machine.

It also assumes that ```$DOCKER_USER``` is set to your docker user id.

You may need to open the firewall for port 8080 using the [console][cloud-console] or the `gcutil` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion`:

```shell
$ gcutil addfirewall --allowed=tcp:8080 --target_tags=kubernetes-minion kubernetes-minion-8080
```

### Step One: Build the image

    $ cd kubernetes/examples/update-demo/image
    $ docker build -t $DOCKER_USER/data .
    $ docker push $DOCKER_USER/data

### Step Two: Run the controller
Now we will turn up two replicas of that image.  They all serve on port 8080, mapped to internal port 80

    $ cd kubernetes
    $ cluster/kubecfg.sh -p 8080:80 run $DOCKER_USER/data 2 dataController

### Step Three: Turn up the UX for the demo
In a different terminal:

    $ cd kubernetes
    $ cluster/kubecfg.sh -proxy -www examples/update-demo/local/

Now visit the the [demo website](http://localhost:8001/static/index.html).  You should see two light blue squares with pod IDs and ip addresses.

### Step Four: Try resizing the controller
Now we will increase the number of replicas from two to four:

    $ cd kubernetes
    $ cluster/kubecfg.sh resize dataController 4

If you go back to the [demo website](http://localhost:8001/static/index.html) you should eventually see four boxes, one for each pod.

### Step Five: Update the docker image
We will now update the docker image to serve a different color.

    $ cd kubernetes/examples/update-demo/image
    $ ${EDITOR} data.json

Edit the ```color``` value so that it is a new color.  For example:
```js
{
  "color": "#F00"
}
```
Will set the color to red.

Once you are happy with the color, build a new image:

    $ docker build -t $DOCKER_USER/data .
    $ docker push $DOCKER_USER/data

### Step Six: Roll the update out to your servers
We will now update the servers that are running out in your cluster.

    $ cd kubernetes
    $ cluster/kubecfg.sh -u=30s rollingupdate dataController

Watch the UX, it will update one pod every 30 seconds until all of the pods have the new color.

[cloud-console]: https://console.developer.google.com
