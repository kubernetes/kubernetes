## Node selection example

This example shows how to assign a pod to a specific node or to one of a set of nodes using node labels and the nodeSelector field in a pod specification. Generally this is unnecessary, as the scheduler will take care of things for you, but you may want to do so in certain circumstances like to ensure that your pod ends up on a machine with an SSD attached to it.

### Step Zero: Prerequisites

This example assumes that you have a basic understanding of kubernetes pods and that you have [turned up a Kubernetes cluster](https://github.com/GoogleCloudPlatform/kubernetes#documentation).

### Step One: Attach label to the node

Run `kubectl get nodes` to get the names of the nodes. Pick out the one that you want to add a label to. Note that label keys must be in the form of DNS labels (as described in the [identifiers doc](/docs/design/identifiers.md)), meaning that they are not allowed to contain any upper-case letters. Then run `kubectl get node <node-name> -o yaml > node.yaml`. The contents of the file should look something like this:

<pre>
apiVersion: v1beta1
creationTimestamp: 2015-02-03T01:16:46Z
hostIP: 104.154.60.112
id: <node-name>
kind: Node
resourceVersion: 12
resources:
  capacity:
    cpu: "1"
    memory: 4.0265318e+09
selfLink: /api/v1beta1/minions/<node-name>
status:
  conditions:
  - kind: Ready
    lastTransitionTime: null
    status: Full
uid: 526a4156-ab42-11e4-9817-42010af0258d
</pre>

Add the labels that you want to the file like this:

<pre>
apiVersion: v1beta1
creationTimestamp: 2015-02-03T01:16:46Z
hostIP: 104.154.60.112
id: <node-name>
kind: Node
<b>labels:
  disktype: ssd</b>
resourceVersion: 12
resources:
  capacity:
    cpu: "1"
    memory: 4.0265318e+09
selfLink: /api/v1beta1/minions/<node-name>
status:
  conditions:
  - kind: Ready
    lastTransitionTime: null
    status: Full
uid: 526a4156-ab42-11e4-9817-42010af0258d
</pre>

Then update the node by running `kubectl update -f node.yaml`. Make sure that the resourceVersion you use in your update call is the same as the resourceVersion returned by the get call. If something about the node changes between your get and your update, the update will fail because the resourceVersion will have changed.

Note that as of 2015-02-03 there are a couple open issues that prevent this from working without modification. Due to [issue #3005](https://github.com/GoogleCloudPlatform/kubernetes/issues/3005), you have to remove all status-related fields from the file, which is both everything under the `status` field as well as the `hostIP` field (removing hostIP isn't required in v1beta3). Due to [issue 4041](https://github.com/GoogleCloudPlatform/kubernetes/issues/4041), you may have to modify the representation of the resource capacity numbers to make them integers. These are both temporary, and fixes are being worked on. In the meantime, you would actually call `kubectl update -f node.yaml` with a file that looks like this:

<pre>
apiVersion: v1beta1
creationTimestamp: 2015-02-03T01:16:46Z
id: <node-name>
kind: Node
<b>labels:
  disktype: ssd</b>
resourceVersion: 12
resources:
  capacity:
    cpu: "1"
    memory: 4026531800
selfLink: /api/v1beta1/minions/<node-name>
uid: 526a4156-ab42-11e4-9817-42010af0258d
</pre>


### Step Two: Add a nodeSelector field to your pod configuration

Take whatever pod config file you want to run, and add a nodeSelector section to it, like this. For example, if this is my pod config:

<pre>
apiVersion: v1beta1
desiredState:
  manifest:
    containers:
      - image: nginx
        name: nginx
    id: nginx
    version: v1beta1
id: nginx
kind: Pod
labels:
  env: test
</pre>

Then add a nodeSelector like so:

<pre>
apiVersion: v1beta1
desiredState:
  manifest:
    containers:
      - image: nginx
        name: nginx
    id: nginx
    version: v1beta1
id: nginx
kind: Pod
labels:
  env: test
<b>nodeSelector:
  disktype: ssd</b>
</pre>

When you then run `kubectl create -f pod.yaml`, the pod will get scheduled on the node that you attached the label to! You can verify that it worked by running `kubectl get pods` and looking at the "host" that the pod was assigned to.

### Conclusion

While this example only covered one node, you can attach labels to as many nodes as you want. Then when you schedule a pod with a nodeSelector, it can be scheduled on any of the nodes that satisfy that nodeSelector. Be careful that it will match at least one node, however, because if it doesn't the pod won't be scheduled at all.
