## OpenShift Origin example

This example shows how to run OpenShift Origin as a pod on an existing Kubernetes cluster.

This example demonstrates usage of a pod with a secret volume mount.

### Step 0: Prerequisites

This example assumes that you have a basic understanding of Kubernetes and that you have forked the repository and [turned up a Kubernetes cluster](https://github.com/GoogleCloudPlatform/kubernetes#contents):

This example has been tested against the **gce** and **vagrant** based KUBERNETES_PROVIDER.

```shell
$ cd kubernetes
$ export KUBERNETES_PROVIDER=gce
$ hack/dev-build-and-up.sh
```

### Step 1: Generate resources

The demonstration will require the following resources:

1. A Kubernetes Secret that contains information needed to securely communicate to your Kubernetes master as an administrator
2. A Kubernetes Pod that contains information for how to run OpenShift Origin that consumes this Secret securely
3. A Kubernetes Service that exposes OpenShift Origin API via an external load balancer
4. A Kubernetes Service that exposes OpenShift Origin UI via an external load balancer

To generate these resources, we will run a script that introspects your configured KUBERNETES_PROVIDER:

```shell
$ examples/openshift-origin/resource-generator.sh
```
A Kubernetes Secret was generated that contains the following data:

1. kubeconfig: a valid kubeconfig file that is used by OpenShift Origin to communicate to the master
2. kube-ca: a certificate authority for the Kubernetes master
3. kube-auth-path: a Kubernetes authorization file
4. kube-cert: a Kubernetes certificate
5. kube-key: a Kubernetes key file

As required by a Kubernetes secret, each piece of data is base64 encoded - with no line wraps.

You can view the file by doing:

```shell
$ cat examples/openshift-origin/secret.json
```

Caution:  This file contains all of the required information to operate as a Kubernetes admin on your cluster, so only share this file with trusted parties.

A Kubernetes Pod file was generated that can run OpenShift Origin on your cluster.

The OpenShift Origin pod file has a volume mount that references the Kubernetes secret we created to know how to work with the underlying Kubernetes provider.

You can view the file by doing:

```shell
$ cat examples/openshift-origin/pod.json
```

Finally, a Kubernetes service was generated for the UI and the API and available via an external load balancer:

``shell
$ cat examples/openshift-origin

### Step 2: Create the secret in Kubernetes

To provision the secret on Kubernetes:

```shell
$ cluster/kubectl.sh create -f examples/openshift-origin/secret.json
```

You should see your secret resource was created by listing:
```shell
$ cluster/kubectl.sh get secrets
```

### Step 3: Provisioning OpenShift Origin

To create the OpenShift Origin pod:

```shell
$ cluster/kubectl.sh create -f examples/openshift-origin/pod.json
```

### Step 4: Provisioning OpenShift Origin Services

To create the OpenShift Origin Services that expose the API and UI:

```shell
$ cluster/kubectl.sh create -f examples/openshift-origin/ui-service.json
$ cluster/kubectl.sh create -f examples/openshift-origin/api-service.json
```

### Step 5: Open Firewall Ports

If you are running on GCE, you need to open the following ports:

```shell
$ gcloud compute instances list

FIND THE MINION NAME PREFIX

$ gcloud compute firewall-rules create openshift-origin-node-8444 --allow tcp:8444 --target-tags kubernetes-minion-prq8
$ gcloud compute firewall-rules create openshift-origin-node-8443 --allow tcp:8443 --target-tags kubernetes-minion-prq8
```
### Step 4: Try out OpenShift Origin

TODO add more detail here: