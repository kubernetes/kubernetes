## OpenShift Origin example

This example shows how to run OpenShift Origin as a pod on an existing LMKTFY cluster.

This example demonstrates usage of a pod with a secret volume mount.

### Step 0: Prerequisites

This example assumes that you have a basic understanding of LMKTFY and that you have forked the repository and [turned up a LMKTFY cluster](https://github.com/GoogleCloudPlatform/lmktfy#contents):

This example has been tested against the **gce** and **vagrant** based LMKTFYRNETES_PROVIDER.

```shell
$ cd lmktfy
$ export LMKTFYRNETES_PROVIDER=gce
$ hack/dev-build-and-up.sh
```

### Step 1: Generate resources

The demonstration will require the following resources:

1. A LMKTFY Secret that contains information needed to securely communicate to your LMKTFY master as an administrator
2. A LMKTFY Pod that contains information for how to run OpenShift Origin that consumes this Secret securely
3. A LMKTFY Service that exposes OpenShift Origin API via an external load balancer
4. A LMKTFY Service that exposes OpenShift Origin UI via an external load balancer

To generate these resources, we will run a script that introspects your configured LMKTFYRNETES_PROVIDER:

```shell
$ examples/openshift-origin/resource-generator.sh
```
A LMKTFY Secret was generated that contains the following data:

1. lmktfyconfig: a valid lmktfyconfig file that is used by OpenShift Origin to communicate to the master
2. lmktfy-ca: a certificate authority for the LMKTFY master
3. lmktfy-auth-path: a LMKTFY authorization file
4. lmktfy-cert: a LMKTFY certificate
5. lmktfy-key: a LMKTFY key file

As required by a LMKTFY secret, each piece of data is base64 encoded - with no line wraps.

You can view the file by doing:

```shell
$ cat examples/openshift-origin/secret.json
```

Caution:  This file contains all of the required information to operate as a LMKTFY admin on your cluster, so only share this file with trusted parties.

A LMKTFY Pod file was generated that can run OpenShift Origin on your cluster.

The OpenShift Origin pod file has a volume mount that references the LMKTFY secret we created to know how to work with the underlying LMKTFY provider.

You can view the file by doing:

```shell
$ cat examples/openshift-origin/pod.json
```

Finally, a LMKTFY service was generated for the UI and the API and available via an external load balancer:

``shell
$ cat examples/openshift-origin

### Step 2: Create the secret in LMKTFY

To provision the secret on LMKTFY:

```shell
$ cluster/lmktfyctl.sh create -f examples/openshift-origin/secret.json
```

You should see your secret resource was created by listing:
```shell
$ cluster/lmktfyctl.sh get secrets
```

### Step 3: Provisioning OpenShift Origin

To create the OpenShift Origin pod:

```shell
$ cluster/lmktfyctl.sh create -f examples/openshift-origin/pod.json
```

### Step 4: Provisioning OpenShift Origin Services

To create the OpenShift Origin Services that expose the API and UI:

```shell
$ cluster/lmktfyctl.sh create -f examples/openshift-origin/ui-service.json
$ cluster/lmktfyctl.sh create -f examples/openshift-origin/api-service.json
```

### Step 5: Open Firewall Ports

If you are running on GCE, you need to open the following ports:

```shell
$ gcloud compute instances list

FIND THE MINION NAME PREFIX

$ gcloud compute firewall-rules create openshift-origin-node-8444 --allow tcp:8444 --target-tags lmktfy-minion-prq8
$ gcloud compute firewall-rules create openshift-origin-node-8443 --allow tcp:8443 --target-tags lmktfy-minion-prq8
```
### Step 4: Try out OpenShift Origin

TODO add more detail here: