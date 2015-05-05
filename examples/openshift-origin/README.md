## OpenShift Origin example

This example shows how to run OpenShift Origin as a pod on an existing Kubernetes cluster.

OpenShift Origin runs with a rich set of role based policy rules out of the box that requires authentication from users
via certificates.  When run as a pod on an existing Kubernetes cluster, it proxies access to the underlying Kubernetes services
to provide security.

As a result, this example is a complex end-to-end configuration that shows how to configure certificates for a service that runs
on Kubernetes, and requires a number of configuration files to be injected dynamically via a secret volume to the pod.

### Step 0: Prerequisites

This example assumes that you have an understanding of Kubernetes and that you have forked the repository.

OpenShift Origin creates privileged containers when running Docker builds during the source-to-image process.

If you are using a Salt based KUBERNETES_PROVIDER (**gce**, **vagrant**, **aws**), you should enable the
ability to create privileged containers via the API.

```shell
$ cd kubernetes
$ vi cluster/saltbase/pillar/privilege.sls

# If true, allow privileged containers to be created by API
allow_privileged: true
```

Now spin up a cluster using your preferred KUBERNETES_PROVIDER

```shell
$ export KUBERNETES_PROVIDER=gce
$ cluster/kube-up.sh
```

Next, let's setup some variables, and create a local folder that will hold generated configuration files.

```shell
$ export OPENSHIFT_EXAMPLE=$(pwd)/examples/openshift-origin
$ export OPENSHIFT_CONFIG=${OPENSHIFT_EXAMPLE}/config
$ mkdir ${OPENSHIFT_CONFIG}
```

### Step 1: Export your Kubernetes configuration file for use by OpenShift pod

OpenShift Origin uses a configuration file to know how to access your Kubernetes cluster with administrative authority.

```
$ cluster/kubectl.sh config view --output=yaml --flatten=true --minify=true > ${OPENSHIFT_CONFIG}/kubeconfig
```

The output from this command will contain a single file that has all the required information needed to connect to your
Kubernetes cluster that you previously provisioned.   This file should be considered sensitive, so do not share this file with
untrusted parties.

We will later use this file to tell OpenShift how to bootstap its own configuration.

### Step 2: Create an External Load Balancer to Route Traffic to OpenShift

An external load balancer is needed to route traffic to our OpenShift master service that will run as a pod on your
Kubernetes cluster.


```shell
$ cluster/kubectl.sh create -f $OPENSHIFT_EXAMPLE/openshift-service.yaml
```

### Step 3: Generate configuration file for your OpenShift master pod

The OpenShift master requires a configuration file as input to know how to bootstrap the system.

In order to build this configuration file, we need to know the public IP address of our external load balancer in order to
build default certificates.

Grab the public IP address of the service we previously created.

```shell
$ export PUBLIC_IP=$(cluster/kubectl.sh get services openshift --template="{{ index .spec.publicIPs 0 }}")
$ echo $PUBLIC_IP
```

Ensure you have a valid PUBLIC_IP address before continuing in the example.

We now need to run a command on your host to generate a proper OpenShift configuration.  To do this, we will volume mount the configuration directory that holds your Kubernetes kubeconfig file from the prior step.

```shell
docker run --privileged -v ${OPENSHIFT_CONFIG}:/config openshift/origin start master --write-config=/config --kubeconfig=/config/kubeconfig --master=https://localhost:8443 --public-master=https://${PUBLIC_IP}:8443
```

You should now see a number of certificates minted in your configuration directory, as well as a master-config.yaml file that tells the OpenShift master how to execute.  In the next step, we will bundle this into a Kubernetes Secret that our OpenShift master pod will consume.

### Step 4: Bundle the configuration into a Secret

We now need to bundle the contents of our configuration into a secret for use by our OpenShift master pod.

OpenShift includes an experimental command to make this easier.

First, update the ownership for the files previously generated:

```
$ sudo -E chown ${USER} -R ${OPENSHIFT_CONFIG}
```

Then run the following command to collapse them into a Kubernetes secret.

```shell
docker run -i -t --privileged -e="OPENSHIFTCONFIG=/config/admin.kubeconfig" -v ${OPENSHIFT_CONFIG}:/config openshift/origin ex bundle-secret openshift-config -f /config &> ${OPENSHIFT_EXAMPLE}/secret.json
```

Now, lets create the secret in your Kubernetes cluster.

```shell
$ cluster/kubectl.sh create -f ${OPENSHIFT_EXAMPLE}/secret.json
```

**NOTE: This secret is secret and should not be shared with untrusted parties.**

### Step 5: Deploy OpenShift Master

We are now ready to deploy OpenShift.

We will deploy a pod that runs the OpenShift master.  The OpenShift master will delegate to the underlying Kubernetes
system to manage Kubernetes specific resources.  For the sake of simplicity, the OpenShift master will run with an embedded etcd to hold OpenShift specific content.  This demonstration will evolve in the future to show how to run etcd in a pod so that content is not destroyed if the OpenShift master fails.

```shell
$  cluster/kubectl.sh create -f ${OPENSHIFT_EXAMPLE}/openshift-controller.yaml
```

You should now get a pod provisioned whose name begins with openshift.

```shell
$ cluster/kubectl.sh get pods | grep openshift
$ cluster/kubectl.sh log openshift-t7147 origin
Running: cluster/../cluster/gce/../../cluster/../_output/dockerized/bin/linux/amd64/kubectl log openshift-t7t47 origin
2015-04-30T15:26:00.454146869Z I0430 15:26:00.454005       1 start_master.go:296] Starting an OpenShift master, reachable at 0.0.0.0:8443 (etcd: [https://10.0.27.2:4001])
2015-04-30T15:26:00.454231211Z I0430 15:26:00.454223       1 start_master.go:297] OpenShift master public address is https://104.197.73.241:8443
```

Depending upon your cloud provider, you may need to open up an external firewall rule for tcp:8443.  For GCE, you can run the following:

```shell
gcloud compute --project "your-project" firewall-rules create "origin" --allow tcp:8443 --network "your-network" --source-ranges "0.0.0.0/0"
```

Consult your cloud provider's documentation for more information.

Open a browser and visit the OpenShift master public address reported in your log.

You can use the CLI commands by running the following:

```shell
$ docker run --privileged --entrypoint="/usr/bin/bash" -it -e="OPENSHIFTCONFIG=/config/admin.kubeconfig" -v ${OPENSHIFT_CONFIG}:/config openshift/origin
$ osc config use-context public-default
$ osc --help
```
