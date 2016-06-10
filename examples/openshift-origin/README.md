<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## OpenShift Origin example

This example shows how to run OpenShift Origin as a pod on an existing Kubernetes cluster.

OpenShift Origin runs with a rich set of role based policy rules out of the box that requires authentication from users via certificates.  When run as a pod on an existing Kubernetes cluster, it proxies access to the underlying Kubernetes services to provide security.

As a result, this example is a complex end-to-end configuration that shows how to configure certificates for a service that runs on Kubernetes, and requires a number of configuration files to be injected dynamically via a secret volume to the pod.

This example will create a pod running the OpenShift Origin master. In addition, it will run a three-pod etcd setup to hold OpenShift content. OpenShift embeds Kubernetes in the stand-alone setup, so the configuration for OpenShift when it is running against an external Kubernetes cluster is different: content specific to Kubernetes will be stored in the Kubernetes etcd repository (i.e. pods, services, replication controllers, etc.), but OpenShift specific content (builds, images, users, policies, etc.) are stored in its etcd setup.

### Step 0: Prerequisites

This example assumes that you have an understanding of Kubernetes and that you have forked the repository.

OpenShift Origin creates privileged containers when running Docker builds during the source-to-image process.

If you are using a Salt based KUBERNETES_PROVIDER (**gce**, **vagrant**, **aws**), you should enable the
ability to create privileged containers via the API.

```sh
$ cd kubernetes
$ vi cluster/saltbase/pillar/privilege.sls

# If true, allow privileged containers to be created by API
allow_privileged: true
```

Now spin up a cluster using your preferred KUBERNETES_PROVIDER. Remember that `kube-up.sh` may start other pods on your minion nodes, so ensure that you have enough resources to run the five pods for this example.


```sh
$ export KUBERNETES_PROVIDER=${YOUR_PROVIDER}
$ cluster/kube-up.sh
```

Next, let's setup some variables, and create a local folder that will hold generated configuration files.

```sh
$ export OPENSHIFT_EXAMPLE=$(pwd)/examples/openshift-origin
$ export OPENSHIFT_CONFIG=${OPENSHIFT_EXAMPLE}/config
$ mkdir ${OPENSHIFT_CONFIG}

$ export ETCD_INITIAL_CLUSTER_TOKEN=$(python -c "import string; import random; print(''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(40)))")
$ export ETCD_DISCOVERY_TOKEN=$(python -c "import string; import random; print(\"etcd-cluster-\" + ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(5)))")
$ sed -i.bak -e "s/INSERT_ETCD_INITIAL_CLUSTER_TOKEN/\"${ETCD_INITIAL_CLUSTER_TOKEN}\"/g" -e "s/INSERT_ETCD_DISCOVERY_TOKEN/\"${ETCD_DISCOVERY_TOKEN}\"/g" ${OPENSHIFT_EXAMPLE}/etcd-controller.yaml
```

This will have created a `etcd-controller.yaml.bak` file in your directory, which you should remember to restore when doing cleanup (or use the given `cleanup.sh`). Finally, let's start up the external etcd pods and the discovery service necessary for their initialization:

```sh
$ kubectl create -f examples/openshift-origin/openshift-origin-namespace.yaml
$ kubectl create -f examples/openshift-origin/etcd-discovery-controller.yaml --namespace="openshift-origin"
$ kubectl create -f examples/openshift-origin/etcd-discovery-service.yaml --namespace="openshift-origin"
$ kubectl create -f examples/openshift-origin/etcd-controller.yaml --namespace="openshift-origin"
$ kubectl create -f examples/openshift-origin/etcd-service.yaml --namespace="openshift-origin"
```

### Step 1: Export your Kubernetes configuration file for use by OpenShift pod

OpenShift Origin uses a configuration file to know how to access your Kubernetes cluster with administrative authority.

```
$ cluster/kubectl.sh config view --output=yaml --flatten=true --minify=true > ${OPENSHIFT_CONFIG}/kubeconfig
```

The output from this command will contain a single file that has all the required information needed to connect to your Kubernetes cluster that you previously provisioned. This file should be considered sensitive, so do not share this file with untrusted parties.

We will later use this file to tell OpenShift how to bootstrap its own configuration.

### Step 2: Create an External Load Balancer to Route Traffic to OpenShift

An external load balancer is needed to route traffic to our OpenShift master service that will run as a pod on your Kubernetes cluster.


```sh
$ cluster/kubectl.sh create -f $OPENSHIFT_EXAMPLE/openshift-service.yaml --namespace="openshift-origin"
```

### Step 3: Generate configuration file for your OpenShift master pod

The OpenShift master requires a configuration file as input to know how to bootstrap the system.

In order to build this configuration file, we need to know the public IP address of our external load balancer in order to build default certificates.

Grab the public IP address of the service we previously created: the two-line script below will attempt to do so, but make sure to check that the IP was set as a result - if it was not, try again after a couple seconds.


```sh
$  export PUBLIC_OPENSHIFT_IP=$(kubectl get services openshift  --namespace="openshift-origin" --template="{{ index .status.loadBalancer.ingress 0 \"ip\" }}")
$  echo ${PUBLIC_OPENSHIFT_IP}
```

You can automate the process with the following script, as it might take more than a minute for the IP to be set and discoverable.

```shell
$ while [ ${#PUBLIC_OPENSHIFT_IP} -lt 1 ]; do
  	echo -n .
  	sleep 1
  	{
	  	export PUBLIC_OPENSHIFT_IP=$(kubectl get services openshift  --namespace="openshift-origin" --template="{{ index .status.loadBalancer.ingress 0 \"ip\" }}")
	  } 2> ${OPENSHIFT_EXAMPLE}/openshift-startup.log
	  if [[ ! ${PUBLIC_OPENSHIFT_IP} =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
		  export PUBLIC_OPENSHIFT_IP=""
  	fi
  done
$ echo
$ echo "Public OpenShift IP set to: ${PUBLIC_OPENSHIFT_IP}"
```

Ensure you have a valid PUBLIC_IP address before continuing in the example.

We now need to run a command on your host to generate a proper OpenShift configuration.  To do this, we will volume mount the configuration directory that holds your Kubernetes kubeconfig file from the prior step.


```sh
$ docker run --privileged -v ${OPENSHIFT_CONFIG}:/config openshift/origin start master --write-config=/config --kubeconfig=/config/kubeconfig --master=https://localhost:8443 --public-master=https://${PUBLIC_OPENSHIFT_IP}:8443 --etcd=http://etcd:2379
```

You should now see a number of certificates minted in your configuration directory, as well as a master-config.yaml file that tells the OpenShift master how to execute.  We need to make some adjustments to this configuration directory in order to allow the OpenShift cluster to use Kubernetes serviceaccounts. First, write the Kubernetes service account key to the `${OPENSHIFT_CONFIG}` directory. The following script assumes you are using GCE. If you are not, use `scp` or `ssh` to get the key from the master node running Kubernetes. It is usually located at `/srv/kubernetes/server.key`.

```shell
$ export ZONE=$(gcloud compute instances list | grep "${KUBE_GCE_INSTANCE_PREFIX}\-master" | awk '{print $2}' | head -1)
$ echo "sudo cat /srv/kubernetes/server.key; exit;" | gcloud compute ssh ${KUBE_GCE_INSTANCE_PREFIX}-master --zone ${ZONE} | grep -Ex "(^\-.*\-$|^\S+$)" > ${OPENSHIFT_CONFIG}/serviceaccounts.private.key

```

Although we are retrieving the private key from the Kubernetes master, OpenShift will take care of the conversion for us so that serviceaccounts are created with the public key. Edit your `master-config.yaml` file in the `${OPENSHIFT_CONFIG}` directory to add `serviceaccounts.private.key` to the list of `publicKeyFiles`:

```shell
$ sed -i -e 's/publicKeyFiles:.*$/publicKeyFiles:/g' -e '/publicKeyFiles:/a \ \ - serviceaccounts.private.key' ${OPENSHIFT_CONFIG}/master-config.yaml
```

Now, the configuration files are complete. In the next step, we will bundle the resulting configuration into a Kubernetes Secret that our OpenShift master pod will consume.

### Step 4: Bundle the configuration into a Secret

We now need to bundle the contents of our configuration into a secret for use by our OpenShift master pod.

OpenShift includes an experimental command to make this easier.

First, update the ownership for the files previously generated:

```
$ sudo -E chown -R ${USER} ${OPENSHIFT_CONFIG}
```

Then run the following command to collapse them into a Kubernetes secret.

```sh
$ docker run -it --privileged -e="KUBECONFIG=/config/admin.kubeconfig" -v ${OPENSHIFT_CONFIG}:/config openshift/origin cli secrets new openshift-config /config -o json &> examples/openshift-origin/secret.json
```

Now, lets create the secret in your Kubernetes cluster.

```sh
$ cluster/kubectl.sh create -f examples/openshift-origin/secret.json --namespace="openshift-origin"
```

**NOTE: This secret is secret and should not be shared with untrusted parties.**

### Step 5: Deploy OpenShift Master

We are now ready to deploy OpenShift.

We will deploy a pod that runs the OpenShift master.  The OpenShift master will delegate to the underlying Kubernetes
system to manage Kubernetes specific resources.  For the sake of simplicity, the OpenShift master will run with an embedded etcd to hold OpenShift specific content.  This demonstration will evolve in the future to show how to run etcd in a pod so that content is not destroyed if the OpenShift master fails.

```sh
$  cluster/kubectl.sh create -f ${OPENSHIFT_EXAMPLE}/openshift-controller.yaml --namespace="openshift-origin"
```

You should now get a pod provisioned whose name begins with openshift.

```sh
$ cluster/kubectl.sh get pods | grep openshift
$ cluster/kubectl.sh log openshift-t7147 origin
Running: cluster/../cluster/gce/../../cluster/../_output/dockerized/bin/linux/amd64/kubectl logs openshift-t7t47 origin
2015-04-30T15:26:00.454146869Z I0430 15:26:00.454005       1 start_master.go:296] Starting an OpenShift master, reachable at 0.0.0.0:8443 (etcd: [https://10.0.27.2:4001])
2015-04-30T15:26:00.454231211Z I0430 15:26:00.454223       1 start_master.go:297] OpenShift master public address is https://104.197.73.241:8443
```

Depending upon your cloud provider, you may need to open up an external firewall rule for tcp:8443.  For GCE, you can run the following:

```sh
$ gcloud compute --project "your-project" firewall-rules create "origin" --allow tcp:8443 --network "your-network" --source-ranges "0.0.0.0/0"
```

Consult your cloud provider's documentation for more information.

Open a browser and visit the OpenShift master public address reported in your log.

You can use the CLI commands by running the following:

```sh
$ docker run --privileged --entrypoint="/usr/bin/bash" -it -e="OPENSHIFTCONFIG=/config/admin.kubeconfig" -v ${OPENSHIFT_CONFIG}:/config openshift/origin
$ osc config use-context public-default
$ osc --help
```

## Cleanup

Clean up your cluster from resources created with this example:

```sh
$ ${OPENSHIFT_EXAMPLE}/cleanup.sh
```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/openshift-origin/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
