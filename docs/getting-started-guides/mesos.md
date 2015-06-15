Getting started with Kubernetes on Mesos
----------------------------------------

**Table of Contents**

- [About Kubernetes on Mesos](#about-kubernetes-on-mesos)
    - [Prerequisites](#prerequisites)
    - [Deploy Kubernetes-Mesos](#deploy-kubernetes-mesos)
    - [Deploy etcd](#deploy-etcd)
    - [Start Kubernetes-Mesos Services](#start-kubernetes-mesos-services)
        - [Validate KM Services](#validate-km-services)
- [Spin up a pod](#spin-up-a-pod)
- [Run the Example Guestbook App](#run-the-example-guestbook-app)
        - [Test Guestbook App](#test-guestbook-app)

## About Kubernetes on Mesos
<!-- TODO: Update, clean up. -->

Mesos allows dynamic sharing of cluster resources between Kubernetes and other first-class Mesos frameworks such as [Hadoop][1], [Spark][2], and [Chronos][3].
Mesos also ensures applications from different frameworks running on your cluster are isolated and that resources are allocated fairly among them.

Mesos clusters can be deployed on nearly every IaaS cloud provider infrastructure or in your own physical datacenter. Kubernetes on Mesos runs on-top of that and therefore allows you to easily move Kubernetes workloads from one of these environments to the other.

This tutorial will walk you through setting up Kubernetes on a Mesos cluster.
It provides a step by step walk through of adding Kubernetes to a Mesos cluster and starting your first pod with an nginx webserver.

**NOTE:** There are [known issues with the current implementation][7] and support for centralized logging and monitoring is not yet available.
Please [file an issue against the kubernetes-mesos project][8] if you have problems completing the steps below.

### Prerequisites

* Understanding of [Apache Mesos][6]
* A running [Mesos cluster on Google Compute Engine][5]
* A [VPN connection][10] to the cluster
* A machine in the cluster which should become the Kubernetes *master node* with:
  * GoLang > 1.2
  * make (i.e. build-essential)
  * Docker

**Note**: You *can*, but you *don't have to* deploy Kubernetes-Mesos on the same machine the Mesos master is running on.

### Deploy Kubernetes-Mesos

Log into the future Kubernetes *master node* over SSH, replacing the placeholder below with the correct IP address.

```bash
ssh jclouds@${ip_address_of_master_node}
```

Build Kubernetes-Mesos.

```bash
$ git clone https://github.com/GoogleCloudPlatform/kubernetes
$ cd kubernetes
$ export KUBERNETES_CONTRIB=mesos
$ make
```

Set some environment variables.
The internal IP address of the master may be obtained via `hostname -i`.

```bash
$ export KUBERNETES_MASTER_IP=$(hostname -i)
$ export KUBERNETES_MASTER=http://${KUBERNETES_MASTER_IP}:8888
```

### Deploy etcd
Start etcd and verify that it is running:

```bash
$ sudo docker run -d --hostname $(uname -n) --name etcd -p 4001:4001 -p 7001:7001 quay.io/coreos/etcd:v2.0.12
```

```bash
$ sudo docker ps
CONTAINER ID   IMAGE                        COMMAND   CREATED   STATUS   PORTS                NAMES
fd7bac9e2301   quay.io/coreos/etcd:v2.0.12  "/etcd"   5s ago    Up 3s    2379/tcp, 2380/...   etcd
```
It's also a good idea to ensure your etcd instance is reachable by testing it
```bash
curl -L http://${KUBERNETES_MASTER_IP}:4001/v2/keys/
```
If connectivity is OK, you will see an output of the available keys in etcd (if any).

### Start Kubernetes-Mesos Services
Update your PATH to more easily run the Kubernetes-Mesos binaries:
```bash
$ export PATH="$(pwd)/_output/local/go/bin:$PATH"
```
Identify your Mesos master: depending on your Mesos installation this is either a `host:port` like `mesos_master:5050` or a ZooKeeper URL like `zk://zookeeper:2181/mesos`.
In order to let Kubernetes survive Mesos master changes, the ZooKeeper URL is recommended for production environments.
```bash
$ export MESOS_MASTER=<host:port or zk:// url>
```
Create a cloud config file `mesos-cloud.conf` in the current directory with the following contents:
```bash
$ cat <<EOF >mesos-cloud.conf
[mesos-cloud]
        mesos-master        = ${MESOS_MASTER}
EOF
```

Now start the kubernetes-mesos API server, controller manager, and scheduler on the master node:

```bash
$ km apiserver \
  --address=${KUBERNETES_MASTER_IP} \
  --etcd-servers=http://${KUBERNETES_MASTER_IP}:4001 \
  --service-cluster-ip-range=10.10.10.0/24 \
  --port=8888 \
  --cloud-provider=mesos \
  --cloud-config=mesos-cloud.conf \
  --v=1 >apiserver.log 2>&1 &

$ km controller-manager \
  --master=${KUBERNETES_MASTER_IP}:8888 \
  --cloud-provider=mesos \
  --cloud-config=./mesos-cloud.conf  \
  --v=1 >controller.log 2>&1 &

$ km scheduler \
  --address=${KUBERNETES_MASTER_IP} \
  --mesos-master=${MESOS_MASTER} \
  --etcd-servers=http://${KUBERNETES_MASTER_IP}:4001 \
  --mesos-user=root \
  --api-servers=${KUBERNETES_MASTER_IP}:8888 \
  --v=2 >scheduler.log 2>&1 &
```

Disown your background jobs so that they'll stay running if you log out.

```bash
$ disown -a
```
#### Validate KM Services
Interact with the kubernetes-mesos framework via `kubectl`:

```bash
$ kubectl get pods
NAME      READY     REASON    RESTARTS   AGE
```

```bash
# NOTE: your service IPs will likely differ
$ kubectl get services
NAME             LABELS                                    SELECTOR   IP(S)          PORT(S)
k8sm-scheduler   component=scheduler,provider=k8sm         <none>     10.10.10.113   10251/TCP
kubernetes       component=apiserver,provider=kubernetes   <none>     10.10.10.1     443/TCP
```

Lastly, look for Kubernetes in the Mesos web GUI by pointing your browser to
`http://<mesos_master_ip:port>`. Make sure you have an active VPN connection.
Go to the Frameworks tab, and look for an active framework named "Kubernetes".

## Spin up a pod

Write a JSON pod description to a local file:

```bash
$ cat <<EOPOD >nginx.yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
EOPOD
```

Send the pod description to Kubernetes using the `kubectl` CLI:

```bash
$ kubectl create -f nginx.yaml
pods/nginx
```

Wait a minute or two while `dockerd` downloads the image layers from the internet.
We can use the `kubectl` interface to monitor the status of our pod:

```bash
$ kubectl get pods
NAME      READY     REASON    RESTARTS   AGE
nginx     1/1       Running   0          14s
```

Verify that the pod task is running in the Mesos web GUI. Click on the
Kubernetes framework. The next screen should show the running Mesos task that
started the Kubernetes pod.

## What next?

Try out some of the standard [Kubernetes examples][9].

**NOTE:** Some examples require Kubernetes DNS to be installed on the cluster.
Future work will add instructions to this guide to enable support for Kubernetes DNS.

**NOTE:** Please be aware that there are [known issues with the current Kubernetes-Mesos implementation][7].

[1]: http://mesosphere.com/docs/tutorials/run-hadoop-on-mesos-using-installer
[2]: http://mesosphere.com/docs/tutorials/run-spark-on-mesos
[3]: http://mesosphere.com/docs/tutorials/run-chronos-on-mesos
[5]: http://open.mesosphere.com/getting-started/cloud/google/mesosphere/
[6]: http://mesos.apache.org/
[7]: https://github.com/mesosphere/kubernetes-mesos/blob/master/docs/issues.md
[8]: https://github.com/mesosphere/kubernetes-mesos/issues
[9]: ../../examples/
[10]: http://open.mesosphere.com/getting-started/cloud/google/mesosphere/#vpn-setup

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/mesos.md?pixel)]()
