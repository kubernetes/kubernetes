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
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/mesos.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
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

Further information is available in the Kubernetes on Mesos [contrib directory][13].

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
git clone https://github.com/kubernetes/kubernetes
cd kubernetes
export KUBERNETES_CONTRIB=mesos
make
```

Set some environment variables.
The internal IP address of the master may be obtained via `hostname -i`.

```bash
export KUBERNETES_MASTER_IP=$(hostname -i)
export KUBERNETES_MASTER=http://${KUBERNETES_MASTER_IP}:8888
```

Note that KUBERNETES_MASTER is used as the api endpoint. If you have existing `~/.kube/config` and point to another endpoint, you need to add option `--server=${KUBERNETES_MASTER}` to kubectl in later steps.

### Deploy etcd

Start etcd and verify that it is running:

```bash
sudo docker run -d --hostname $(uname -n) --name etcd \
  -p 4001:4001 -p 7001:7001 quay.io/coreos/etcd:v2.0.12 \
  --listen-client-urls http://0.0.0.0:4001 \
  --advertise-client-urls http://${KUBERNETES_MASTER_IP}:4001
```

```console
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
export PATH="$(pwd)/_output/local/go/bin:$PATH"
```

Identify your Mesos master: depending on your Mesos installation this is either a `host:port` like `mesos-master:5050` or a ZooKeeper URL like `zk://zookeeper:2181/mesos`.
In order to let Kubernetes survive Mesos master changes, the ZooKeeper URL is recommended for production environments.

```bash
export MESOS_MASTER=<host:port or zk:// url>
```

Create a cloud config file `mesos-cloud.conf` in the current directory with the following contents:

```console
$ cat <<EOF >mesos-cloud.conf
[mesos-cloud]
        mesos-master        = ${MESOS_MASTER}
EOF
```

Now start the kubernetes-mesos API server, controller manager, and scheduler on the master node:

```console
$ km apiserver \
  --address=${KUBERNETES_MASTER_IP} \
  --etcd-servers=http://${KUBERNETES_MASTER_IP}:4001 \
  --service-cluster-ip-range=10.10.10.0/24 \
  --port=8888 \
  --cloud-provider=mesos \
  --cloud-config=mesos-cloud.conf \
  --secure-port=0 \
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
  --cluster-dns=10.10.10.10 \
  --cluster-domain=cluster.local \
  --v=2 >scheduler.log 2>&1 &
```

Disown your background jobs so that they'll stay running if you log out.

```bash
disown -a
```

#### Validate KM Services

Add the appropriate binary folder to your `PATH` to access kubectl:

```bash
export PATH=<path/to/kubernetes-directory>/platforms/linux/amd64:$PATH
```

Interact with the kubernetes-mesos framework via `kubectl`:

```console
$ kubectl get pods
NAME      READY     STATUS    RESTARTS   AGE
```

```console
# NOTE: your service IPs will likely differ
$ kubectl get services
NAME             LABELS                                    SELECTOR   IP(S)          PORT(S)
k8sm-scheduler   component=scheduler,provider=k8sm         <none>     10.10.10.113   10251/TCP
kubernetes       component=apiserver,provider=kubernetes   <none>     10.10.10.1     443/TCP
```

Lastly, look for Kubernetes in the Mesos web GUI by pointing your browser to
`http://<mesos-master-ip:port>`. Make sure you have an active VPN connection.
Go to the Frameworks tab, and look for an active framework named "Kubernetes".

## Spin up a pod

Write a JSON pod description to a local file:

```bash
$ cat <<EOPOD >nginx.yaml
```

```yaml
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

```console
$ kubectl create -f ./nginx.yaml
pods/nginx
```

Wait a minute or two while `dockerd` downloads the image layers from the internet.
We can use the `kubectl` interface to monitor the status of our pod:

```console
$ kubectl get pods
NAME      READY     STATUS    RESTARTS   AGE
nginx     1/1       Running   0          14s
```

Verify that the pod task is running in the Mesos web GUI. Click on the
Kubernetes framework. The next screen should show the running Mesos task that
started the Kubernetes pod.

## Launching kube-dns

Kube-dns is an addon for Kubernetes which adds DNS-based service discovery to the cluster. For a detailed explanation see [DNS in Kubernetes][4].

The kube-dns addon runs as a pod inside the cluster. The pod consists of three co-located containers:
- a local etcd instance
- the [skydns][11] DNS server
- the kube2sky process to glue skydns to the state of the Kubernetes cluster.

The skydns container offers DNS service via port 53 to the cluster. The etcd communication works via local 127.0.0.1 communication

We assume that kube-dns will use
- the service IP `10.10.10.10`
- and the `cluster.local` domain.

Note that we have passed these two values already as parameter to the apiserver above.

A template for an replication controller spinning up the pod with the 3 containers can be found at [cluster/addons/dns/skydns-rc.yaml.in][11] in the repository. The following steps are necessary in order to get a valid replication controller yaml file:

- replace `{{ pillar['dns_replicas'] }}`  with `1`
- replace `{{ pillar['dns_domain'] }}` with `cluster.local.`
- add `--kube_master_url=${KUBERNETES_MASTER}` parameter to the kube2sky container command.

In addition the service template at [cluster/addons/dns/skydns-svc.yaml.in][12] needs the following replacement:

- `{{ pillar['dns_server'] }}` with `10.10.10.10`.

To do this automatically:

```bash
sed -e "s/{{ pillar\['dns_replicas'\] }}/1/g;"\
"s,\(command = \"/kube2sky\"\),\\1\\"$'\n'"        - --kube_master_url=${KUBERNETES_MASTER},;"\
"s/{{ pillar\['dns_domain'\] }}/cluster.local/g" \
  cluster/addons/dns/skydns-rc.yaml.in > skydns-rc.yaml
sed -e "s/{{ pillar\['dns_server'\] }}/10.10.10.10/g" \
  cluster/addons/dns/skydns-svc.yaml.in > skydns-svc.yaml
```

Now the kube-dns pod and service are ready to be launched:

```bash
kubectl create -f ./skydns-rc.yaml
kubectl create -f ./skydns-svc.yaml
```

Check with `kubectl get pods --namespace=kube-system` that 3/3 containers of the pods are eventually up and running. Note that the kube-dns pods run in the `kube-system` namespace, not in  `default`.

To check that the new DNS service in the cluster works, we start a busybox pod and use that to do a DNS lookup. First create the `busybox.yaml` pod spec:

```bash
cat <<EOF >busybox.yaml
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  namespace: default
spec:
  containers:
  - image: busybox
    command:
      - sleep
      - "3600"
    imagePullPolicy: IfNotPresent
    name: busybox
  restartPolicy: Always
EOF
```

Then start the pod:

```bash
kubectl create -f ./busybox.yaml
```

When the pod is up and running, start a lookup for the Kubernetes master service, made available on 10.10.10.1 by default:

```bash
kubectl  exec busybox -- nslookup kubernetes
```

If everything works fine, you will get this output:

```console
Server:    10.10.10.10
Address 1: 10.10.10.10

Name:      kubernetes
Address 1: 10.10.10.1
```

## What next?

Try out some of the standard [Kubernetes examples][9].

Read about Kubernetes on Mesos' architecture in the [contrib directory][13].

**NOTE:** Some examples require Kubernetes DNS to be installed on the cluster.
Future work will add instructions to this guide to enable support for Kubernetes DNS.

**NOTE:** Please be aware that there are [known issues with the current Kubernetes-Mesos implementation][7].

[1]: http://mesosphere.com/docs/tutorials/run-hadoop-on-mesos-using-installer
[2]: http://mesosphere.com/docs/tutorials/run-spark-on-mesos
[3]: http://mesosphere.com/docs/tutorials/run-chronos-on-mesos
[4]: ../../cluster/addons/dns/README.md
[5]: http://open.mesosphere.com/getting-started/cloud/google/mesosphere/
[6]: http://mesos.apache.org/
[7]: ../../contrib/mesos/docs/issues.md
[8]: https://github.com/mesosphere/kubernetes-mesos/issues
[9]: ../../examples/
[10]: http://open.mesosphere.com/getting-started/cloud/google/mesosphere/#vpn-setup
[11]: ../../cluster/addons/dns/skydns-rc.yaml.in
[12]: ../../cluster/addons/dns/skydns-svc.yaml.in
[13]: ../../contrib/mesos/README.md


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/mesos.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
