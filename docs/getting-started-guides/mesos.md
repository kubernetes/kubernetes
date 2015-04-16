## Getting started with Kubernetes on Mesos

<!-- TODO: Update, clean up. -->

Mesos allows dynamic sharing of cluster resources between Kubernetes and other first-class Mesos frameworks such as [Hadoop][1], [Spark][2], and [Chronos][3].
Mesos ensures applications from different frameworks running on your cluster are isolated and that resources are allocated fairly.

Running Kubernetes on Mesos allows you to easily move Kubernetes workloads from one cloud provider to another to your own physical datacenter.

This tutorial will walk you through setting up Kubernetes on a Mesos cluster on [Google Cloud Plaform][4].
It provides a step by step walk through of adding Kubernetes to a Mesos cluster and running the classic GuestBook demo application.
The walkthrough presented here is based on the v0.4.x series of the Kubernetes-Mesos project, which itself is based on Kubernetes v0.11.0.

### Prerequisites

* Understanding of [Apache Mesos][11]
* Mesos cluster on [Google Compute Engine][5]
* Identify the Mesos master node external IP from Mesosphere [cluster launch pad][12]
* A [VPN connection to the cluster][6].

### Deploy Kubernetes-Mesos

Log into the master node over SSH, replacing the placeholder below with the correct IP address.

```bash
ssh jclouds@${ip_address_of_master_node}
```

Build Kubernetes-Mesos.

```bash
$ git clone https://github.com/mesosphere/kubernetes-mesos k8sm
$ mkdir -p bin && sudo docker run --rm -v $(pwd)/bin:/target \
  -v $(pwd)/k8sm:/snapshot -e GIT_BRANCH=release-0.4 \
  mesosphere/kubernetes-mesos:build
```

Set some environment variables.
The internal IP address of the master is visible via the cluster details page on the Mesosphere launchpad, or may be obtained via `hostname -i`.

```bash
$ export servicehost=$(hostname -i)
$ export mesos_master=${servicehost}:5050
$ export KUBERNETES_MASTER=http://${servicehost}:8888
```
### Deploy etcd
Start etcd and verify that it is running:

```bash
$ sudo docker run -d --hostname $(hostname -f) --name etcd -p 4001:4001 -p 7001:7001 coreos/etcd
```

```bash
$ sudo docker ps
CONTAINER ID   IMAGE                COMMAND   CREATED   STATUS   PORTS                NAMES
fd7bac9e2301   coreos/etcd:latest   "/etcd"   5s ago    Up 3s    2379/tcp, 2380/...   etcd
```
It's also a good idea to ensure your etcd instance is reachable by testing it
```bash
curl -L http://$servicehost:4001/v2/keys/
```
If connectivity is OK, you will see an output of the available keys in etcd (if any).

### Start Kubernetes-Mesos Services
Start the kubernetes-mesos API server, controller manager, and scheduler on a Mesos master node:

```bash
$ ./bin/km apiserver \
  --address=${servicehost} \
  --mesos_master=${mesos_master} \
  --etcd_servers=http://${servicehost}:4001 \
  --portal_net=10.10.10.0/24 \
  --port=8888 \
  --cloud_provider=mesos \
  --v=1 >apiserver.log 2>&1 &

$ ./bin/km controller-manager \
  --master=$servicehost:8888 \
  --mesos_master=${mesos_master} \
  --v=1 >controller.log 2>&1 &

$ ./bin/km scheduler \
  --address=${servicehost} \
  --mesos_master=${mesos_master} \
  --etcd_servers=http://${servicehost}:4001 \
  --mesos_user=root \
  --api_servers=$servicehost:8888 \
  --v=2 >scheduler.log 2>&1 &
```

Kubernetes-mesos will start up kubelets automatically, but currently the service
proxy needs to be started manually. Start the service proxy on each Mesos slave:

```bash
$ sudo ./bin/km proxy \
  --bind_address=${servicehost} \
  --etcd_servers=http://${servicehost}:4001 \
  --logtostderr=true >proxy.log 2>&1 &
```

Disown your background jobs so that they'll stay running if you log out.

```bash
$ disown -a
```
#### Validate KM Services
Interact with the kubernetes-mesos framework via `kubectl`:

```bash
$ bin/kubectl get pods
POD        IP        CONTAINER(S)        IMAGE(S)        HOST        LABELS        STATUS
```

```bash
$ bin/kubectl get services       # your service IPs will likely differ
NAME            LABELS                                    SELECTOR            IP             PORT
kubernetes      component=apiserver,provider=kubernetes   <none>              10.10.10.2     443
kubernetes-ro   component=apiserver,provider=kubernetes   <none>              10.10.10.1     80
```
Lastly, use the Mesos CLI tool to validate the Kubernetes scheduler framework has been registered and running:
```bash
$ mesos state | grep "Kubernetes"
         "name": "Kubernetes",
```
Or, look for Kubernetes in the Mesos web GUI. You can get there by clicking the
Mesos logo on the Mesosphere launchpad page, or by pointing your browser to
`http://${mesos_master}`. Make sure you have an active [VPN connection][6].
Go to the Frameworks tab, and look for an active framework named "Kubernetes".

## Spin up a pod

Write a JSON pod description to a local file:

```bash
$ cat <<EOPOD >nginx.json
{ "kind": "Pod",
"apiVersion": "v1beta1",
"id": "nginx-id-01",
"desiredState": {
  "manifest": {
    "version": "v1beta1",
    "containers": [{
      "name": "nginx-01",
      "image": "nginx",
      "ports": [{
        "containerPort": 80,
        "hostPort": 31000
      }],
      "livenessProbe": {
        "enabled": true,
        "type": "http",
        "initialDelaySeconds": 30,
        "httpGet": {
          "path": "/index.html",
          "port": "8081"
        }
      }
    }]
  }
},
"labels": {
  "name": "foo"
} }
EOPOD
```

Send the pod description to Kubernetes using the `kubectl` CLI:

```bash
$ bin/kubectl create -f nginx.json
nginx-id-01
```

Wait a minute or two while `dockerd` downloads the image layers from the internet.
We can use the `kubectl` interface to monitor the status of our pod:

```bash
$ bin/kubectl get pods
POD          IP           CONTAINER(S)  IMAGE(S)          HOST                       LABELS                STATUS
nginx-id-01  172.17.5.27  nginx-01      nginx             10.72.72.178/10.72.72.178  cluster=gce,name=foo  Running
```

Verify that the pod task is running in the Mesos web GUI. Click on the
Kubernetes framework. The next screen should show the running Mesos task that
started the Kubernetes pod.

## Run the Example Guestbook App

Following the instructions from the kubernetes-mesos [examples/guestbook][7]:

```bash
$ export ex=k8sm/examples/guestbook
$ bin/kubectl create -f $ex/redis-master.json
$ bin/kubectl create -f $ex/redis-master-service.json
$ bin/kubectl create -f $ex/redis-slave-controller.json
$ bin/kubectl create -f $ex/redis-slave-service.json
$ bin/kubectl create -f $ex/frontend-controller.json

$ cat <<EOS >/tmp/frontend-service
{
  "id": "frontend",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 9998,
  "selector": {
    "name": "frontend"
  },
  "publicIPs": [
    "${servicehost}"
  ]
}
EOS
$ bin/kubectl create -f /tmp/frontend-service
```

Watch your pods transition from `Pending` to `Running`:

```bash
$ watch 'bin/kubectl get pods'
```

Review your Mesos cluster's tasks:

```bash
$ mesos ps
   TIME   STATE    RSS     CPU    %MEM  COMMAND  USER                   ID
 0:00:05    R    41.25 MB  0.5   64.45    none   root  0597e78b-d826-11e4-9162-42010acb46e2
 0:00:08    R    41.58 MB  0.5   64.97    none   root  0595b321-d826-11e4-9162-42010acb46e2
 0:00:10    R    41.93 MB  0.75  65.51    none   root  ff8fff87-d825-11e4-9162-42010acb46e2
 0:00:10    R    41.93 MB  0.75  65.51    none   root  0597fa32-d826-11e4-9162-42010acb46e2
 0:00:05    R    41.25 MB  0.5   64.45    none   root  ff8e01f9-d825-11e4-9162-42010acb46e2
 0:00:10    R    41.93 MB  0.75  65.51    none   root  fa1da063-d825-11e4-9162-42010acb46e2
 0:00:08    R    41.58 MB  0.5   64.97    none   root  b9b2e0b2-d825-11e4-9162-42010acb46e2
```
The number of Kubernetes pods listed earlier (from `bin/kubectl get pods`) should equal to the number active Mesos tasks listed the previous listing (`mesos ps`).

Next, determine the internal IP address of the front end [service portal][8]:

```bash
$ bin/kubectl get services
NAME            LABELS                                    SELECTOR            IP             PORT
kubernetes      component=apiserver,provider=kubernetes   <none>              10.10.10.2     443
kubernetes-ro   component=apiserver,provider=kubernetes   <none>              10.10.10.1     80
redismaster     <none>                                    name=redis-master   10.10.10.49    10000
redisslave      name=redisslave                           name=redisslave     10.10.10.109   10001
frontend        <none>                                    name=frontend       10.10.10.149   9998
```

Interact with the frontend application via curl using the front-end service IP address from above:

```bash
$ curl http://${frontend_service_ip_address}:9998/index.php?cmd=get\&key=messages
{"data": ""}
```

Or via the Redis CLI:

```bash
$ sudo apt-get install redis-tools
$ redis-cli -h ${redis_master_service_ip_address} -p 10000
10.233.254.108:10000> dump messages
"\x00\x06,world\x06\x00\xc9\x82\x8eHj\xe5\xd1\x12"
```
#### Test Guestbook App
Or interact with the frontend application via your browser, in 2 steps:

First, open the firewall on the master machine.

```bash
# determine the internal port for the frontend service portal
$ sudo iptables-save|grep -e frontend  # -- port 36336 in this case
-A KUBE-PORTALS-CONTAINER -d 10.10.10.149/32 -p tcp -m comment --comment frontend -m tcp --dport 9998 -j DNAT --to-destination 10.22.183.23:36336
-A KUBE-PORTALS-CONTAINER -d 10.22.183.23/32 -p tcp -m comment --comment frontend -m tcp --dport 9998 -j DNAT --to-destination 10.22.183.23:36336
-A KUBE-PORTALS-HOST -d 10.10.10.149/32 -p tcp -m comment --comment frontend -m tcp --dport 9998 -j DNAT --to-destination 10.22.183.23:36336
-A KUBE-PORTALS-HOST -d 10.22.183.23/32 -p tcp -m comment --comment frontend -m tcp --dport 9998 -j DNAT --to-destination 10.22.183.23:36336

# open up access to the internal port for the frontend service portal
$ sudo iptables -A INPUT -i eth0 -p tcp -m state --state NEW,ESTABLISHED -m tcp \
  --dport ${internal_frontend_service_port} -j ACCEPT
```

Next, add a firewall rule in the Google Cloud Platform Console. Choose Compute >
Compute Engine > Networks, click on the name of your mesosphere-* network, then
click "New firewall rule" and allow access to TCP port 9998.

![Google Cloud Platform firewall configuration][9]

Now, you can visit the guestbook in your browser!

![Kubernetes Guestbook app running on Mesos][10]

[1]: http://mesosphere.com/docs/tutorials/run-hadoop-on-mesos-using-installer
[2]: http://mesosphere.com/docs/tutorials/run-spark-on-mesos
[3]: http://mesosphere.com/docs/tutorials/run-chronos-on-mesos
[4]: http://cloud.google.com
[5]: https://google.mesosphere.com
[6]: http://mesosphere.com/docs/getting-started/cloud/google/mesosphere/#vpn-setup
[7]: https://github.com/mesosphere/kubernetes-mesos/tree/v0.4.0/examples/guestbook
[8]: https://github.com/GoogleCloudPlatform/kubernetes/blob/v0.11.0/docs/services.md#ips-and-portals
[9]: mesos/k8s-firewall.png
[10]: mesos/k8s-guestbook.png
[11]: http://mesos.apache.org/
[12]: https://google.mesosphere.com/clusters
