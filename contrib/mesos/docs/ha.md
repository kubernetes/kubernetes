# High Availability

Kubernetes on Mesos will eventually support two HA modes:

* [Hot-standby](#hot-standby) (*work-in-progress*)
* [Cold-standby](#cold-standby)

Hot-standby mode is currently still work-in-progress as controller manager is not
yet HA-aware (the work is being tracked [here][2]). Nevertheless, we will
describe how hot-standby mode is intended to work. It is recommended to use
cold-standby mode for HA for the time being until this work is done. In hot-standby
mode all master components (apiserver, controller manager, and scheduler)
actively run on every master node. Additional logic is added to the controller
manager and scheduler to coordinate their access to the etcd backend to deal
with concurrency issues when modifying cluster state. As apiserver does not
modify cluster state, multiple of these can run concurrently without
coordination.  When the leader (i.e., the node whose scheduler is active)
crashes, other master nodes will detect the failure after some time and then
elect a new leader.

In cold-standby mode, similar to hot-standby mode apiserver will actively run
on every master node.  However, only one scheduler and controller manager will
run at any instance in time. This is coordinated by a small external program
called `podmaster` that uses etcd to perform leadership selection, and only on
the leader node will the `podmaster` start the scheduler and controller
manager. Cold-standby mode is how Kubernetes supports HA, and more information
can be found [here][1].

## Hot-standby

### Scheduler

The implementation of the scheduler HA feature includes:

- Checkpointing by default (`--checkpoint`)
- Large failover-timeout by default (`--failover-timeout`)
- Hot-failover w/ multiple scheduler instances (`--ha`)
- Best effort task reconciliation on failover

#### Multiple Instances

Multiple scheduler instances may be run to support a warm-standby scenario in which one scheduler fails and another takes over immediately.
But at any moment in time only one scheduler is actually registered with the leading Mesos master.
Scheduler leader election is implemented using etcd so it is important to have an HA etcd configuration established for reliable scheduler HA.

It is currently recommended that no more than 2 scheduler instances be running at the same time.
Running more than 2 schedulers at once may work but has not been extensively tested.
YMMV.

#### Failover

Scheduler failover may be triggered by either the following events:

- loss of leadership when running in HA mode (`--ha`).
- the leading scheduler process receives a USR1 signal.

It is currently possible signal failover to a single, non-HA scheduler process.
In this case, if there are problems launching a replacement scheduler process then the cluster may be without a scheduler until another is manually started.

#### How To

##### Command Line Arguments

- `--ha` is required to enable scheduler HA and multi-scheduler leader election.
- `--km-path` or else (`--executor-path` and `--proxy-path`) should reference non-local-file URI's and must be identical across schedulers.

If you have HDFS installed on your slaves then you can specify HDFS URI locations for the binaries:

```shell
$ hdfs dfs -put -f bin/km hdfs:///km
$ ./bin/km scheduler ... --mesos-master=zk://zk1:2181,zk2:2181/mesos --ha --km-path=hdfs:///km
```

**IMPORTANT:** some command line parameters specified for the scheduler process are passed to the Kubelet-executor and so are subject to compatibility tests:

- a Mesos master will not recognize differently configured executors as being compatible, and so...
- a scheduler will refuse to accept any offer for slave resources if there are incompatible executors running on the slave.

Within the scheduler, compatibility is largely determined by comparing executor configuration hashes:
  a hash is calculated from a subset of the executor-related command line parameters provided to the scheduler process.
The command line parameters that affect the hash calculation are listed below.

- `--allow-privileged`
- `--api-servers`
- `--auth-path`
- `--cluster-*`
- `--executor-*`
- `--kubelet-*`
- `--km-path`
- `--mesos-cgroup-prefix`
- `--mesos-launch-grace-period`
- `--minion-*`
- `--profiling`
- `--proxy-*`
- `--static-pods-config`

## Cold-standby

Setting up Kubernetes on Mesos in cold-standby mode is similar to Kubernetes in
standalone mode described in [Kubernetes HA][1]. However, special attention is
needed when setting up K8sm scheduler so that when the currently active
scheduler crashes/dies, a new one can be instantiated and take over the work.
More precisely, the new scheduler needs to be compatible with the executors
that were started previously by the dead scheduler.

### Environment Variables

We will set up K8sm master on 2 nodes in HA mode. The same steps can be
extended to set up more master nodes to deal with more concurrent failures. We
will define a few environment variables first to describe the testbed
environment.

```
MESOS_IP=192.168.0.1
MESOS_PORT=5050

ETCD_IP=192.168.0.2
ETCD_PORT=4001

K8S_1_IP=192.168.0.3
K8S_2_IP=192.168.0.4
K8S_APISERVER_PORT=8080
K8S_SCHEDULER_PORT=10251

NGINX_IP=192.168.0.5
NGINX_APISERVER_PORT=80
NGINX_SCHEDULER_PORT=81
```

Other than the 2 K8sm master nodes (`192.168.0.3` and `192.168.0.4`), we also
define a Mesos master at `192.168.0.1`, an etcd server at `192.168.0.2`, and an
Nginx server that load balances between the 2 K8sm master nodes. 

### K8sm Container Image

We use podmaster to coordinate leadership selection amongst K8sm masters.
However, podmaster needs to run in a container (preferably in a pod), and on
the leader node, its podmaster will instantiate scheduler and controller
manager also in their separate pods. The podmaster image is pre-built and can
be obtained from `gcr.io/google_containers/podmaster`. An official image that
contains the `km` binary to start apiserver, scheduler, and controller
manager is not yet available. But it can be built fairly easily.

```shell
$ cat <<EOF >Dockerfile
FROM ubuntu
MAINTAINER Hai Huang <haih@us.ibm.com>
RUN mkdir -p /opt/kubernetes
COPY kubernetes/_output/dockerized/bin/linux/amd64/ /opt/kubernetes
ENTRYPOINT ["/opt/kubernetes/km"]
EOF
$ cat <<EOF >build.sh
#!/bin/bash
K8SM_IMAGE_NAME=haih/k8sm
git clone https://github.com/mesosphere/kubernetes
cd kubernetes
git checkout release-v0.7-v1.1
KUBERNETES_CONTRIB=mesos build/run.sh make
cd ..
sudo docker build -t $K8SM_IMAGE_NAME --no-cache .
EOF
$ chmod 755 build.sh
$ ./build.sh
```

Make sure Docker engine is running locally as we will compile Kubernetes using
a Docker image. One can also change the image name and which Kubernetes release
to compile by modifying the script. After the script has finished running,
there should be a local Docker image called `haih/k8sm` (use `docker images` to
check).

Optionally, we can also push the image to Docker Hub (i.e., `docker push
$K8SM_IMAGE_NAME`) so we do not have to compile the image on every K8sm master
node.

**IMPORTANT:** Mesosphere team is currently maintaining the stable K8sm release in
a separate [fork][3]. At the time of this writing, the latest stable release is
`release-v0.7-v1.1`.


### Configure ETCD

We assume there's an etcd server on `$ETCD_IP`. Ideally this should be a
cluster of etcd servers running in HA mode backed up by redundant persistent
storage. For testing purposes, on the etcd server one can spin up an etcd
instance in a Docker container.

```shell
$ docker run -d --hostname $(uname -n) --name etcd \
  -p ${ETCD_PORT}:${ETCD_PORT} \
  quay.io/coreos/etcd:v2.0.12 \
  --listen-client-urls http://0.0.0.0:${ETCD_PORT} \
  --advertise-client-urls http://${ETCD_IP}:${ETCD_PORT}
```

### Configure Podmaster

Since we plan to run all K8sm components and podmaster in pods, we can use
`kubelet` to bootstrap these pods by specifying a manifests directory. 

```shell
$ mkdir -p /etc/kubernetes/manifests/
$ mkdir -p /srv/kubernetes/manifests/
```

Once the kubelet has started, it will check the manifests directory periodically
to see if it needs to start or stop pods. Pods can be started by putting their
specification yaml files into the manifests directory, and subsequently they
can be stopped by removing these yaml files.

```shell
$ cat <<EOF > /etc/kubernetes/manifests/podmaster.yaml
apiVersion: v1
kind: Pod
metadata:
  name: kube-podmaster
  namespace: kube-system
spec:
  hostNetwork: true
  containers:
  - name: scheduler-elector
    image: gcr.io/google_containers/podmaster:1.1
    command:
    - /podmaster
    - --etcd-servers=http://${ETCD_IP}:${ETCD_PORT}
    - --key=scheduler
    - --whoami=${MY_IP}
    - --source-file=/src/manifests/scheduler.yaml
    - --dest-file=/dst/manifests/scheduler.yaml
    volumeMounts:
    - mountPath: /src/manifests
      name: manifest-src
      readOnly: true
    - mountPath: /dst/manifests
      name: manifest-dst
  - name: controller-manager-elector
    image: gcr.io/google_containers/podmaster:1.1
    command:
    - /podmaster
    - --etcd-servers=http://${ETCD_IP}:${ETCD_PORT}
    - --key=controller
    - --whoami=${MY_IP}
    - --source-file=/src/manifests/controller-mgr.yaml
    - --dest-file=/dst/manifests/controller-mgr.yaml
    terminationMessagePath: /dev/termination-log
    volumeMounts:
    - mountPath: /src/manifests
      name: manifest-src
      readOnly: true
    - mountPath: /dst/manifests
      name: manifest-dst
  volumes:
  - hostPath:
      path: /srv/kubernetes/manifests
    name: manifest-src
  - hostPath:
      path: /etc/kubernetes/manifests
    name: manifest-dst
EOF
```

One must change `$MY_IP` to either `$K8S_1_IP` or `K8S_2_IP` depending
on which master node you are currently setting up the podmaster. Podmasters
will compete with each other for leadership, and the winner will copy scheduler
and controller manager's pod specification yaml files from
`/srv/kubernetes/manifests/` to `/etc/kubernetes/manifests/`. When the kubelet
detects these new yaml files, it will start the corresponding pods.

### Configure Scheduler

The scheduler pod specification will be put into `/srv/kubernetes/manifests/`.

```shell
$ cat <<EOF > /srv/kubernetes/manifests/scheduler.yaml
apiVersion: v1
kind: Pod
metadata:
  name: kube-scheduler
  namespace: kube-system
spec:
  hostNetwork: true
  containers:
  - name: kube-scheduler
    image: haih/k8sm:latest
    imagePullPolicy: IfNotPresent
    command:
    - /opt/kubernetes/km
    - scheduler
    - --address=${MY_IP}
    - --advertised-address=${NGINX_IP}:${NGINX_SCHEDULER_PORT}
    - --mesos-master=${MESOS_IP}:${MESOS_PORT}
    - --etcd-servers=http://${ETCD_IP}:${ETCD_PORT}
    - --api-servers=${NGINX_IP}:${NGINX_APISERVER_PORT}
    - --v=10
EOF
```

Again, one must change `$MY_IP` to either `$K8S_1_IP` or `K8S_2_IP` depending
on which master node is currently being working on. Even though we have not set
up Nginx yet, we can still specify `--api-servers` and `--advertised-address`
using Nginx's address and ports (make sure Nginx is already running before
turning on the scheduler). Having `--api-servers` point to Nginx allows
executors to maintain connectivity to one of the apiservers even when one or
more apiservers is down as Nginx can automatically re-route requests to a
working apiserver.

It is critically important to point `--advertised-address` to Nginx so all the
schedulers would be assigned the same executor ID. Otherwise, if we assign
`--advertised-address=${K8S_1_IP}` on the first K8s master and
`--advertised-address=${K8S_2_IP}` on the second K8s master, they would
generate different executor IDs. During a fail-over, the new scheduler would
not be able to use the executor started by the failed scheduler. If so, one
could get this error message in the scheduler log:

> Declining incompatible offer...

### Configure Controller Manager

The controller manager pod specification will also be put into `/srv/kubernetes/manifests/`.

```shell
$ cat <<EOF > /srv/kubernetes/manifests/controller-mgr.yaml
apiVersion: v1
kind: Pod
metadata:
  name: kube-controller-manager
  namespace: kube-system
spec:
  hostNetwork: true
  containers:
  - name: kube-controller-manager
    image: haih/k8sm:latest
    imagePullPolicy: IfNotPresent
    command:
    - /opt/kubernetes/km
    - controller-manager
    - --master=http://${NGINX_IP}:${NGINX_APISERVER_PORT}
    - --cloud-provider=mesos
    - --cloud-config=/etc/kubernetes/mesos-cloud.conf
    volumeMounts:
    - mountPath: /etc/kubernetes
      name: kubernetes-config
      readOnly: true
  volumes:
  - hostPath:
      path: /etc/kubernetes
    name: kubernetes-config
EOF
```

Controller manager also needs a mesos configuration file as one of its
parameters, and this configuration file is written to
`/etc/kubernetes/mesos-cloud.conf`.

```shell
$ cat <<EOF >/etc/kubernetes/mesos-cloud.conf
[mesos-cloud]
        mesos-master        = ${MESOS_IP}:${MESOS_PORT}
EOF
```

### Configure Apiserver

Apiserver runs on every master node, so its specification file is put into
`/etc/kubernetes/manifests/`.

```shell
cat <<EOF > /etc/kubernetes/manifests/apiserver.yaml
apiVersion: v1
kind: Pod
metadata:
  name: kube-apiserver
  namespace: kube-system
spec:
  hostNetwork: true
  containers:
  - name: kube-apiserver
    image: haih/k8sm:latest
    imagePullPolicy: IfNotPresent
    command:
    - /opt/kubernetes/km
    - apiserver
    - --insecure-bind-address=0.0.0.0
    - --etcd-servers=http://${ETCD_IP}:${ETCD_PORT}
    - --allow-privileged=true
    - --service-cluster-ip-range=10.10.10.0/24
    - --insecure-port=${K8S_APISERVER_PORT}
    - --cloud-provider=mesos
    - --cloud-config=/etc/kubernetes/mesos-cloud.conf
    - --advertise-address=${MY_IP}
    ports:
    - containerPort: ${K8S_APISERVER_PORT}
      hostPort: ${K8S_APISERVER_PORT}
      name: local
    volumeMounts:
    - mountPath: /etc/kubernetes
      name: kubernetes-config
      readOnly: true
  volumes:
  - hostPath:
      path: /etc/kubernetes
    name: kubernetes-config
EOF
```

Again, one must change `$MY_IP` to either `$K8S_1_IP` or `K8S_2_IP`
depending on which master node is currently being working on.

To summarize our current setup: we have apiserver and podmaster's pod
specification files put into `/etc/kubernetes/manifests/` so they run on every
master node.  Scheduler and controller manager's pod specification files are
put into `/srv/kubernetes/manifests/`, and they will be copied into
`/etc/kubernetes/manifests/` by their podmaster if and only if their podmaster was
elected the leader.

### Configure Nginx

Nginx needs to be configured to load balance for both the apiserver and scheduler.
For testing purpose, one can start Nginx in a Docker container.

```shell
cat <<EOF >nginx.conf
events {
  worker_connections  4096;  ## Default: 1024
}

http {
  upstream apiservers {
    server ${K8S_1_IP}:${K8S_APISERVER_PORT};
    server ${K8S_2_IP}:${K8S_APISERVER_PORT};
  }

  upstream schedulers {
    server ${K8S_1_IP}:${K8S_SCHEDULER_PORT};
    server ${K8S_2_IP}:${K8S_SCHEDULER_PORT};
  }

  server {
    listen ${NGINX_APISERVER_PORT};
    location / {
      proxy_pass              http://apiservers;
      proxy_next_upstream     error timeout invalid_header http_500;
      proxy_connect_timeout   2;
      proxy_buffering off;
      proxy_read_timeout 12h;
      proxy_send_timeout 12h;
    }
  }

  server {
    listen ${NGINX_SCHEDULER_PORT};
    location / {
      proxy_pass              http://schedulers;
      proxy_next_upstream     error timeout invalid_header http_500;
      proxy_connect_timeout   2;
      proxy_buffering off;
      proxy_read_timeout 12h;
      proxy_send_timeout 12h;
    }
  }
}
EOF
$ docker run \
  -p $NGINX_APISERVER_PORT:$NGINX_APISERVER_PORT \
  -p $NGINX_SCHEDULER_PORT:$NGINX_SCHEDULER_PORT \
  --name nginx \
  -v `pwd`/nginx.conf:/etc/nginx/nginx.conf:ro \
  -d nginx:latest
```

For the sake of clarity, configuring Nginx to support HTTP over TLS/SPDY is
outside of our scope. However, one should keep in mind that without TLS/SPDY
properly configured, some `kubectl` commands might not work properly. This
problem is documented [here][4].

### Start Kubelet

To start everything up, we need to start the kubelet on K8s master nodes so
they can start apiserver and podmaster. On the leader node, podmaster will
subsequently start the scheduler and controller manager.

```shell
$ mkdir -p /var/log/kubernetes
$ kubelet \
  --api_servers=http://127.0.0.1:${K8S_APISERVER_PORT} \
  --register-node=false \
  --allow-privileged=true \
  --config=/etc/kubernetes/manifests \
  1>/var/log/kubernetes/kubelet.log 2>&1 &
```

### Verification

On each of the K8s master nodes, one can run `docker ps` to verify that there
is an apiserver pod and a podmaster pod running, and on one of the K8s master
nodes, there is a controller manager and a scheduler pod running.

One should also verify if we can create user pods in the K8sm cluster

```shell
$ export KUBERNETES_MASTER=http://${NGINX_IP}:${NGINX_APISERVER_PORT} 
$ kubectl create -f <userpod yaml file>
$ kubectl get pods
```

The pod should be shown in a `Running` state after some short amount of time.

### Tuning

During a fail-over, cold-standby mode takes some time before a new scheduler
can be started to take over the work from the failed one. However, one can
tune various parameters to shorten this time.

Podmaster has `--sleep` and `--ttl-secs` parameters that can be tuned, and both
allow for faster failure detection. However, it is probably not a good idea to
set `--ttl-secs` too small to minimize false positives.

Kubelet has `--file-check-frequency` parameter that controls how frequently it 
checks the manifests directory. It is set to 20 seconds by default.

[1]: http://kubernetes.io/v1.0/docs/admin/high-availability.html
[2]: https://github.com/mesosphere/kubernetes-mesos/issues/457
[3]: https://github.com/mesosphere/kubernetes
[4]: https://github.com/kubernetes/kubernetes/blob/master/contrib/mesos/docs/issues.md#kubectl

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/mesos/docs/ha.md?pixel)]()
