## Running kubernetes locally via Docker

The following instructions show you how to set up a simple, single node kubernetes cluster using Docker.

Here's a diagram of what the final result will look like:
![Kubernetes Single Node on Docker](k8s-singlenode-docker.png)

### Step One: Run etcd
```sh
docker run --net=host -d gcr.io/google_containers/etcd:2.0.9 /usr/local/bin/etcd --addr=127.0.0.1:4001 --bind-addr=0.0.0.0:4001 --data-dir=/var/etcd/data
```

### Step Two: Run the master
```sh
docker run --net=host -d -v /var/run/docker.sock:/var/run/docker.sock  gcr.io/google_containers/hyperkube:v0.16.2 /hyperkube kubelet --api_servers=http://localhost:8080 --v=2 --address=0.0.0.0 --enable_server --hostname_override=127.0.0.1 --config=/etc/kubernetes/manifests
```

This actually runs the kubelet, which in turn runs a [pod](http://docs.k8s.io/pods.md) that contains the other master components.

### Step Three: Run the service proxy
*Note, this could be combined with master above, but it requires --privileged for iptables manipulation*
```sh
docker run -d --net=host --privileged gcr.io/google_containers/hyperkube:v0.16.2 /hyperkube proxy --master=http://127.0.0.1:8080 --v=2
```

### Test it out
At this point you should have a running kubernetes cluster.  You can test this by downloading the kubectl 
binary
([OS X](https://storage.googleapis.com/kubernetes-release/release/v0.16.2/bin/darwin/amd64/kubectl))
([linux](https://storage.googleapis.com/kubernetes-release/release/v0.16.2/bin/linux/amd64/kubectl))

*Note:*
On OS/X you will need to set up port forwarding via ssh:
```sh
boot2docker ssh -L8080:localhost:8080
```

List the nodes in your cluster by running::

```sh
kubectl get nodes
```

This should print:
```
NAME        LABELS    STATUS
127.0.0.1   <none>    Ready
```

If you are running different kubernetes clusters, you may need to specify ```-s http://localhost:8080``` to select the local cluster.

### Run an application
```sh
kubectl -s http://localhost:8080 run-container nginx --image=nginx --port=80
```

now run ```docker ps``` you should see nginx running.  You may need to wait a few minutes for the image to get pulled.

### Expose it as a service:
```sh
kubectl expose rc nginx --port=80
```

This should print:
```
NAME      LABELS    SELECTOR              IP          PORT(S)
nginx     <none>    run-container=nginx   <ip-addr>   80/TCP
```

Hit the webserver:
```sh
curl <insert-ip-from-above-here>
```

Note that you will need run this curl command on your boot2docker VM if you are running on OS X.

### A note on turning down your cluster
Many of these containers run under the management of the ```kubelet``` binary, which attempts to keep containers running, even if they fail.  So, in order to turn down
the cluster, you need to first kill the kubelet container, and then any other containers.

You may use ```docker ps -a | awk '{print $1}' | xargs docker kill```, note this removes _all_ containers running under Docker, so use with caution.
