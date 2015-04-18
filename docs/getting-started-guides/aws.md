## Getting started on AWS

### Prerequisites

1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)
3. You need an AWS [instance profile and role](http://docs.aws.amazon.com/IAM/latest/UserGuide/instance-profiles.html) with EC2 full access.

### Cluster turnup

Using ```wget```
```sh
export KUBERNETES_PROVIDER=aws; wget -q -O - https://get.k8s.io | bash
```

or if you prefer ```curl```

```sh
export KUBERNETES_PROVIDER=aws; curl -sS https://get.k8s.io | bash
```


NOTE: The script will provision a new VPC and a 4 node k8s cluster in us-west-2 (Oregon). It'll also try to create or
reuse a keypair called "kubernetes", and IAM profiles called "kubernetes-master" and "kubernetes-minion".  If these
already exist, make sure you want them to be used here.

Once the cluster is up, it will print the ip address of your cluster, this process takes about 5 to 10 minutes.

```
export KUBERNETES_MASTER=https://<ip-address>
```

Also setup your path to point to the released binaries:
```
export PATH=$PATH:$PWD:/kubernetes/cluster
```

If you run into trouble come ask questions on IRC at #google-containers on freenode.


### Running a container (simple version)

Once you have your cluster created you can use ```${SOME_DIR}/kubernetes/cluster/kubectl.sh``` to access
the kubernetes api.

The `kubectl.sh` line below spins up two containers running
[Nginx](http://nginx.org/en/) running on port 80:

```bash
cluster/kubectl.sh run-container my-nginx --image=nginx --replicas=2 --port=80
```

To stop the containers:

```bash
cluster/kubectl.sh stop rc my-nginx
```

To delete the containers:

```bash
cluster/kubectl.sh delete rc my-nginx
```

### Running a container (more complete version)

```bash
cd kubernetes
cluster/kubectl.sh create -f docs/getting-started-guides/pod.json
```

Where pod.json contains something like:

```json
{
  "id": "php",
  "kind": "Pod",
  "apiVersion": "v1beta1",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "php",
      "containers": [{
        "name": "nginx",
        "image": "nginx",
        "ports": [{
          "containerPort": 80,
          "hostPort": 8081
        }],
        "livenessProbe": {
          "enabled": true,
          "type": "http",
          "initialDelaySeconds": 30,
          "httpGet": {
            "path": "/index.html",
            "port": 8081
          }
        }
      }]
    }
  },
  "labels": {
    "name": "foo"
  }
}
```

You can see your cluster's pods:

```bash
cluster/kubectl.sh get pods
```

and delete the pod you just created:

```bash
cluster/kubectl.sh delete pods php
```

Since this pod is scheduled on a minion running in AWS, you will have to enable incoming tcp traffic via the port specified in the
pod manifest before you see the nginx welcome page. After doing so, it should be visible at http://<external ip of minion running nginx>:<port from manifest>.

Look in `examples/` for more examples

### Tearing down the cluster
```bash
cd kubernetes
cluster/kube-down.sh
```

### Running examples

Take a look at [next steps](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/guestbook)

### Cloud Formation [optional]
There is a contributed [example](aws-coreos.md) from [CoreOS](http://www.coreos.com) using Cloud Formation.
