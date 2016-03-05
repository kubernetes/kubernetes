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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Create the first cluster, but pass MULTIZONE to tell the cluster to manage multiple zones; creating nodes in us-central1-a.

GCE:
```
MULTIZONE=1 KUBE_GCE_ZONE=us-central1-a cluster/kube-up.sh
```

AWS:
```
MULTIZONE=1 KUBE_AWS_ZONE=us-west-2a cluster/kube-up.sh
```


View the nodes; you can see that they are tagged with the zone they are in (they are all in us-central1-a so far):

```
> kubectl get nodes

NAME                                         LABELS                                                                                                                                                                                                              STATUS    AGE
ip-172-20-0-155.us-west-2.compute.internal   beta.kubernetes.io/instance-type=m3.medium,failure-domain.beta.kubernetes.io/region=us-west-2,failure-domain.beta.kubernetes.io/zone=us-west-2a,kubernetes.io/hostname=ip-172-20-0-155.us-west-2.compute.internal   Ready     3m
ip-172-20-0-79.us-west-2.compute.internal    beta.kubernetes.io/instance-type=m3.medium,failure-domain.beta.kubernetes.io/region=us-west-2,failure-domain.beta.kubernetes.io/zone=us-west-2a,kubernetes.io/hostname=ip-172-20-0-79.us-west-2.compute.internal    Ready     3m
```

Add more nodes to the existing cluster, reusing the existing master, running in a different zone (us-central1-b):

```
KUBE_USE_EXISTING_MASTER=true MULTIZONE=1 KUBE_GCE_ZONE=us-central1-b cluster/kube-up.sh
```

AWS:
```
KUBE_USE_EXISTING_MASTER=true MULTIZONE=1 KUBE_SUBNET_CIDR=172.20.1.0/24 MASTER_INTERNAL_IP=172.20.0.9 KUBE_AWS_ZONE=us-west-2b cluster/kube-up.sh
```


View the nodes again; 3 more nodes should have launched and be tagged in us-central1-b:

```
> kubectl get nodes

NAME                                         LABELS                                                                                                                                                                                                              STATUS    AGE
ip-172-20-0-114.us-west-2.compute.internal   beta.kubernetes.io/instance-type=m3.medium,failure-domain.beta.kubernetes.io/region=us-west-2,failure-domain.beta.kubernetes.io/zone=us-west-2a,kubernetes.io/hostname=ip-172-20-0-114.us-west-2.compute.internal   Ready     36m
ip-172-20-0-13.us-west-2.compute.internal    beta.kubernetes.io/instance-type=m3.medium,failure-domain.beta.kubernetes.io/region=us-west-2,failure-domain.beta.kubernetes.io/zone=us-west-2a,kubernetes.io/hostname=ip-172-20-0-13.us-west-2.compute.internal    Ready     36m
ip-172-20-1-110.us-west-2.compute.internal   beta.kubernetes.io/instance-type=m3.medium,failure-domain.beta.kubernetes.io/region=us-west-2,failure-domain.beta.kubernetes.io/zone=us-west-2b,kubernetes.io/hostname=ip-172-20-1-110.us-west-2.compute.internal   Ready     8m
ip-172-20-1-216.us-west-2.compute.internal   beta.kubernetes.io/instance-type=m3.medium,failure-domain.beta.kubernetes.io/region=us-west-2,failure-domain.beta.kubernetes.io/zone=us-west-2b,kubernetes.io/hostname=ip-172-20-1-216.us-west-2.compute.internal   Ready     7m
```

Create a volume (only PersistentVolumes are supported for zone affinity):

```
kubectl create -f - <<EOF
{
  "kind": "PersistentVolumeClaim",
  "apiVersion": "v1",
  "metadata": {
    "name": "claim1",
    "annotations": {
        "volume.alpha.kubernetes.io/storage-class": "foo"
    }
  },
  "spec": {
    "accessModes": [
      "ReadWriteOnce"
    ],
    "resources": {
      "requests": {
        "storage": "5Gi"
      }
    }
  }
}
EOF
```

The PV is also tagged with the zone & region is is in (us-central1-a):

```
> kubectl get pv
NAME           LABELS                                                                                                        CAPACITY   ACCESSMODES   STATUS      CLAIM     REASON    AGE
my-data-disk   failure-domain.alpha.kubernetes.io/region=us-central1,failure-domain.alpha.kubernetes.io/zone=us-central1-a   5Gi        RWO           Available                       5s
```

Create a pod that uses the PVC:

```
kubectl create -f - <<EOF
kind: Pod
apiVersion: v1
metadata:
  name: mypod
spec:
  containers:
    - name: myfrontend
      image: nginx
      volumeMounts:
      - mountPath: "/var/www/html"
        name: mypd
  volumes:
    - name: mypd
      persistentVolumeClaim:
        claimName: claim1
EOF
```

Verify that the pod was created in us-central1-a (it had to be, to mount the volume):

```
> kubectl describe pod mypod | grep Node
Node:                           kubernetes-minion-lqrm/10.240.0.4

> kubectl get node kubernetes-minion-lqrm
NAME                     LABELS                                                                                                                                                      STATUS    AGE
kubernetes-minion-lqrm   failure-domain.alpha.kubernetes.io/region=us-central1,failure-domain.alpha.kubernetes.io/zone=us-central1-a,kubernetes.io/hostname=kubernetes-minion-lqrm   Ready     24m
```

Now we'll verify that pods in an RC are spread across zones.  Launch more nodes in us-central1-f:

```
KUBE_USE_EXISTING_MASTER=true MULTIZONE=1 KUBE_GCE_ZONE=us-central1-f cluster/kube-up.sh
```

AWS:
TODO: Make it so we don't have to specify MASTER_INTERNAL_IP

```
KUBE_USE_EXISTING_MASTER=true MULTIZONE=1 KUBE_SUBNET_CIDR=172.20.2.0/24 MASTER_INTERNAL_IP=172.20.0.9 KUBE_AWS_ZONE=us-west-2c cluster/kube-up.sh
```

kubectl get nodes
```

Create the guestbook-go example, which includes an RC of size 3, running a simple web app:

```
find examples/guestbook-go/ -name '*.json' | xargs -I {} kubectl create -f {}
```

The pods should be spread across all 3 zones:

```
> kubectl get pods | cut -f 1 -d ' ' | grep guestbook | xargs kubectl describe pod | grep Node
Node:                           kubernetes-minion-jwc0/10.240.0.7
Node:                           kubernetes-minion-jkst/10.240.0.10
Node:                           kubernetes-minion-lqrm/10.240.0.4

> kubectl get node kubernetes-minion-jwc0 kubernetes-minion-jkst kubernetes-minion-lqrm
NAME                     LABELS                                                                                                                                                      STATUS    AGE
kubernetes-minion-jwc0   failure-domain.alpha.kubernetes.io/region=us-central1,failure-domain.alpha.kubernetes.io/zone=us-central1-b,kubernetes.io/hostname=kubernetes-minion-jwc0   Ready     25m
NAME                     LABELS                                                                                                                                                      STATUS    AGE
kubernetes-minion-jkst   failure-domain.alpha.kubernetes.io/region=us-central1,failure-domain.alpha.kubernetes.io/zone=us-central1-f,kubernetes.io/hostname=kubernetes-minion-jkst   Ready     3m
NAME                     LABELS                                                                                                                                                      STATUS    AGE
kubernetes-minion-lqrm   failure-domain.alpha.kubernetes.io/region=us-central1,failure-domain.alpha.kubernetes.io/zone=us-central1-a,kubernetes.io/hostname=kubernetes-minion-lqrm   Ready     30m

```

On AWS:
```
> kubectl get pods | cut -f 1 -d ' ' | grep guestbook | xargs kubectl describe pod | grep Node
Node:                           ip-172-20-2-236.us-west-2.compute.internal/172.20.2.236
Node:                           ip-172-20-0-114.us-west-2.compute.internal/172.20.0.114
Node:                           ip-172-20-1-110.us-west-2.compute.internal/172.20.1.110
```

(note the subnets)

Load-balancers can span zones; the guestbook-go example includes an example load-balanced service:

```
> kubectl describe service guestbook | grep LoadBalancer.Ingress
LoadBalancer Ingress:   130.211.126.21

> ip=130.211.126.21

> curl -s http://${ip}:3000/env | grep HOSTNAME
  "HOSTNAME": "guestbook-44sep",

> (for i in `seq 20`; do curl -s http://${ip}:3000/env | grep HOSTNAME; done)  | sort | uniq
  "HOSTNAME": "guestbook-44sep",
  "HOSTNAME": "guestbook-hum5n",
  "HOSTNAME": "guestbook-ppm40",
```

When you're done, clean up:

```
KUBE_USE_EXISTING_MASTER=true KUBE_GCE_ZONE=us-central1-b cluster/kube-down.sh
KUBE_USE_EXISTING_MASTER=true KUBE_GCE_ZONE=us-central1-f cluster/kube-down.sh
KUBE_GCE_ZONE=us-central1-a cluster/kube-down.sh
```

```
KUBE_AWS_ZONE=us-west-2a cluster/kube-down.sh
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/ubernetes-lite.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
