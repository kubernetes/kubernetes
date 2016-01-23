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

```
MULTIZONE=1 KUBE_GCE_ZONE=us-central1-a cluster/kube-up.sh
```

View the nodes; you can see that they are tagged with the zone they are in (they are all in us-central1-a so far):

```
kubectl get nodes
```

Add more nodes to the existing cluster, reusing the existing master, running in a different zone (us-central1-b):

```
KUBE_USE_EXISTING_MASTER=true MULTIZONE=1 KUBE_GCE_ZONE=us-central1-b cluster/kube-up.sh
```

View the nodes again; 3 more nodes should have launched and be tagged in us-central1-b:

```
kubectl get nodes
```

Create a volume (only PersistentVolumes are supported for zone affinity):

```
gcloud compute disks create --size=5GB --zone=us-central1-a my-data-disk

kubectl create -f - <<EOF
kind: PersistentVolume
apiVersion: v1
metadata:
  name: my-data-disk
spec:
  capacity:
    storage: "5Gi"
  accessModes:
    - "ReadWriteOnce"
  gcePersistentDisk:
    pdName: "my-data-disk"
    fsType: "ext4"
EOF
```

The PV is also tagged with the zone & region is is in (us-central1-a):

```
> kubectl get pv
NAME           LABELS                                                                                                        CAPACITY   ACCESSMODES   STATUS      CLAIM     REASON    AGE
my-data-disk   failure-domain.alpha.kubernetes.io/region=us-central1,failure-domain.alpha.kubernetes.io/zone=us-central1-a   5Gi        RWO           Available                       5s
```

Create a PersistentVolumeClaim and a pod that uses it:

```
kubectl create -f - <<EOF
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: pvc1
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
EOF

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
        claimName: pvc1
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

Load-balancers can span zones; the guestbook-go example includes an example load-balanced service:

```
> kubectl describe service guestbook | grep LoadBalancer.Ingress
LoadBalancer Ingress:   130.211.126.21

> ip=130.211.126.21

> curl -s http://${ip}:3000/env | grep HOSTNAME
  "HOSTNAME": "guestbook-44sep",

> (for i in `seq 20`; do curl -s http://<ip>:3000/env | grep HOSTNAME; done)  | sort | uniq
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




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/ubernetes-lite.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
