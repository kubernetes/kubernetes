# Private Docker Registry in Kubernetes

Kubernetes offers an optional private Docker registry addon, which you can turn
on when you bring up a cluster or install later.  This gives you a place to
store truly private Docker images for your cluster.

## How it works

The private registry runs as a `Pod` in your cluster.  It does not currently
support SSL or authentication, which triggers Docker's "insecure registry"
logic.  To work around this, we run a proxy on each node in the cluster,
exposing a port onto the node (via a hostPort), which Docker accepts as
"secure", since it is accessed by `localhost`.

## Turning it on

Some cluster installs (e.g. GCE) support this as a cluster-birth flag.  The
`ENABLE_CLUSTER_REGISTRY` variable in `cluster/gce/config-default.sh` governs
whether the registry is run or not.  To set this flag, you can specify
`KUBE_ENABLE_CLUSTER_REGISTRY=true` when running `kube-up.sh`.  If your cluster
does not include this flag, the following steps should work.  Note that some of
this is cloud-provider specific, so you may have to customize it a bit.

### Make some storage

The primary job of the registry is to store data.  To do that we have to decide
where to store it.  For cloud environments that have networked storage, we can
use Kubernetes's `PersistentVolume` abstraction.  The following template is
expanded by `salt` in the GCE cluster turnup, but can easily be adapted to
other situations:

<!-- BEGIN MUNGE: EXAMPLE registry-pv.yaml.in -->
```yaml
kind: PersistentVolume
apiVersion: v1
metadata:
  name: kube-system-kube-registry-pv
  labels:
    kubernetes.io/cluster-service: "true"
spec:
{% if pillar.get('cluster_registry_disk_type', '') == 'gce' %}
  capacity:
    storage: {{ pillar['cluster_registry_disk_size'] }}
  accessModes:
    - ReadWriteOnce
  gcePersistentDisk:
    pdName: "{{ pillar['cluster_registry_disk_name'] }}"
    fsType: "ext4"
{% endif %}
```
<!-- END MUNGE: EXAMPLE registry-pv.yaml.in -->

If, for example, you wanted to use NFS you would just need to change the
`gcePersistentDisk` block to `nfs`. See
[here](../../../docs/user-guide/volumes.md) for more details on volumes.

Note that in any case, the storage (in the case the GCE PersistentDisk) must be
created independently - this is not something Kubernetes manages for you (yet).

### I don't want or don't have persistent storage

If you are running in a place that doesn't have networked storage, or if you
just want to kick the tires on this without committing to it, you can easily
adapt the `ReplicationController` specification below to use a simple
`emptyDir` volume instead of a `persistentVolumeClaim`.

## Claim the storage

Now that the Kubernetes cluster knows that some storage exists, you can put a
claim on that storage.  As with the `PersistentVolume` above, you can start
with the `salt` template:

<!-- BEGIN MUNGE: EXAMPLE registry-pvc.yaml.in -->
```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: kube-registry-pvc
  namespace: kube-system
  labels:
    kubernetes.io/cluster-service: "true"
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {{ pillar['cluster_registry_disk_size'] }}
```
<!-- END MUNGE: EXAMPLE registry-pvc.yaml.in -->

This tells Kubernetes that you want to use storage, and the `PersistentVolume`
you created before will be bound to this claim (unless you have other
`PersistentVolumes` in which case those might get bound instead).  This claim
gives you the right to use this storage until you release the claim.

## Run the registry

Now we can run a Docker registry:

<!-- BEGIN MUNGE: EXAMPLE registry-rc.yaml -->
```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: kube-registry-v0
  namespace: kube-system
  labels:
    k8s-app: kube-registry
    version: v0
    kubernetes.io/cluster-service: "true"
spec:
  replicas: 1
  selector:
    k8s-app: kube-registry
    version: v0
  template:
    metadata:
      labels:
        k8s-app: kube-registry
        version: v0
        kubernetes.io/cluster-service: "true"
    spec:
      containers:
      - name: registry
        image: registry:2
        resources:
          limits:
            cpu: 100m
            memory: 100Mi
        env:
        - name: REGISTRY_HTTP_ADDR
          value: :5000
        - name: REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY
          value: /var/lib/registry
        volumeMounts:
        - name: image-store
          mountPath: /var/lib/registry
        ports:
        - containerPort: 5000
          name: registry
          protocol: TCP
      volumes:
      - name: image-store
        persistentVolumeClaim:
          claimName: kube-registry-pvc
```
<!-- END MUNGE: EXAMPLE registry-rc.yaml -->

## Expose the registry in the cluster

Now that we have a registry `Pod` running, we can expose it as a Service:

<!-- BEGIN MUNGE: EXAMPLE registry-svc.yaml -->
```yaml
apiVersion: v1
kind: Service
metadata:
  name: kube-registry
  namespace: kube-system
  labels:
    k8s-app: kube-registry
    kubernetes.io/cluster-service: "true"
    kubernetes.io/name: "KubeRegistry"
spec:
  selector:
    k8s-app: kube-registry
  ports:
  - name: registry
    port: 5000
    protocol: TCP
```
<!-- END MUNGE: EXAMPLE registry-svc.yaml -->

## Expose the registry on each node

Now that we have a running `Service`, we need to expose it onto each Kubernetes
`Node` so that Docker will see it as `localhost`.  We can load a `Pod` on every
node by dropping a YAML file into the kubelet config directory
(/etc/kubernetes/manifests by default).

<!-- BEGIN MUNGE: EXAMPLE ../../saltbase/salt/kube-registry-proxy/kube-registry-proxy.yaml -->
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kube-registry-proxy
  namespace: kube-system
spec:
  containers:
  - name: kube-registry-proxy
    image: gcr.io/google_containers/kube-registry-proxy:0.3
    resources:
      limits:
        cpu: 100m
        memory: 50Mi
    env:
    - name: REGISTRY_HOST
      value: kube-registry.kube-system.svc.cluster.local
    - name: REGISTRY_PORT
      value: "5000"
    - name: FORWARD_PORT
      value: "5000"
    ports:
    - name: registry
      containerPort: 5000
      hostPort: 5000
```
<!-- END MUNGE: EXAMPLE ../../saltbase/salt/kube-registry-proxy/kube-registry-proxy.yaml -->

This ensures that port 5000 on each node is directed to the registry `Service`.
You should be able to verify that it is running by hitting port 5000 with a web
browser and getting a 404 error:

```console
$ curl localhost:5000
404 page not found
```

## Using the registry

To use an image hosted by this registry, simply say this in your `Pod`'s
`spec.containers[].image` field:

```yaml
    image: localhost:5000/user/container
```

Before you can use the registry, you have to be able to get images into it,
though.  If you are building an image on your Kubernetes `Node`, you can spell
out `localhost:5000` when you build and push.  More likely, though, you are
building locally and want to push to your cluster.

You can use `kubectl` to set up a port-forward from your local node to a
running Pod:

```console
$ POD=$(kubectl get pods --namespace kube-system -l k8s-app=kube-registry \
            -o template --template '{{range .items}}{{.metadata.name}} {{.status.phase}}{{"\n"}}{{end}}' \
            | grep Running | head -1 | cut -f1 -d' ')

$ kubectl port-forward --namespace kube-system $POD 5000:5000 &
```

Now you can build and push images on your local computer as
`localhost:5000/yourname/container` and those images will be available inside
your kubernetes cluster with the same name.

# More Extensions

- [Use GCS as storage backend](gcs/README.md)
- [Enable TLS/SSL](tls/README.md)
- [Enable Authentication](auth/README.md)

## Future improvements

* Allow port-forwarding to a Service rather than a pod (#15180)


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/registry/README.md?pixel)]()
