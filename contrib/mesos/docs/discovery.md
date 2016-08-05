# Discovery

## DNS

### kube-dns

[**kube-dns**](https://github.com/kubernetes/kubernetes/blob/release-1.1/docs/admin/dns.md) is a Kubernetes add-on that works out of the box with Kubernetes-Mesos.
For details on usage see the implementation in the `cluster/mesos/docker` source tree.
kube-dns provides records both for services and pods.

### mesos-dns

**NOTE:** There is still no support for publishing Kubernetes *services* in mesos-dns.

**mesos-dns** communicates with the leading Mesos master to build a DNS record set that reflects the tasks running in a Mesos cluster as documented here: http://mesosphere.github.io/mesos-dns/docs/naming.html.
As of Kubernetes-Mesos [release v0.7.2](https://github.com/mesosphere/kubernetes/releases/tag/v0.7.2-v1.1.5) there is experimental support in the scheduler to populate a task's *discovery-info* field in order to generate alternative/more friendly record names in mesos-dns, for *pods* only.

To enable this feature, set `--mesos-generate-task-discovery=true` when launching the scheduler.

The following discovery-info fields may be set using labels (without a namespace prefix) or else `k8s.mesosphere.io/discovery-XXX` annotations:

* `visibility`: may be `framework`, `external`, or `cluster` (defaults to `cluster`)
* `environment`
* `location`
* `name` (this alters record set generation in *mesos-dns*)
* `version`

In the case where both a label as well as an annotation are supplied the value of the annotation is observed.
The interpretation of value of the `name` label (and `discovery-name` annotation) is a special case: the generated Mesos `discovery-info.name` value will be `${name}.${pod-namespace}.pod`; all other discovery-info values are passed through without modification.

#### Example 1: Use a `name` label on a pod template
```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: frontend
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: guestbook
        tier: frontend
        name: custom-name
    spec:
      containers:
      - name: php-redis
        image: gcr.io/google_samples/gb-frontend:v3
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
        env:
        - name: GET_HOSTS_FROM
          value: dns
        ports:
        - containerPort: 80
```

#### Example 2: Use a `discovery-name` annotation on a pod template
```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: frontend
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: guestbook
        tier: frontend
      annotations:
        k8s.mesosphere.io/discovery-name: custom-name
    spec:
      containers:
      - name: php-redis
        image: gcr.io/google_samples/gb-frontend:v3
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
        env:
        - name: GET_HOSTS_FROM
          value: dns
        ports:
        - containerPort: 80
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/mesos/docs/discovery.md?pixel)]()
