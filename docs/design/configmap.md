<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/design/configmap.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Generic Configuration Object

## Abstract

The `ConfigMap` API resource stores data used for the configuration of
applications deployed on Kubernetes.

The main focus of this resource is to:

* Provide dynamic distribution of configuration data to deployed applications.
* Encapsulate configuration information and simplify `Kubernetes` deployments.
* Create a flexible configuration model for `Kubernetes`.

## Motivation

A `Secret`-like API resource is needed to store configuration data that pods can
consume.

Goals of this design:

1.  Describe a `ConfigMap` API resource.
2.  Describe the semantics of consuming `ConfigMap` as environment variables.
3.  Describe the semantics of consuming `ConfigMap` as files in a volume.

## Use Cases

1. As a user, I want to be able to consume configuration data as environment
variables.
2. As a user, I want to be able to consume configuration data as files in a
volume.
3. As a user, I want my view of configuration data in files to be eventually
consistent with changes to the data.

### Consuming `ConfigMap` as Environment Variables

A series of events for consuming `ConfigMap` as environment variables:

1. Create a `ConfigMap` object.
2. Create a pod to consume the configuration data via environment variables.
3. The pod is scheduled onto a node.
4. The Kubelet retrieves the `ConfigMap` resource(s) referenced by the pod and
starts the container processes with the appropriate configuration data from
environment variables.

### Consuming `ConfigMap` in Volumes

A series of events for consuming `ConfigMap` as configuration files in a volume:

1. Create a `ConfigMap` object.
2. Create a new pod using the `ConfigMap` via a volume plugin.
3. The pod is scheduled onto a node.
4. The Kubelet creates an instance of the volume plugin and calls its `Setup()`
method.
5. The volume plugin retrieves the `ConfigMap` resource(s) referenced by the pod
and projects the appropriate configuration data into the volume.

### Consuming `ConfigMap`  Updates

Any long-running system has configuration that is mutated over time. Changes
made to configuration data must be made visible to pods consuming data in
volumes so that they can respond to those changes.

The `resourceVersion` of the `ConfigMap` object will be updated by the API
server every time the object is modified. After an update, modifications will be
made visible to the consumer container:

1. Create a `ConfigMap` object.
2. Create a new pod using the `ConfigMap` via the volume plugin.
3. The pod is scheduled onto a node.
4. During the sync loop, the Kubelet creates an instance of the volume plugin
and calls its `Setup()` method.
5. The volume plugin retrieves the `ConfigMap` resource(s) referenced by the pod
and projects the appropriate data into the volume.
6. The `ConfigMap` referenced by the pod is updated.
7. During the next iteration of the `syncLoop`, the Kubelet creates an instance
of the volume plugin and calls its `Setup()` method.
8. The volume plugin projects the updated data into the volume atomically.

It is the consuming pod's responsibility to make use of the updated data once it
is made visible.

Because environment variables cannot be updated without restarting a container,
configuration data consumed in environment variables will not be updated.

### Advantages

* Easy to consume in pods; consumer-agnostic
* Configuration data is persistent and versioned
* Consumers of configuration data in volumes can respond to changes in the data

## Proposed Design

### API Resource

The `ConfigMap` resource will be added to the main API:

```go
package api

// ConfigMap holds configuration data for pods to consume.
type ConfigMap struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

  // Data contains the configuration data.  Each key must be a valid
  // DNS_SUBDOMAIN or leading dot followed by valid DNS_SUBDOMAIN.
	Data map[string]string `json:"data,omitempty"`
}

type ConfigMapList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []ConfigMap `json:"items"`
}
```

A `Registry` implementation for `ConfigMap` will be added to
`pkg/registry/configmap`.

### Environment Variables

The `EnvVarSource` will be extended with a new selector for `ConfigMap`:

```go
package api

// EnvVarSource represents a source for the value of an EnvVar.
type EnvVarSource struct {
  // other fields omitted

  // Selects a key of a ConfigMap.
  ConfigMapKeyRef *ConfigMapKeySelector `json:"configMapKeyRef,omitempty"`
}

// Selects a key from a ConfigMap.
type ConfigMapKeySelector struct {
  // The ConfigMap to select from.
  LocalObjectReference `json:",inline"`
  // The key to select.
  Key string `json:"key"`
}
```

### Volume Source

A new `ConfigMapVolumeSource` type of volume source containing the `ConfigMap`
object will be added to the `VolumeSource` struct in the API:

```go
package api

type VolumeSource struct {
  // other fields omitted
  ConfigMap *ConfigMapVolumeSource `json:"configMap,omitempty"`
}

// Represents a volume that holds configuration data.
type ConfigMapVolumeSource struct {
  LocalObjectReference `json:",inline"`
  // A list of keys to project into the volume.
  // If unspecified, each key-value pair in the Data field of the
  // referenced ConfigMap will be projected into the volume as a file whose name
  // is the key and content is the value.
  // If specified, the listed keys will be project into the specified paths, and
  // unlisted keys will not be present.
  Items []KeyToPath `json:"items,omitempty"`
}

// Represents a mapping of a key to a relative path.
type KeyToPath struct {
  // The name of the key to select
  Key string `json:"key"`

  // The relative path name of the file to be created.
  // Must not be absolute or contain the '..' path. Must be utf-8 encoded.
  // The first item of the relative path must not start with '..'
  Path string `json:"path"`
}
```

**Note:** The update logic used in the downward API volume plug-in will be
extracted and re-used in the volume plug-in for `ConfigMap`.

### Changes to Secret

We will update the Secret volume plugin to have a similar API to the new
`ConfigMap` volume plugin. The secret volume plugin will also begin updating
secret content in the volume when secrets change.

## Examples

#### Consuming `ConfigMap` as Environment Variables

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: etcd-env-config
data:
  number-of-members: "1"
  initial-cluster-state: new
  initial-cluster-token: DUMMY_ETCD_INITIAL_CLUSTER_TOKEN
  discovery-token: DUMMY_ETCD_DISCOVERY_TOKEN
  discovery-url: http://etcd-discovery:2379
  etcdctl-peers: http://etcd:2379
```

This pod consumes the `ConfigMap` as environment variables:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: config-env-example
spec:
  containers:
  - name: etcd
    image: openshift/etcd-20-centos7
    ports:
    - containerPort: 2379
      protocol: TCP
    - containerPort: 2380
      protocol: TCP
    env:
    - name: ETCD_NUM_MEMBERS
      valueFrom:
        configMapKeyRef:
          name: etcd-env-config
          key: number-of-members
    - name: ETCD_INITIAL_CLUSTER_STATE
      valueFrom:
        configMapKeyRef:
          name: etcd-env-config
          key: initial-cluster-state
    - name: ETCD_DISCOVERY_TOKEN
      valueFrom:
        configMapKeyRef:
          name: etcd-env-config
          key: discovery-token
    - name: ETCD_DISCOVERY_URL
      valueFrom:
        configMapKeyRef:
          name: etcd-env-config
          key: discovery-url
    - name: ETCDCTL_PEERS
      valueFrom:
        configMapKeyRef:
          name: etcd-env-config
          key: etcdctl-peers
```

#### Consuming `ConfigMap` as Volumes

`redis-volume-config` is intended to be used as a volume containing a config
file:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-volume-config
data:
  redis.conf: "pidfile /var/run/redis.pid\nport 6379\ntcp-backlog 511\ndatabases 1\ntimeout 0\n"
```

The following pod consumes the `redis-volume-config` in a volume:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: config-volume-example
spec:
  containers:
    - name: redis
      image: kubernetes/redis
      command: ["redis-server", "/mnt/config-map/etc/redis.conf"]
      ports:
        - containerPort: 6379
      volumeMounts:
        - name: config-map-volume
          mountPath: /mnt/config-map
  volumes:
    - name: config-map-volume
      configMap:
        name: redis-volume-config
        items:
          - path: "etc/redis.conf"
            key: redis.conf
```

## Future Improvements

In the future, we may add the ability to specify an init-container that can
watch the volume contents for updates and respond to changes when they occur.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/configmap.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
