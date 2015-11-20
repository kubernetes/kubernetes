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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/config_data.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Generic Configuration Object

## Abstract

This proposal proposes a new API resource, `ConfigData`, that stores data used for the configuration
of applications deployed on `Kubernetes`.

The main focus points of this proposal are:

* Dynamic distribution of configuration data to deployed applications.
* Encapsulate configuration information and simplify `Kubernetes` deployments.
* Create a flexible configuration model for `Kubernetes`.

## Motivation

A `Secret`-like API resource is needed to store configuration data that pods can consume.

Goals of this design:

1.  Describe a `ConfigData` API resource
2.  Describe the semantics of consuming `ConfigData` as environment variables
3.  Describe the semantics of consuming `ConfigData` as files in a volume

## Use Cases

1. As a user, I want to be able to consume configuration data as environment variables
2. As a user, I want to be able to consume configuration data as files in a volume
3. As a user, I want my view of configuration data in files to be eventually consistent with changes
   to the data

### Consuming `ConfigData` as Environment Variables

Many programs read their configuration from environment variables.  `ConfigData` should be possible
to consume in environment variables.  The rough series of events for consuming `ConfigData` this way
is:

1. A `ConfigData` object is created
2. A pod that consumes the configuration data via environment variables is created
3. The pod is scheduled onto a node
4. The kubelet retrieves the `ConfigData` resource(s) referenced by the pod and starts the container
   processes with the appropriate data in environment variables

### Consuming `ConfigData` in Volumes

Many programs read their configuration from configuration files.  `ConfigData` should be possible
to consume in a volume.  The rough series of events for consuming `ConfigData` this way
is:

1. A `ConfigData` object is created
2. A new pod using the `ConfigData` via the volume plugin is created
3. The pod is scheduled onto a node
4. The Kubelet creates an instance of the volume plugin and calls its `Setup()` method
5. The volume plugin retrieves the `ConfigData` resource(s) referenced by the pod and projects
   the appropriate data into the volume

### Consuming `ConfigData`  Updates

Any long-running system has configuration that is mutated over time.  Changes made to configuration
data must be made visible to pods consuming data in volumes so that they can respond to those
changes.

The `resourceVersion` of the `ConfigData` object will be updated by the API server every time the
object is modified.  After an update, modifications will be made visible to the consumer container:

1. A `ConfigData` object is created
2. A new pod using the `ConfigData` via the volume plugin is created
3. The pod is scheduled onto a node
4. During the sync loop, the Kubelet creates an instance of the volume plugin and calls its
   `Setup()` method
5. The volume plugin retrieves the `ConfigData` resource(s) referenced by the pod and projects
   the appropriate data into the volume
6. The `ConfigData` referenced by the pod is updated
7. During the next iteration of the `syncLoop`, the Kubelet creates an instance of the volume plugin
   and calls its `Setup()` method
8. The volume plugin projects the updated data into the volume atomically

It is the consuming pod's responsibility to make use of the updated data once it is made visible.

Because environment variables cannot be updated without restarting a container, configuration data
consumed in environment variables will not be updated.

### Advantages

* Easy to consume in pods; consumer-agnostic
* Configuration data is persistent and versioned
* Consumers of configuration data in volumes can respond to changes in the data

## Proposed Design

### API Resource

The `ConfigData` resource will be added to the `extensions` API Group:

```go
package api

// ConfigData holds configuration data for pods to consume.
type ConfigData struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

  // Data contains the configuration data.  Each key must be a valid DNS_SUBDOMAIN or leading
  // dot followed by valid DNS_SUBDOMAIN.
	Data map[string]string `json:"data,omitempty"`
}

type ConfigDataList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []ConfigData `json:"items"`
}
```

A `Registry` implementation for `ConfigData` will be added to `pkg/registry/configdata`.

### Environment Variables

The `EnvVarSource` will be extended with a new selector for config data:

```go
package api

// EnvVarSource represents a source for the value of an EnvVar.
type EnvVarSource struct {
  // other fields omitted

  // Specifies a ConfigData key
  ConfigData *ConfigDataSelector `json:"configData,omitempty"`
}

// ConfigDataSelector selects a key of a ConfigData.
type ConfigDataSelector struct {
  // The name of the ConfigData to select a key from.
  ConfigDataName string `json:"configDataName"`
  // The key of the ConfigData to select.
  Key string `json:"key"`
}
```

### Volume Source

A new `ConfigDataVolumeSource` type of volume source containing the `ConfigData` object will be
added to the `VolumeSource` struct in the API:

```go
package api

type VolumeSource struct {
  // other fields omitted
  ConfigData *ConfigDataVolumeSource `json:"configData,omitempty"`
}

// ConfigDataVolumeSource represents a volume that holds configuration data
type ConfigDataVolumeSource struct {
  // A list of config data keys to project into the volume in files
  Files []ConfigDataVolumeFile `json:"files"`
}

// ConfigDataVolumeFile represents a single file containing config data
type ConfigDataVolumeFile struct {
  ConfigDataSelector `json:",inline"`

  // The relative path name of the file to be created.
  // Must not be absolute or contain the '..' path. Must be utf-8 encoded.
  // The first item of the relative path must not start with '..'
  Path string `json:"path"`
}
```

**Note:** The update logic used in the downward API volume plug-in will be extracted and re-used in
the volume plug-in for `ConfigData`.

## Examples

#### Consuming `ConfigData` as Environment Variables

```yaml
apiVersion: extensions/v1beta1
kind: ConfigData
metadata:
  name: etcd-env-config
data:
  number_of_members: 1
  initial_cluster_state: new
  initial_cluster_token: DUMMY_ETCD_INITIAL_CLUSTER_TOKEN
  discovery_token: DUMMY_ETCD_DISCOVERY_TOKEN
  discovery_url: http://etcd-discovery:2379
  etcdctl_peers: http://etcd:2379
```

This pod consumes the `ConfigData` as environment variables:

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
        configData:
          configDataName: etcd-env-config
          key: number_of_members
    - name: ETCD_INITIAL_CLUSTER_STATE
      valueFrom:
        configData:
          configDataName: etcd-env-config
          key: initial_cluster_state
    - name: ETCD_DISCOVERY_TOKEN
      valueFrom:
        configData:
          configDataName: etcd-env-config
          key: discovery_token
    - name: ETCD_DISCOVERY_URL
      valueFrom:
        configData:
          configDataName: etcd-env-config
          key: discovery_url
    - name: ETCDCTL_PEERS
      valueFrom:
        configData:
          configDataName: etcd-env-config
          key: etcdctl_peers
```

### Consuming `ConfigData` as Volumes

`redis-volume-config` is intended to be used as a volume containing a config file:

```yaml
apiVersion: extensions/v1beta1
kind: ConfigData
metadata:
  name: redis-volume-config
data:
  redis.conf: "pidfile /var/run/redis.pid\nport6379\ntcp-backlog 511\n databases 1\ntimeout 0\n"
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
      command: "redis-server /mnt/config-data/etc/redis.conf"
      ports:
        - containerPort: 6379
      volumeMounts:
        - name: config-data-volume
          mountPath: /mnt/config-data
  volumes:
  - name: config-data-volume
    configData:
      files:
        - path: "etc/redis.conf"
          configDataName: redis-volume-config
          key: redis.conf
```

### Future Improvements

In the future, we may add the ability to specify an init-container that can watch the volume
contents for updates and respond to changes when they occur.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/config_data.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
