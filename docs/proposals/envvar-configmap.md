# ConfigMaps as environment variables

## Abstract

A proposal for populating environment variables of a container from an entire ConfigMap.

## Proposed Design

Containers can specify the ConfigMaps that are consumed as environment variables.
Each key defined in the ConfigMap's `Data` object must be a "C" identifier. If an invalid key is present, the container will fail to start.

Environment variables are currently defined by servies or the `Env` object in a container, where the `Env` takes precedence.
The introduction of ConfigMaps adds a third possibility. To prevent any change in behavior, the `Env` object will still override any environment variable introduced by a ConfigMap. A ConfigMap is allowed to override variables defined by services.
Variable references defined by an `EnvVar` struct can be resolved by values defined in other `EnvVar`s or those introduced by ConfigMaps and services.

To prevent collisions amongst multiple ConfigMaps, each defined ConfigMap can have an associated prefix that is appended to each key in that ConfigMap. This also makes it obvious to the user where a given environment variable definition came from.

### Kubectl updates

The describe command will display the configmap name that have been defined as part of the environment variable section.

### API Resource

A new `EnvFromSource` type containing a `ConfigMapRef` will be added to the `Container` struct.

```go
// EnvFromSource represents the source of a set of ConfigMaps
type EnvFromSource struct {
  // A string to place in front of every key
  Prefix string
	// The ConfigMap to select from
	ConfigMapRef *LocalObjectReference `json:"configMapRef,omitempty"`
}


type Container struct {
  // List of sources to populate environment variables in the container.
  // The keys defined within a source must be a "C" identifier. An invalid key
  // will prevent the container from starting.
  // All env values will take precedence over any listed source.
  // Cannot be updated.
  // +optional
  EnvFrom []EnvFromSource `json:"envFrom,omitempty"`
}

```

### Examples

### Consuming `ConfigMap` as Environment Variables

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: etcd-env-config
data:
  number_of_members: "1"
  initial_cluster_state: new
  initial_cluster_token: DUMMY_ETCD_INITIAL_CLUSTER_TOKEN
  discovery_token: DUMMY_ETCD_DISCOVERY_TOKEN
  discovery_url: http://etcd_discovery:2379
  etcdctl_peers: http://etcd:2379
  duplicate_key: FROM_CONFIG_MAP
  REPLACE_ME: "a value"
```

This pod consumes the entire `ConfigMap` as environment variables:

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
    - Name: duplicate_key
      Value: FROM_ENV
    - Name: expansion
      Value: ${REPLACE_ME}
    envFrom:
    - configMapRef:
        name: etcd-env-config
```

The resulting environment variables will be:

```
number_of_members="1"
initial_cluster_state="new"
initial_cluster_token="DUMMY_ETCD_INITIAL_CLUSTER_TOKEN"
discovery_token="DUMMY_ETCD_DISCOVERY_TOKEN"
discovery_url="http://etcd_discovery:2379"
etcdctl_peers="http://etcd:2379"
duplicate_key="FROM_ENV"
expansion="a value"
REPLACE_ME="a value"
```

### Consuming multiple `ConfigMap` as Environment Variables

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: env-config
data:
  key1: a
  key2: b
```

This pod consumes the entire `ConfigMap` as environment variables:

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
    envFrom:
    - prefix: cm1_
      configMapRef:
        name: env-config
    - prefix: cm2_
      configMapRef:
        name: env-config
```

The resulting environment variables will be:

```
cm1_key1="a"
cm1_key2="b"
cm2_key1="a"
cm2_key2="b"
```

### Future

Add similar support for Secrets.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/envvar-configmap.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
