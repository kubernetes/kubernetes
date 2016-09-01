<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Downward API for resource limits and requests

## Background

Currently the downward API (via environment variables and volume plugin) only
supports exposing a Pod's name, namespace, annotations, labels and its IP
([see details](http://kubernetes.io/docs/user-guide/downward-api/)). This
document explains the need and design to extend them to expose resources
(e.g. cpu, memory) limits and requests.

## Motivation

Software applications require configuration to work optimally with the resources they're allowed to use.
Exposing the requested and limited amounts of available resources inside containers will allow
these applications to be configured more easily. Although docker already
exposes some of this information inside containers, the downward API helps
exposing this information in a runtime-agnostic manner in Kubernetes.

## Use cases

As an application author, I want to be able to use cpu or memory requests and
limits to configure the operational requirements of my applications inside containers.
For example, Java applications expect to be made aware of the available heap size via
a command line argument to the JVM, for example: java -Xmx:`<heap-size>`. Similarly, an
application may want to configure its thread pool based on available cpu resources and
the exported value of GOMAXPROCS.

## Design

This is mostly driven by the discussion in [this issue](https://github.com/kubernetes/kubernetes/issues/9473).
There are three approaches discussed in this document to obtain resources limits
and requests to be exposed as environment variables and volumes inside
containers:

1. The first approach requires users to specify full json path selectors
in which selectors are relative to the pod spec. The benefit of this
approach is to specify pod-level resources, and since containers are
also part of a pod spec, it can be used to specify container-level
resources too.

2. The second approach requires specifying partial json path selectors
which are relative to the container spec. This approach helps
in retrieving a container specific resource limits and requests, and at
the same time, it is simpler to specify than full json path selectors.

3. In the third approach, users specify fixed strings (magic keys) to retrieve
resources limits and requests and do not specify any json path
selectors. This approach is similar to the existing downward API
implementation approach. The advantages of this approach are that it is
simpler to specify that the first two, and does not require any type of
conversion between internal and versioned objects or json selectors as
discussed below.

Before discussing a bit more about merits of each approach, here is a
brief discussion about json path selectors and some implications related
to their use.

#### JSONpath selectors

Versioned objects in kubernetes have json tags as part of their golang fields.
Currently, objects in the internal API have json tags, but it is planned that
these will eventually be removed (see [3933](https://github.com/kubernetes/kubernetes/issues/3933)
for discussion). So for discussion in this proposal, we assume that
internal objects do not have json tags. In the first two approaches
(full and partial json selectors), when a user creates a pod and its
containers, the user specifies a json path selector in the pod's
spec to retrieve values of its limits and requests. The selector
is composed of json tags similar to json paths used with kubectl
([json](http://kubernetes.io/docs/user-guide/jsonpath/)). This proposal
uses kubernetes' json path library to process the selectors to retrieve
the values. As kubelet operates on internal objects (without json tags),
and the selectors are part of versioned objects, retrieving values of
the limits and requests can be handled using these two solutions:

1. By converting an internal object to versioned object, and then using
the json path library to retrieve the values from the versioned object
by processing the selector.

2. By converting a json selector of the versioned objects to internal
object's golang expression and then using the json path library to
retrieve the values from the internal object by processing the golang
expression. However, converting a json selector of the versioned objects
to internal object's golang expression will still require an instance
of the versioned object, so it seems more work from the first solution
unless there is another way without requiring the versioned object.

So there is a one time conversion cost associated with the first (full
path) and second (partial path) approaches, whereas the third approach
(magic keys) does not require any such conversion and can directly
work on internal objects. If we want to avoid conversion cost and to
have implementation simplicity, my opinion is that magic keys approach
is relatively easiest to implement to expose limits and requests with
least impact on existing functionality.

To summarize merits/demerits of each approach:

|Approach | Scope | Conversion cost | JSON selectors | Future extension|
| ---------- | ------------------- | -------------------| ------------------- | ------------------- |
|Full selectors | Pod/Container | Yes | Yes | Possible |
|Partial selectors | Container | Yes | Yes | Possible |
|Magic keys | Container | No | No | Possible|

Note: Please note that pod resources can always be accessed using existing `type ObjectFieldSelector` object
in conjunction with partial selectors and magic keys approaches.

### API with full JSONpath selectors

Full json path selectors specify the complete path to the resources
limits and requests relative to pod spec.

#### Environment variables

This table shows how selectors can be used for various requests and
limits to be exposed as environment variables. Environment variable names
are examples only and not necessarily as specified, and the selectors do not
have to start with dot.

| Env Var Name | Selector |
| ---- | ------------------- |
| CPU_LIMIT | spec.containers[?(@.name=="container-name")].resources.limits.cpu|
| MEMORY_LIMIT | spec.containers[?(@.name=="container-name")].resources.limits.memory|
| CPU_REQUEST | spec.containers[?(@.name=="container-name")].resources.requests.cpu|
| MEMORY_REQUEST | spec.containers[?(@.name=="container-name")].resources.requests.memory |

#### Volume plugin

This table shows how selectors can be used for various requests and
limits to be exposed as volumes. The path names are examples only and
not necessarily as specified, and the selectors do not have to start with dot.


| Path | Selector |
| ---- | ------------------- |
| cpu_limit | spec.containers[?(@.name=="container-name")].resources.limits.cpu|
| memory_limit| spec.containers[?(@.name=="container-name")].resources.limits.memory|
| cpu_request | spec.containers[?(@.name=="container-name")].resources.requests.cpu|
| memory_request |spec.containers[?(@.name=="container-name")].resources.requests.memory|

Volumes are pod scoped, so a selector must be specified with a container name.

Full json path selectors will use existing `type ObjectFieldSelector`
to extend the current implementation for resources requests and limits.

```
// ObjectFieldSelector selects an APIVersioned field of an object.
type ObjectFieldSelector struct {
     APIVersion string `json:"apiVersion"`
     // Required: Path of the field to select in the specified API version
     FieldPath string `json:"fieldPath"`
}
```

#### Examples

These examples show how to use full selectors with environment variables and volume plugin.

```
apiVersion: v1
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh","-c", "env" ]
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
      env:
        - name: CPU_LIMIT
          valueFrom:
            fieldRef:
              fieldPath: spec.containers[?(@.name=="test-container")].resources.limits.cpu
```

```
apiVersion: v1
kind: Pod
metadata:
  name: kubernetes-downwardapi-volume-example
spec:
  containers:
    - name: client-container
      image: gcr.io/google_containers/busybox
      command: ["sh", "-c", "while true; do if [[ -e /etc/labels ]]; then cat /etc/labels; fi; if [[ -e /etc/annotations ]]; then cat /etc/annotations; fi;sleep 5; done"]
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
      volumeMounts:
        - name: podinfo
          mountPath: /etc
          readOnly: false
  volumes:
    - name: podinfo
      downwardAPI:
        items:
          - path: "cpu_limit"
            fieldRef:
              fieldPath: spec.containers[?(@.name=="client-container")].resources.limits.cpu
```

#### Validations

For APIs with full json path selectors, verify that selectors are
valid relative to pod spec.


### API with partial JSONpath selectors

Partial json path selectors specify paths to resources limits and requests
relative to the container spec. These will be implemented by introducing a
`ContainerSpecFieldSelector` (json: `containerSpecFieldRef`) to extend the current
implementation for `type DownwardAPIVolumeFile struct` and `type EnvVarSource struct`.

```
// ContainerSpecFieldSelector selects an APIVersioned field of an object.
type ContainerSpecFieldSelector struct {
     APIVersion string `json:"apiVersion"`
     // Container name
     ContainerName string `json:"containerName,omitempty"`
     // Required: Path of the field to select in the specified API version
     FieldPath string `json:"fieldPath"`
}

// Represents a single file containing information from the downward API
type DownwardAPIVolumeFile struct {
     // Required: Path is  the relative path name of the file to be created.
     Path string `json:"path"`
     // Selects a field of the pod: only annotations, labels, name and
     // namespace are supported.
     FieldRef *ObjectFieldSelector `json:"fieldRef, omitempty"`
     // Selects a field of the container: only resources limits and requests
     // (resources.limits.cpu, resources.limits.memory, resources.requests.cpu,
     // resources.requests.memory) are currently supported.
     ContainerSpecFieldRef *ContainerSpecFieldSelector `json:"containerSpecFieldRef,omitempty"`
}

// EnvVarSource represents a source for the value of an EnvVar.
// Only one of its fields may be set.
type EnvVarSource struct {
     // Selects a field of the container: only resources limits and requests
     // (resources.limits.cpu, resources.limits.memory, resources.requests.cpu,
     // resources.requests.memory) are currently supported.
     ContainerSpecFieldRef *ContainerSpecFieldSelector `json:"containerSpecFieldRef,omitempty"`
     // Selects a field of the pod; only name and namespace are supported.
     FieldRef *ObjectFieldSelector `json:"fieldRef,omitempty"`
     // Selects a key of a ConfigMap.
     ConfigMapKeyRef *ConfigMapKeySelector `json:"configMapKeyRef,omitempty"`
     // Selects a key of a secret in the pod's namespace.
     SecretKeyRef *SecretKeySelector `json:"secretKeyRef,omitempty"`
}
```

#### Environment variables

This table shows how partial selectors can be used for various requests and
limits to be exposed as environment variables. Environment variable names
are examples only and not necessarily as specified, and the selectors do not
have to start with dot.

| Env Var Name | Selector |
| -------------------- | -------------------|
| CPU_LIMIT | resources.limits.cpu |
| MEMORY_LIMIT | resources.limits.memory |
| CPU_REQUEST | resources.requests.cpu |
| MEMORY_REQUEST | resources.requests.memory |

Since environment variables are container scoped, it is optional
to specify container name as part of the partial selectors as they are
relative to container spec. If container name is not specified, then
it defaults to current container. However, container name could be specified
to expose variables from other containers.

#### Volume plugin

This table shows volume paths and partial selectors used for resources cpu and memory.
Volume path names are examples only and not necessarily as specified, and the
selectors do not have to start with dot.

| Path | Selector |
| -------------------- | -------------------|
| cpu_limit | resources.limits.cpu |
| memory_limit | resources.limits.memory |
| cpu_request | resources.requests.cpu |
| memory_request | resources.requests.memory |

Volumes are pod scoped, the container name must be specified as part of
`containerSpecFieldRef` with them.

#### Examples

These examples show how to use partial selectors with environment variables and volume plugin.

```
apiVersion: v1
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh","-c", "env" ]
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
      env:
        - name: CPU_LIMIT
          valueFrom:
            containerSpecFieldRef:
              fieldPath: resources.limits.cpu
```

```
apiVersion: v1
kind: Pod
metadata:
  name: kubernetes-downwardapi-volume-example
spec:
  containers:
    - name: client-container
      image: gcr.io/google_containers/busybox
      command: ["sh", "-c", "while true; do if [[ -e /etc/labels ]]; then cat /etc/labels; fi; if [[ -e /etc/annotations ]]; then cat /etc/annotations; fi; sleep 5; done"]
      resources:
        requests:
          memory: "64Mi"
	  cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
      volumeMounts:
        - name: podinfo
          mountPath: /etc
          readOnly: false
  volumes:
    - name: podinfo
      downwardAPI:
        items:
          - path: "cpu_limit"
            containerSpecFieldRef:
              containerName: "client-container"
              fieldPath: resources.limits.cpu
```

#### Validations

For APIs with partial json path selectors, verify
that selectors are valid relative to container spec.
Also verify that container name is provided with volumes.


### API with magic keys

In this approach, users specify fixed strings (or magic keys) to retrieve resources
limits and requests. This approach is similar to the existing downward
API implementation approach. The fixed string used for resources limits and requests
for cpu and memory are `limits.cpu`, `limits.memory`,
`requests.cpu` and `requests.memory`. Though these strings are same
as json path selectors but are processed as fixed strings. These will be implemented by
introducing a `ResourceFieldSelector` (json: `resourceFieldRef`) to extend the current
implementation for `type DownwardAPIVolumeFile struct` and `type EnvVarSource struct`.

The fields in ResourceFieldSelector are `containerName` to specify the name of a
container, `resource` to specify the type of a resource (cpu or memory), and `divisor`
to specify the output format of values of exposed resources. The default value of divisor
is `1` which means cores for cpu and bytes for memory. For cpu, divisor's valid
values are `1m` (millicores), `1`(cores), and for memory, the valid values in fixed point integer
(decimal) are `1`(bytes), `1k`(kilobytes), `1M`(megabytes), `1G`(gigabytes),
`1T`(terabytes), `1P`(petabytes), `1E`(exabytes), and in their power-of-two equivalents `1Ki(kibibytes)`,
`1Mi`(mebibytes), `1Gi`(gibibytes), `1Ti`(tebibytes), `1Pi`(pebibytes), `1Ei`(exbibytes).
For more information about these resource formats, [see details](resources.md).

Also, the exposed values will be `ceiling` of the actual values in the requestd format in divisor.
For example, if requests.cpu is `250m` (250 millicores) and the divisor by default is `1`, then
exposed value will be `1` core. It is because 250 millicores when converted to cores will be 0.25 and
the ceiling of 0.25 is 1.

```
type ResourceFieldSelector struct {
     // Container name
     ContainerName string `json:"containerName,omitempty"`
     // Required: Resource to select
     Resource string `json:"resource"`
     // Specifies the output format of the exposed resources
     Divisor resource.Quantity `json:"divisor,omitempty"`
}

// Represents a single file containing information from the downward API
type DownwardAPIVolumeFile struct {
     // Required: Path is  the relative path name of the file to be created.
     Path string `json:"path"`
     // Selects a field of the pod: only annotations, labels, name and
     // namespace are supported.
     FieldRef *ObjectFieldSelector `json:"fieldRef, omitempty"`
     // Selects a resource of the container: only resources limits and requests
     // (limits.cpu, limits.memory, requests.cpu and requests.memory) are currently supported.
     ResourceFieldRef *ResourceFieldSelector `json:"resourceFieldRef,omitempty"`
}

// EnvVarSource represents a source for the value of an EnvVar.
// Only one of its fields may be set.
type EnvVarSource struct {
     // Selects a resource of the container: only resources limits and requests
     // (limits.cpu, limits.memory, requests.cpu and requests.memory) are currently supported.
     ResourceFieldRef *ResourceFieldSelector `json:"resourceFieldRef,omitempty"`
     // Selects a field of the pod; only name and namespace are supported.
     FieldRef *ObjectFieldSelector `json:"fieldRef,omitempty"`
     // Selects a key of a ConfigMap.
     ConfigMapKeyRef *ConfigMapKeySelector `json:"configMapKeyRef,omitempty"`
     // Selects a key of a secret in the pod's namespace.
     SecretKeyRef *SecretKeySelector `json:"secretKeyRef,omitempty"`
}
```

#### Environment variables

This table shows environment variable names and strings used for resources cpu and memory.
The variable names are examples only and not necessarily as specified.

| Env Var Name | Resource |
| -------------------- | -------------------|
| CPU_LIMIT | limits.cpu |
| MEMORY_LIMIT | limits.memory |
| CPU_REQUEST | requests.cpu |
| MEMORY_REQUEST | requests.memory |

Since environment variables are container scoped, it is optional
to specify container name as part of the partial selectors as they are
relative to container spec. If container name is not specified, then
it defaults to current container. However, container name could be specified
to expose variables from other containers.

#### Volume plugin

This table shows volume paths and strings used for resources cpu and memory.
Volume path names are examples only and not necessarily as specified.

| Path | Resource |
| -------------------- | -------------------|
| cpu_limit | limits.cpu |
| memory_limit | limits.memory|
| cpu_request | requests.cpu |
| memory_request | requests.memory |

Volumes are pod scoped, the container name must be specified as part of
`resourceFieldRef` with them.

#### Examples

These examples show how to use magic keys approach with environment variables and volume plugin.

```
apiVersion: v1
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh","-c", "env" ]
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
      env:
        - name: CPU_LIMIT
          valueFrom:
            resourceFieldRef:
              resource: limits.cpu
        - name: MEMORY_LIMIT
          valueFrom:
            resourceFieldRef:
              resource: limits.memory
              divisor: "1Mi"
```

In the above example, the exposed values of CPU_LIMIT and MEMORY_LIMIT will be 1 (in cores) and 128 (in Mi), respectively.

```
apiVersion: v1
kind: Pod
metadata:
  name: kubernetes-downwardapi-volume-example
spec:
  containers:
    - name: client-container
      image: gcr.io/google_containers/busybox
      command: ["sh", "-c","while true; do if [[ -e /etc/labels ]]; then cat /etc/labels; fi; if [[ -e /etc/annotations ]]; then cat /etc/annotations; fi; sleep 5; done"]
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
      volumeMounts:
        - name: podinfo
          mountPath: /etc
          readOnly: false
  volumes:
    - name: podinfo
      downwardAPI:
        items:
          - path: "cpu_limit"
            resourceFieldRef:
              containerName: client-container
              resource: limits.cpu
              divisor: "1m"
          - path: "memory_limit"
            resourceFieldRef:
              containerName: client-container
              resource: limits.memory
```

In the above example, the exposed values of CPU_LIMIT and MEMORY_LIMIT will be 500 (in millicores) and 134217728 (in bytes), respectively.


#### Validations

For APIs with magic keys, verify that the resource strings are valid and is one
of `limits.cpu`, `limits.memory`, `requests.cpu` and `requests.memory`.
Also verify that container name is provided with volumes.

## Pod-level and container-level resource access

Pod-level resources (like `metadata.name`, `status.podIP`) will always be accessed with `type ObjectFieldSelector` object in
all approaches. Container-level resources will be accessed by `type ObjectFieldSelector`
with full selector approach; and by `type ContainerSpecFieldRef` and `type ResourceFieldRef`
with partial and magic keys approaches, respectively. The following table
summarizes resource access with these approaches.

| Approach | Pod resources| Container resources |
| -------------------- | -------------------|-------------------|
| Full selectors | `ObjectFieldSelector` | `ObjectFieldSelector`|
| Partial selectors | `ObjectFieldSelector`| `ContainerSpecFieldRef` |
| Magic keys | `ObjectFieldSelector`| `ResourceFieldRef` |

## Output format

The output format for resources limits and requests will be same as
cgroups output format, i.e. cpu in cpu shares (cores multiplied by 1024
and rounded to integer) and memory in bytes. For example, memory request
or limit of `64Mi` in the container spec will be output as `67108864`
bytes, and cpu request or limit of `250m` (millicores) will be output as
`256` of cpu shares.

## Implementation approach

The current implementation of this proposal will focus on the API with magic keys
approach. The main reason for selecting this approach is that it might be
easier to incorporate and extend resource specific functionality.

## Applied example

Here we discuss how to use exposed resource values to set, for example, Java
memory size or GOMAXPROCS for your applications. Lets say, you expose a container's
(running an application like tomcat for example) requested memory as `HEAP_SIZE`
and requested cpu as CPU_LIMIT (or could be GOMAXPROCS directly) environment variable.
One way to set the heap size or cpu for this application would be to wrap the binary
in a shell script, and then export `JAVA_OPTS` (assuming your container image supports it)
and GOMAXPROCS environment variables inside the container image. The spec file for the
application pod could look like:

```
apiVersion: v1
kind: Pod
metadata:
  name: kubernetes-downwardapi-volume-example
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh","-c", "env" ]
      resources:
        requests:
          memory: "64M"
          cpu: "250m"
        limits:
          memory: "128M"
          cpu: "500m"
      env:
        - name: HEAP_SIZE
          valueFrom:
            resourceFieldRef:
              resource: requests.memory
        - name: CPU_LIMIT
          valueFrom:
            resourceFieldRef:
              resource: requests.cpu
```

Note that the value of divisor by default is `1`. Now inside the container,
the HEAP_SIZE (in bytes) and GOMAXPROCS (in cores) could be exported as:

```
export JAVA_OPTS="$JAVA_OPTS -Xmx:$(HEAP_SIZE)"

and

export GOMAXPROCS=$(CPU_LIMIT)"
```



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/downward_api_resources_limits_requests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
