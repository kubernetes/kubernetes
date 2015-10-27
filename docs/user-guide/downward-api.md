<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Downward API

It is sometimes useful for a container to have information about itself, but we
want to be careful not to over-couple containers to Kubernetes. The downward
API allows containers to consume information about themselves or the system and
expose that information how they want it, without necessarily coupling to the
Kubernetes client or REST API.

An example of this is a "legacy" app that is already written assuming
that a particular environment variable will hold a unique identifier.  While it
is often possible to "wrap" such applications, this is tedious and error prone,
and violates the goal of low coupling.  Instead, the user should be able to use
the Pod's name, for example, and inject it into this well-known variable.

## Capabilities

The following information is available to a `Pod` through the downward API:

*   The pod's name
*   The pod's namespace
*   The pod's IP

More information will be exposed through this same API over time.

## Exposing pod information into a container

Containers consume information from the downward API using environment
variables or using a volume plugin.

### Environment variables

Most environment variables in the Kubernetes API use the `value` field to carry
simple values.  However, the alternate `valueFrom` field allows you to specify
a `fieldRef` to select fields from the pod's definition.  The `fieldRef` field
is a structure that has an `apiVersion` field and a `fieldPath` field.  The
`fieldPath` field is an expression designating a field of the pod.  The
`apiVersion` field is the version of the API schema that the `fieldPath` is
written in terms of.  If the `apiVersion` field is not specified it is
defaulted to the API version of the enclosing object.

The `fieldRef` is evaluated and the resulting value is used as the value for
the environment variable.  This allows users to publish their pod's name in any
environment variable they want.

## Example

This is an example of a pod that consumes its name and namespace via the
downward API:

<!-- BEGIN MUNGE: EXAMPLE downward-api/dapi-pod.yaml -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh", "-c", "env" ]
      env:
        - name: MY_POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: MY_POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: MY_POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
  restartPolicy: Never
```

[Download example](downward-api/dapi-pod.yaml?raw=true)
<!-- END MUNGE: EXAMPLE downward-api/dapi-pod.yaml -->



### Downward API volume

Using a similar syntax it's possible to expose pod information to containers using plain text files.
Downward API are dumped to a mounted volume. This is achieved using a `downwardAPI`
volume type and the different items represent the files to be created. `fieldPath` references the field to be exposed.

Downward API volume permits to store more complex data like [`metadata.labels`](labels.md) and [`metadata.annotations`](annotations.md). Currently key/value pair set fields are saved using `key="value"` format:

```
key1="value1"
key2="value2"
```

In future, it will be possible to specify an output format option.

Downward API volumes can expose:

*   The pod's name
*   The pod's namespace
*   The pod's labels
*   The pod's annotations

The downward API volume refreshes its data in step with the kubelet refresh loop. When labels will be modifiable on the fly without respawning the pod containers will be able to detect changes through mechanisms such as [inotify](https://en.wikipedia.org/wiki/Inotify).

In future, it will be possible to specify a specific annotation or label.

## Example

This is an example of a pod that consumes its labels and annotations via the downward API volume, labels and annotations are dumped in `/etc/podlabels` and in `/etc/annotations`, respectively:

<!-- BEGIN MUNGE: EXAMPLE downward-api/volume/dapi-volume.yaml -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kubernetes-downwardapi-volume-example
  labels:
    zone: us-est-coast
    cluster: test-cluster1
    rack: rack-22
  annotations:
    build: two
    builder: john-doe
spec:
  containers:
    - name: client-container
      image: gcr.io/google_containers/busybox
      command: ["sh", "-c", "while true; do if [[ -e /etc/labels ]]; then cat /etc/labels; fi; if [[ -e /etc/annotations ]]; then cat /etc/annotations; fi; sleep 5; done"]
      volumeMounts:
        - name: podinfo
          mountPath: /etc
          readOnly: false
  volumes:
    - name: podinfo
      downwardAPI:
        items:
          - path: "labels"
            fieldRef:
              fieldPath: metadata.labels
          - path: "annotations"
            fieldRef:
              fieldPath: metadata.annotations
```

[Download example](downward-api/volume/dapi-volume.yaml?raw=true)
<!-- END MUNGE: EXAMPLE downward-api/volume/dapi-volume.yaml -->

Some more thorough examples:
   * [environment variables](environment-guide/)
   * [downward API](downward-api/)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/downward-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
