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
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/downward-api.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

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

More information will be exposed through this same API over time.

## Exposing pod information into a container

Containers consume information from the downward API using environment
variables.  In the future, containers will also be able to consume the downward
API via a volume plugin.

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
  restartPolicy: Never
```

[Download example](downward-api/dapi-pod.yaml)
<!-- END MUNGE: EXAMPLE downward-api/dapi-pod.yaml -->

Some more thorough examples:
   * [environment variables](environment-guide/)
   * [downward API](downward-api/)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/downward-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
