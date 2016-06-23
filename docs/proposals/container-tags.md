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

# Container Tags

## Abstract

We propose to extend `PodSpec` and `Container` with a way for the
client to specify metadata tags to appear on the containers of the
pod.  This supports use cases in which Kubernetes is not the only
system involved; in such situations there are reasons why the user may
wish to put some metadata on the containers.  Examples include tagging
to identify the tenant in a multi-tenant scenario (at least, when a
tenant can have more than one k8s namespace), and supporting tooling
and practices that operate on containers.

The multi-tenant scenario has been discussed with the networking SIG,
and this model change has been briefly discussed in the Node SIG.

## Model changes

There are two model changes.  The `PodSpec` will be extended with a
field holding a set of tags that is applied to every container of the
pod.  The `Container` is extended with a field holding a set of tags
that apply to just that container.
>NOTE: Initially the model changes will be implemented as annotations.

```go
// ContainerTags is a set of metadata tags that is applied to
// every container of the pod.  The exact tagging mechanism is
// specific to the container runtime being used.  The user should use
// keys and values that the container runtime can support.
type map[string]string ContainerTags

// PodSpec is a description of a pod
type PodSpec struct {
	...

	// Tags that apply to every container of the pod.
	ContainersTags ContainerTags `json:"containersTags,omitempty"` 
}

...

// Container represents a single container that is expected to be run on the host.
type Container struct {
	...

	// Tags that apply to this container.
	ContainerTags ContainerTags `json:"containerTags,omitempty"` 
}
```

## Prototyping via annotations

The above extensions will be represented at first as a json encoded annotations like so:


```yaml
kind: Pod
apiVersion: v1
metadata:
  annotations:
    containers-annotations.alpha.kubernetes.io: |
      {
        "foo": "bar",
        "baz": "oof"
      }
    container-annotations.alpha.kubernetes.io: |
      {
        "containerX": {
          "tag11": "val11",
          "tag12": "val12",
        },
        "containerY": {
          "tag21": "val21"
        }
      }
...
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/container-tags.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
