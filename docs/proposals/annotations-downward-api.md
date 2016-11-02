# Exposing annotations via environment downward API

Author: Michal Rostecki \<mrostecki@mirantis.com\>

## Introduction

Annotations of the pod can be taken through the Kubernetes API, but currently
there is no way to pass them to the application inside the container. This means
that annotations can be used by the core Kubernetes services and the user outside
of the Kubernetes cluster.

Of course using Kubernetes API from the application running inside the container
managed by Kubernetes is technically possible, but that's an idea which denies
the principles of microservices architecture.

The purpose of the proposal is to allow to pass the annotation as the environment
variable to the container.

### Use-case

The primary usecase for this proposal are StatefulSets. There is an idea to expose
StatefulSet index to the applications running inside the pods managed by StatefulSet.
Since StatefulSet creates pods as the API objects, passing this index as an
annotation seems to be a valid way to do this. However, to finally pass this
information to the containerized application, we need to pass this annotation.
That's why the downward API for annotations is needed here.

## API

The exact `fieldPath` to the annotation will look like:

```
metadata.annotations['annotationKey']
```

So, assuming that we would want to pass the `pod.beta.kubernetes.io/petset-index`
annotation as a `PETSET_INDEX` variable, the environment variable definition
will look like:

```
env:
  - name: PETSET_INDEX
    valueFrom:
      fieldRef:
        fieldPath: metadata.annotations['pod.beta.kubernetes.io/petset-index']
```

## Implementation

In general, this environment downward API part will be implemented in the same
place as the other metadata - as a label conversion function.

The most tricky part of implementation will be proper validation of the `fieldPath`.
We shouldn't allow any of these characters `\`, `'`, `[`, `]` to be a part of
`annotationKey` without escaping.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/annotations-downward-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
