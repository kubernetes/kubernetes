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
[here](http://releases.k8s.io/release-1.1/docs/proposals/image-spec.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

A proposal for adding an `ImageSpec` type to container specs, to facilitate alternative image formats and more
deterministic image references.

## Motivation

Currently, all of the information about where to locate an image or how to verify the image is embedded in a single
string in the container spec. However that opaque string is not structured and sometimes confusing for users if they
are not familiar with how the string is being parsed.

For example, `gcr.io/google_containers/nginx` refers to an image located in `gcr.io` registry, with the repository path
as `google_containers/etcd`; While `library/nginx` refers to an image located in `docker.io` registry, with the repository path
as `library/nginx`, which is same as `nginx` because it is an [official Docker repository](https://docs.docker.com/docker-hub/official_repos/).

Besides, having only one string to represent the image precludes the possibility of adding any other image types
such as [ACI](https://github.com/appc/spec), [Fabric8 Java containers](http://fabric8.io/gitbook/javaContainer.html),
etc.

The goal of this proposal is to add an *optional* `ImageSpec` field in the `Container` object,
which will give users the ability to:

1. Express the image types they want to use.
2. Express how and where they want the images to be fetched from.

## Design

1. Add an optional field called `ImageSpec` in the `Container` object and make `Image` field optional.
2. For GET requests, We will show both the `Image` and `ImageSpec` fields for the sake of backward compatibility.
3. For other requests, either `Image` or `ImageSpec` should be specified but not both. These two fields are
   mutually exclusive. However `ImageSpec` will always be populated internally if `Image` is being used.
4. If `ImageSpec` is not empty, it MUST have one and ONLY one non-empty field to specify the details of the image,
similiar to the `VolumeSource` type.

#### Open Qustions
1. Should we be able to mix different image format in one pod?
2. Can we just use merely `ImageSpec` in the internal API objects now? However this will make the `Image` string
   be filled with registry info when returned to users, e.g. `Image: nginx` would become `Image: docker.io/nginx`.
   If this is unacceptable for now, then we can do it later.
3. Do we need to also add `ImageSpec` in the `ContainerStatus`? Maybe we should just remove the `Image` field in
   `ContainerStatus` in the future as this field cannot vary from the spec, so only `ImageSpec` in the `Container`
   should be enough.

Following is an example of how new internal API objects will look like:

```go
package api

type Container struct {
  // Other fields ommited for brevity.

  // Optional, Image specifies the name of the image, it assumes Docker
  // image format. Either Image or ImageSpec should be specified, but not both.
  Image *string `json:"image",omitempty`

  // Optional, ImageSpec specifies the attributes of the image, including
  // the format, the discovery name, etc.
  // Either Image or ImageSpec should be specified, but not both.
  ImageSpec *ImageSpec `json:"imageSpec,omitempty"`
}

// ImageSpec represents the specification of the image.
// Only one of its members may be specified.
type ImageSpec struct {
  // DockerImage represents the image specification of a Docker image.
  DockerImage *DockerImageSpec `json:"dockerImage,omitempty"`
  // ACI represents the image specification of an appc container image.
  ACI *ACISpec `json:"aci,omitempty"`
  ...
}

// DockerImageSpec represents the image specification of a Docker image.
type DockerImageSpec struct {
  // Required: Name of the registry, e.g. "gcr.io".
  Registry string
  // Required: Name of the image, this can include the repository's name, e.g. "google_containers/etcd".
  Name string
  // Optional: Tag of the image, default to "latest".
  Tag *string `json:"tag,omitempty"`
  // Optional: Digest of the image, default to empty.
  Digest *string `json:"digest,omitempty"`
}

// ACISpec represents the image specification of an appc container image.
type ACISpec struct {
  // Required: Name of the image. e.g. "coreos.com/etcd".
  Name string
  // Optional: Tag of the image, default to "latest".
  Tag *string `json:"tag,omitempty"`
  // Optional: ID of the image, default to empty. The ID must be a string of
  // the format "hash-value", where "hash" is the hash algorithm used and "value"
  // is the hex encoded string of the digest. Currently the only permitted hash
  // algorithm is sha512.
  ID *string `json:"id,omitempty"`
}
```

## Examples

1.  Old client using `pod.Spec.Containers[x].Image`

    An old client creates a pod:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      containers:
      - name: foo
        image: test-repo/foo
      - name: bar
        image: gcr.io/test-repo/bar
    ```
    
    looks to old clients like:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      containers:
      - name: foo
        image: test-repo/foo
      - name: bar
        image: gcr.io/test-repo/bar
    status:
      containerStatuses:
      - containerID: docker://foo
        image: test-repo/foo
        ...
      - containerID: docker://bar
        image: gcr.io/test-repo/bar
        ...
    ```
    
    looks to new clients like:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
     containers:
     - name: foo
       image: test-repo/foo
       imageSpec:
         dockerImage:
           registry: docker.io
           name: test-repo/foo
     - name: bar
       image: gcr.io/test-repo/bar
       imageSpec:
         dockerImage:
           registry: gcr.io
           name: test-repo/bar
    status:
      containerStatuses:
      - containerID: docker://foo
        image: test-repo/foo
        ...
      - containerID: docker://bar
        image: gcr.io/test-repo/bar
        ...
    ```
    *Note that the `ImageSpec` field will be filled according to the `Image` string, and it will assume Docker image
    format.*

2.  New client using `pod.Spec.Containers[x].Image`

    A new client creates a pod:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      containers:
      - name: foo
        image: test-repo/foo
      - name: bar
        image: gcr.io/test-repo/bar
    ```
    
    looks to old clients like:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      containers:
      - name: foo
        image: test-repo/foo
      - name: bar
        image: gcr.io/test-repo/bar
    status:
      containerStatuses:
      - containerID: docker://foo
        image: test-repo/foo
        ...
      - containerID: docker://bar
        image: gcr.io/test-repo/bar
        ...
    ```
    
    looks to new clients like:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
     containers:
     - name: foo
       image: test-repo/foo
       imageSpec:
         dockerImage:
           registry: docker.io
           name: test-repo/foo
     - name: bar
       image: gcr.io/test-repo/bar
       imageSpec:
         dockerImage:
           registry: gcr.io
           name: test-repo/bar
    status:
      containerStatuses:
      - containerID: docker://foo
        image: test-repo/foo
        ...
      - containerID: docker://bar
        image: gcr.io/test-repo/bar
        ...
    ```
    *Same as using old API, the `ImageSpec` field will be filled according to the `Image` string, and it will assume Docker image
    format.*

3.  New client using `pod.Spec.Containers[x].ImageSpec` with Docker image format

    A new client creates a pod:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
    name: test-pod
    spec:
      containers:
      - name: foo
        imageSpec:
          dockerImage:
            registry: docker.io
            name: test-repo/foo
      - name: bar
        imageSpec:
          dockerImage:
            registry: gcr.io
            name: test-repo/bar
    ```
    
    looks to old clients like:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
    name: test-pod
    spec:
      containers:
      - name: foo
        image: docker.io/test-repo/foo
      - name: bar
        image: gcr.io/test-repo/bar
    status:
      containerStatuses:
      - containerID: docker://foo
        image: test-repo/foo
        ...
      - containerID: docker://bar
        image: gcr.io/test-repo/bar
        ...
    ```
    
    looks to new clients like:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
    name: test-pod
    spec:
      containers:
      - name: foo
        image: docker.io/test-repo/foo
        imageSpec:
          dockerImage:
            registry: docker.io
            name: test-repo/foo
      - name: bar
        image: gcr.io/test-repo/bar
        imageSpec:
          dockerImage:
            registry: gcr.io
            name: test-repo/bar
    status:
      containerStatuses:
      - containerID: docker://foo
        image: test-repo/foo
        ...
      - containerID: docker://bar
        image: gcr.io/test-repo/bar
        ...
    ```

    *Note that the `Image` string will be in the form of `[registry/name]`, where `name` contains both the repository
    name and the actual image name.*

3.  New client using `pod.Spec.Containers[x].ImageSpec` with other image format, e.g. ACI.

    A new client creates a pod:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
    name: test-pod
    spec:
      containers:
      - name: etcd
        imageSpec:
          aci:
            name: coreos.com/etcd
            tag: v2.2.0
      - name: flannel
        imageSpec:
          aci:
            name: coreos.com/flannel
    ```
    
    looks to old clients like:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
    name: test-pod
    spec:
      containers:
      - name: etcd
        image: coreos.com/etcd:v2.2.0
      - name: flannel
        image: coreos.com/flannel
    status:
      containerStatuses:
      - containerID: rkt://foo
        image: coreos.com/etcd
        ...
      - containerID: rkt://bar
        image: coreos.com/flannel
        ...
    ```

    *Note that an old client cannot know from this specification what the image format type is.
    This also means that for old servers, the image spec will be translated to a Docker-like image string,
    and if the Docker image does not exist, the pod cannot be created based on this spec. However, users
    SHOULD NOT send such requests to servers using the old API version, as old servers will not recognize
    image formats other than Docker.*

    looks to new clients like:
    
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
    name: test-pod
    spec:
      containers:
      - name: etcd
        image: coreos.com/etcd:v2.2.0
        imageSpec:
          aci:
            name: coreos.com/etcd
            tag: v2.2.0
      - name: flannel
        image: coreos.com/flannel
        imageSpec:
          aci:
            name: coreos.com/flannel
    status:
      containerStatuses:
      - containerID: rkt://foo
        image: coreos.com/etcd
        ...
      - containerID: rkt://bar
        image: coreos.com/flannel
        ...
    ```

## Kubelet/container runtime changes

1. The [generic image puller](https://github.com/kubernetes/kubernetes/blob/v1.2.0-alpha.4/pkg/kubelet/container/image_puller.go#L82)
   should be changed to use `ImageSpec`.

2. Image related functions in the [container runtime interface](https://github.com/kubernetes/kubernetes/blob/v1.2.0-alpha.4/pkg/kubelet/container/runtime.go#L75),
   (including `PullImage`, `IsImagePresent`, `RemoveImage`) should be changed to use `ImageSpec`.
   Besides, these functions should return an error if the container runtime doesn't support the given image format.

## Testing

1. Add test cases to verify the API semantics proposed here, e.g. One and only one image format should be present in
   the `ImageSpec`, etc.

2. Add backward compatibility tests to verify the compatibility by converting the objects between internal API and
   versioned APIs. Examples above or similar examples should be used as the test cases.

3. Add unit tests in kubelet package to verify the changes made in kubelet/container runtime.

4. E2E test cases should be added to verify things such as:
   - `ImageSpec` and `Image` are mutually exclusive, and at least one of them should be specified.
   - Container runtime should reject unsupported image format.
   - ACI should be supported when using rkt as a runtime.


## Future work

Eventually, the scheduler should be aware of the image format, and it should prevent a pod being scheduled
to a node whose container runtime doesn't support the specified image format. This requires:
1. Kubelet reports the supported image format info in `NodeSystemInfo` or the labels, annotations of the Node.
2. Adding more fields in `PolicyPredicate` (see [#18262](https://github.com/kubernetes/kubernetes/pull/18262)) to
   allow the scheduler schedule according to the node info and the image info.

## Reference

This topic is originally discussed in [#7203](https://github.com/kubernetes/kubernetes/issues/7203)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/image-spec.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
