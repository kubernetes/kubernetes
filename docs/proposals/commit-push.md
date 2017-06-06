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

# commit and push capability proposal

### Motivation

There are some use case to commit a container, and push its resulting image to a registry, for example:

1. For a run-once pod whose result represents an image that should be pushed to a registry for consumption
by other actors in the system

2. There is someone desire to be able to modify a running container and then commit/push the change so they
could, for example, make a debug change and then scale up the deployment to test it in a clustered mode,
without having to start from scratch with building a new image.

The goal of this proposal is to add the commit/push capability, which will allow user commit+push their
containers manually, or automically after container exits.



### Design

1. Add `pod/commit` and `pod/push` subsource to support user commit+push their containers manually.

2. Add `PosStop` life-cycle hook to support commit+push automically after container exits.


### API

1) We will add

```
ImagePushSecrets []LocalObjectReference `json:"imagePushSecrets,omitempty"`
```

to `PodSpec`, using to push the image to a registry.

2) We will add a `PodCommitOptions` and `PodPushOptions`api, indicating the commit/push options.

```
// PodCommitOptions is the query options to a Pod's commit call
type PodCommitOptions struct {
    // Container which to commit.
	Container string `json:"container",omitempty`

	// Author(e.g.,"foo, <foo@bar.com>)
	Author string `json:"author",omitempty`

	// Commit Message
	Message string `json:"message",omitempty`

	// Pause container during commit
	Pause bool `json:"pause",omitempty`

	// Apply Dockerfile instruction to the created image
	Change []string `json:"change",omitempty`
}

// PodPushOptions is the query options to a Pod's push call
type PodPushOptions struct {
	// Name of the image
	Name string `json:"name",omitempty`

	// Tag of the image
	Tag string `json:"tag",omitempty`

	// Registry server to push the image
	Registry string `json:"registry",omitempty`
}

```

3) We will add `PostStop` to `Lifecyle`, using to commit/push container automically after container exits

```
type Lifecycle struct {
	/*
	...
	*/
	PostStop *Handler `json:"postStop",omitempty`
}

type Handler struct {
	/*
	...
	*/
	RuntimeExec *RuntimeExecAction `json:"runtimeExec",omitempty`
}

type RuntimeExecAction struct {
	Commands []RuntimeCommand `json:"commands",omitempty`
}

type RuntimeCommand struct {
	Command RuntimeCommandType `json:"command",omitempty`
	CommitOptions *PodCommitOptions `json:"commitOptions",omitempty`
	PushOptions *PodPushOptions `json:"pushOptions",omitempty`
}

type RuntimeCommandType string

const(
	RuntimeCommandCommit RuntimeCommandType = "Commit" 
	RuntimeCommandPush   RuntimeCommandType = "Push"
)
```


### Implementation Details

1) Add `AvoidGC(containerID string)` method to the ImageManager, which will add the container to a "NotGC" set,
 when ImageManager gc not used images, it will not gc the image in the "NotGC" set.

2) When user commit their container using the `pod/commit` subsource, the generated image will be add to the "NotGC" set.
The image can not be GCed until user has successfully pushed their container. However, if user just commited but not pushed
their container, the image could still be unfortunately GCed/lost if kubelet restart or node crashed. So we recommend user
push their container after commit. After all, I am not sure I see a use case for commit but no-push.

3) It seems that rkt doesn't support commit&push by now(correct me if I am wrong), we can return a error message(indicating
that the command is not supported) to user when user want manually commit/push their container. If user want commit/push a
ACI container automically through the lifecycle hook, we can disallow it at validation time(I notice there is image-spec
proposal [image-spec](https://github.com/kubernetes/kubernetes/pull/18308) which allow us distinguish Docker image and appc container image).

### Further work

1) rkt may support commit/push images in the future(I am not sure).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/commit-push.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
