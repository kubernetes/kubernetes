# Admission Controllers

## What are they?

An admission control plug-in is a piece of code that intercepts requests to the Kubernetes
API server prior to persistence of the object.  The plug-in code is in the API server process
and must be compiled into the binary in order to be used at this time.

Each admission control plug-in is run in sequence before a request is accepted into the cluster.  If
any of the plug-ins in the sequence reject the request, the entire request is rejected immediately
and an error is returned to the end-user.

Admission control plug-ins may mutate the incoming object in some cases to apply system configured
defaults.  In addition, admission control plug-ins may mutate related resources as part of request
processing to do things like increment quota usage.

## Why do I need them?

Many advanced features in Kubernetes require an admission control plug-in to be enabled in order
to properly support the feature.  As a result, a Kubernetes API server that is not properly
configured with the right set of admission control plug-ins is an incomplete server and will not
support all the features you expect.

## How do I turn on an admission control plug-in?

The Kubernetes API server supports a flag, ```admission_control``` that takes a comma-delimited,
ordered list of admission control choices to invoke prior to modifying objects in the cluster.

## What does each plug-in do?

### AlwaysAdmit

This plug-in will accept all incoming requests made to the Kubernetes API server.

### AlwaysDeny

This plug-in will reject all mutating requests made to the Kubernetes API server.  It's largely intended
for testing purposes and is not recommended for usage in a real deployment.

### DenyExecOnPrivileged

This plug-in will intercept all requests to exec a command in a pod if that pod has a privileged container.

If your cluster supports privileged containers, and you want to restrict the ability of end-users to exec
commands in those containers, we strongly encourage enabling this plug-in.

### ServiceAccount

This plug-in limits admission of Pod creation requests based on the Pod's ```ServiceAccount```.

1. If the pod does not have a ```ServiceAccount```, it modifies the pod's ```ServiceAccount``` to "default".
2. It ensures that the ```ServiceAccount``` referenced by a pod exists.
3. If ```LimitSecretReferences``` is true, it rejects the pod if the pod references ```Secret``` objects which the pods
```ServiceAccount``` does not reference.
4. If the pod does not contain any ```ImagePullSecrets```, the ```ImagePullSecrets``` of the
```ServiceAccount``` are added to the pod.
5. If ```MountServiceAccountToken``` is true, it adds a ```VolumeMount``` with the pod's
```ServiceAccount``` API token secret to containers in the pod.

We strongly recommend using this plug-in if you intend to make use of Kubernetes ```ServiceAccount``` objects.

### SecurityContextDeny

This plug-in will deny any ```SecurityContext``` that defines options that were not available on the ```Container```.

### ResourceQuota

This plug-in will observe the incoming request and ensure that it does not violate any of the constraints
enumerated in the ```ResourceQuota``` object in a ```Namespace```.  If you are using ```ResourceQuota```
objects in your Kubernetes deployment, you MUST use this plug-in to enforce quota constraints.

It is strongly encouraged that this plug-in is configured last in the sequence of admission control plug-ins.  This is
so that quota is not prematurely incremented only for the request to be rejected later in admission control.

### LimitRanger

This plug-in will observe the incoming request and ensure that it does not violate any of the constraints
enumerated in the ```LimitRange``` object in a ```Namespace```.  If you are using ```LimitRange``` objects in
your Kubernetes deployment, you MUST use this plug-in to enforce those constraints.

### NamespaceExists

This plug-in will observe all incoming requests that attempt to create a resource in a Kubernetes ```Namespace```
and reject the request if the ```Namespace``` was not previously created.  We strongly recommend running
this plug-in to ensure integrity of your data.

### NamespaceAutoProvision (deprecated)

This plug-in will observe all incoming requests that attempt to create a resource in a Kubernetes ```Namespace```
and create a new ```Namespace``` if one did not already exist previously.

We strongly recommend ```NamespaceExists``` over ```NamespaceAutoProvision```.

### NamespaceLifecycle

This plug-in enforces that a ```Namespace``` that is undergoing termination cannot have new content created in it.

A ```Namespace``` deletion kicks off a sequence of operations that remove all content (pods, services, etc.) in that
namespace.  In order to enforce integrity of that process, we strongly recommend running this plug-in.

Once ```NamespaceAutoProvision``` is deprecated, we anticipate ```NamespaceLifecycle``` and ```NamespaceExists``` will
be merged into a single plug-in that enforces the life-cycle of a ```Namespace``` in Kubernetes.

## Is there a recommended set of plug-ins to use?

Yes.

For Kubernetes 1.0, we strongly recommend running the following set of admission control plug-ins:

```shell
--admission_control=NamespaceLifecycle,NamespaceExists,LimitRanger,SecurityContextDeny,ServiceAccount,ResourceQuota
```
