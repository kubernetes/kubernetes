<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
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

## PSP RBAC Example

This example demonstrates the usage of *PodSecurityPolicy* to control access to privileged containers
based on role and groups.

### Prerequisites

The server must be started to enable the appropriate APIs and flags

1.  allow privileged containers
1.  allow security contexts
1.  enable RBAC and accept any token
1.  enable PodSecurityPolicies
1.  use the PodSecurityPolicy admission controller

If you are using the `local-up-cluster.sh` script you may enable these settings with the following syntax

```
PSP_ADMISSION=true ALLOW_PRIVILEGED=true ALLOW_SECURITY_CONTEXT=true ALLOW_ANY_TOKEN=true ENABLE_RBAC=true RUNTIME_CONFIG="extensions/v1beta1=true,extensions/v1beta1/podsecuritypolicy=true" hack/local-up-cluster.sh
```

### Using the protected port

It is important to note that this example uses the following syntax to test with RBAC

1.  `--server=https://127.0.0.1:6443`: when performing requests this ensures that the protected port is used so
that RBAC will be enforced
1.  `--token={user}/{group(s)}`: this syntax allows a request to specify the username and groups to use for
testing.  It relies on the `ALLOW_ANY_TOKEN` setting.

## Creating the policies, roles, and bindings

### Policies

The first step to enforcing cluster constraints via PSP is to create your policies.  In this
example we will use two policies, `restricted` and `privileged`.  For simplicity, the only difference
between these policies is the ability to run a privileged container.

```yaml
apiVersion: extensions/v1beta1
kind: PodSecurityPolicy
metadata:
  name: privileged
spec:
  fsGroup:
    rule: RunAsAny
  privileged: true
  runAsUser:
    rule: RunAsAny
  seLinux:
    rule: RunAsAny
  supplementalGroups:
    rule: RunAsAny
  volumes:
  - '*'
---
apiVersion: extensions/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  fsGroup:
    rule: RunAsAny
  runAsUser:
    rule: RunAsAny
  seLinux:
    rule: RunAsAny
  supplementalGroups:
    rule: RunAsAny
  volumes:
  - '*'

```

To create these policies run

```
$ kubectl --server=https://127.0.0.1:6443 --token=foo/system:masters create -f examples/podsecuritypolicy/rbac/policies.yaml 
podsecuritypolicy "privileged" created
podsecuritypolicy "restricted" created
```

### Roles and bindings

In order to a `PodSecurityPolicy` a user must have the ability to perform the `use` verb on the policy.
The `use` verb is a special verb that grants access to use the policy while
not allowing any other access.  This verb is specific to `PodSecurityPolicy`.
To enable the `use` access we will create cluster roles.  In this example we will provide the roles:

1. `restricted-psp-user`: this role allows the `use` verb on the `restricted` policy only
2. `privileged-psp-user`: this role allows the `use` verb on the `privileged` policy only


To associate roles with users we will use groups via a `RoleBinding`.  This example uses
the following groups:

1. `privileged`: this group is bound to the `privilegedPSP` role and `restrictedPSP` role which gives users
in this group access to both policies.
1. `restricted`: this group is bound to the `restrictedPSP` role
1. `system:authenticated`: this is a system group for any authenticated user.  It is bound to the `edit`
role which is already provided by the cluster.

To create these roles and bindings run

```
$ kubectl --server=https://127.0.0.1:6443 --token=foo/system:masters create -f examples/podsecuritypolicy/rbac/roles.yaml 
clusterrole "restricted-psp-user" created
clusterrole "privileged-psp-user" created

$ kubectl --server=https://127.0.0.1:6443 --token=foo/system:masters create -f examples/podsecuritypolicy/rbac/bindings.yaml 
clusterrolebinding "privileged-psp-users" created
clusterrolebinding "restricted-psp-users" created
clusterrolebinding "edit" created
```

## Testing access

### Restricted user can create non-privileged pods

Create the pod

```
$ kubectl --server=https://127.0.0.1:6443 --token=foo/restricted-psp-users create -f examples/podsecuritypolicy/rbac/pod.yaml 
pod "nginx" created
```

Check the PSP that allowed the pod

```
$ kubectl get pod nginx -o yaml | grep psp
    kubernetes.io/psp: restricted
```

### Restricted user cannot create privileged pods

Delete the existing pod

```
$ kubectl delete pod nginx
pod "nginx" deleted
```

Create the privileged pod

```
$ kubectl --server=https://127.0.0.1:6443 --token=foo/restricted-psp-users create -f examples/podsecuritypolicy/rbac/pod_priv.yaml 
Error from server (Forbidden): error when creating "examples/podsecuritypolicy/rbac/pod_priv.yaml": pods "nginx" is forbidden: unable to validate against any pod security policy: [spec.containers[0].securityContext.privileged: Invalid value: true: Privileged containers are not allowed]
```

### Privileged user can create non-privileged pods

```
$ kubectl --server=https://127.0.0.1:6443 --token=foo/privileged-psp-users create -f examples/podsecuritypolicy/rbac/pod.yaml 
pod "nginx" created
```

Check the PSP that allowed the pod.  Note, this could be the `restricted` or `privileged` PSP since both allow
for the creation of non-privileged pods.

```
$ kubectl get pod nginx -o yaml | egrep "psp|privileged"
    kubernetes.io/psp: privileged
      privileged: false
```

### Privileged user can create privileged pods

Delete the existing pod

```
$ kubectl delete pod nginx
pod "nginx" deleted
```

Create the privileged pod

```
$ kubectl --server=https://127.0.0.1:6443 --token=foo/privileged-psp-users create -f examples/podsecuritypolicy/rbac/pod_priv.yaml 
pod "nginx" created
```

Check the PSP that allowed the pod.

```
$ kubectl get pod nginx -o yaml | egrep "psp|privileged"
    kubernetes.io/psp: privileged
      privileged: true
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/podsecuritypolicy/rbac/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
