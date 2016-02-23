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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/user-guide/security-context.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Security Contexts

A security context defines the operating system security settings (uid, gid, capabilities, SELinux role, etc..) applied to a container. See [security context design](../design/security_context.md) for more details.

There are two levels of security context: pod level security context, and container level security context.

## Pod Level Security Context
Setting security context at the pod applies those settings to all containers in the pod 

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-world
spec:
  containers:
  # specification of the pod’s containers
  # ...
  securityContext:
    fsGroup: 1234
    supplementalGroups: [5678]
    seLinuxOptions:
      level: "s0:c123,c456"
```

Please refer to the [API documentation](https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/blob/HEAD/docs/api-reference/v1/definitions.html#_v1_podsecuritycontext) for a detailed listing and
description of all the fields available within the pod security
context.

### Volume Security context

Another functionality of pod level security context is that it applies
those settings to volumes where applicable. Specifically `fsGroup` and
`seLinuxOptions` are applied to the volume as follows:

#### `fsGroup`

Volumes which support ownership management are modified to be owned
and writable by the GID specified in `fsGroup`. See the
[Ownership Management design document](../proposals/volume-ownership-management.md)
for more details.

#### `seLinuxOptions`

Volumes which support SELinux labeling are relabled to be accessable
by the label specified unders `seLinuxOptions`. Usually you will only
need to set the `level` section. This sets the SELinux MCS label given
to all containers within the pod as well as the volume.

*Attention*: Once the MCS label is specified in the pod description
 all pods containers with he same label will able to access the
 volume. So if interpod protection is needed you must ensure each pod
 is assigned a unique MCS label.

## Container Level Security Context

Container level security context settings are applied to the specific
container and override settings made at the pod level where there is
overlap. Container level settings however do not affect the pod's
volumes.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-world
spec:
  containers:
    - name: hello-world-container
      # The container definition
      # ...
      securityContext:
        privileged: true
        seLinuxOptions:
          level: "s0:c123,c456"
```

Please refer to the
[API documentation](https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/blob/HEAD/docs/api-reference/v1/definitions.html#_v1_securitycontext)
for a detailed listing and description of all the fields available
within the container security context.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/security-context.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
