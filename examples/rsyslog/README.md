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
[here](http://releases.k8s.io/release-1.0/examples/rsyslog/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# rsyslog Service

This example creates a syslog service that listens on a ClusterIP and writes log
files to a persistent volume. The syslog service is configured using a secrets
volume.

## Creating Secrets Volume

The configuration for the rsyslog daemon is stored in a secrets volume which
is attached when the pod is started. Use the `make_secrets.go` script to help
create the volume:

```sh
% go run examples/utils/make_secrets.go examples/rsyslog/rsyslog-cfg | \
	kubectl create -f -
secrets/rsyslog-cfg
```

## Creating a Persistent Volume

The `rsyslog-example` replication controller creates a pod that uses a volume
claim to find its persistent volume. This means you'll need to have an available
persistent volume of sufficient size to which the claim can bind.

The configuration for the persistent volume will differ based on the type.
Here's a configuration that would create a persistent volume backed by an
(existing) GCE persistent disk:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: syslog-volume
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  gcePersistentDisk:
    pdName: "syslog-volume"
    fsType: "ext4"
```

You can modify this example in `examples/rsyslog/rsyslog-pv-gce.yaml.tmpl` to
match your needs or use it directly by [creating and formatting a persistent
disk](https://cloud.google.com/compute/docs/disks/persistent-disks) named
`syslog-volume` and then creating the Persisten Volume:

```sh
% kc create -f examples/rsyslog/rsyslog-pv-gce.yaml.tmpl
persistentvolumes/syslog-volume
```

Creating the persistent volume will make it available:

```sh
% kubectl get pv syslog-volume
NAME             LABELS    CAPACITY      ACCESSMODES   STATUS      CLAIM     REASON
syslog-volume    <none>    53687091200   RWO           Available
```

When you create the persistent volume claim in the next step it will bind
to this available volume.

Learn more about [Persistent Volumes & Claims](../../docs/user-guide/persistent-volumes.md).

## Create the rsyslog Service, Volume Claim and Replication Controller

The example configuration for the
[Service](rsyslog-service.yaml),
[Replication Controller](rsyslog-controller.yaml), and
[Persistent Volume Claim](rsyslog-pvc.yaml) should all work without
modification:

```sh
% kubectl create -f examples/rsyslog
replicationcontrollers/rsyslog-controller
persistentvolumeclaims/rsyslog-volume
services/rsyslog-service
```

We can now write to the rsyslog service:

```sh
% kubectl get -l app=rsyslog po
NAME                       READY     STATUS    RESTARTS   AGE
rsyslog-controller-vgox2   1/1       Running   0          1m

% kubectl port-forward -p rsyslog-controller-vgox2 1514:514  &
I1027 17:34:42.546609    3524 portforward.go:225] Forwarding from 127.0.0.1:1514 -> 514
I1027 17:34:42.546837    3524 portforward.go:225] Forwarding from [::1]:1514 -> 514

% echo "This is my log message" | nc -v localhost 1514
Connection to localhost 1514 port [tcp/*] succeeded!
I1027 17:34:55.425638    3524 portforward.go:251] Handling connection for 1514

% kill %1
[1]  + exit 2     kubectl port-forward -p rsyslog-controller-vgox2 1514:514
% kubectl exec rsyslog-controller-vgox2 tail /log/messages
2015-10-28T00:33:27.795167+00:00 rsyslog-controller-vgox2 rsyslogd: [origin software="rsyslogd" swVersion="8.9.0" x-pid="1" x-info="http://www.rsyslog.com"] start
2015-10-28T00:34:55.911615+00:00 This is my log message
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/rsyslog/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
