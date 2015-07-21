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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/known-issues.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Known Issues

This document summarizes known issues with existing Kubernetes releases.

Please consult this document before filing new bugs.

### Release 1.0.1

   * `exec` liveness/readiness probes leak resources due to Docker exec leaking resources (#10659)
   * `docker load` sometimes hangs which causes the `kube-apiserver` not to start.  Restarting the Docker daemon should fix the issue (#10868)
   * The kubelet on the master node doesn't register with the `kube-apiserver` so statistics aren't collected for master daemons (#10891)
   * Heapster and InfluxDB both leak memory (#10653)
   * Wrong node cpu/memory limit metrics from Heapster (https://github.com/GoogleCloudPlatform/heapster/issues/399)
   * Services that set `type=LoadBalancer` can not use port `10250` because of Google Compute Engine firewall limitations
   * Add-on services can not be created or deleted via `kubectl` or the Kubernetes API (#11435)
   * If a pod with a GCE PD is created and deleted in rapid succession, it may fail to attach/mount correctly leaving PD data inaccessible (or corrupted in the worst case). (https://github.com/GoogleCloudPlatform/kubernetes/issues/11231#issuecomment-122049113)
      * Suggested temporary work around: introduce a 1-2 minute delay between deleting and recreating a pod with a PD on the same node.
   * Explicit errors while detaching GCE PD could prevent PD from ever being detached (#11321)
   * GCE PDs may sometimes fail to attach (#11302)
   * If multiple Pods use the same RBD volume in read-write mode, it is possible data on the RBD volume could get corrupted. This problem has been found in environments where both apiserver and etcd rebooted and Pods were redistributed.
      * A workaround is to ensure there is no other Ceph client using the RBD volume before mapping RBD image in read-write mode. For example, `rados -p poolname listwatchers image_name.rbd` can list RBD clients that are mapping the image.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/known-issues.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
