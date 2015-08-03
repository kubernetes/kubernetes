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
[here](http://releases.k8s.io/release-1.0/examples/aws_ebs/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
This is a simple web server pod which serves HTML from an AWS EBS
volume.

Create a volume in the same region as your node add your volume
information in the pod description file aws-ebs-web.yaml then create
the pod:

```sh
  $ kubectl create -f examples/aws_ebs/aws-ebs-web.yaml
```

Add some data to the volume if is empty:

```sh
  $ echo  "Hello World" >& /var/lib/kubelet/plugins/kubernetes.io/aws-ebs/mounts/aws/{Region}/{Volume ID}/index.html
```

You should now be able to query your web server:

```sh
  $ curl <Pod IP address>
  $ Hello World
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/aws_ebs/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
