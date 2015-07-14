<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://releases.k8s.io/HEAD/docs/warning.png)
![WARNING](http://releases.k8s.io/HEAD/docs/warning.png)
![WARNING](http://releases.k8s.io/HEAD/docs/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://releases.k8s.io/HEAD/docs/warning.png)
![WARNING](http://releases.k8s.io/HEAD/docs/warning.png)
![WARNING](http://releases.k8s.io/HEAD/docs/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
This is a simple web server pod which serves HTML from an AWS EBS
volume.

Create a volume in the same region as your node add your volume
information in the pod description file aws-ebs-web.yaml then create
the pod:
```shell
  $ kubectl create -f aws-ebs-web.yaml
```
Add some data to the volume if is empty:
```shell
  $ echo  "Hello World" >& /var/lib/kubelet/plugins/kubernetes.io/aws-ebs/mounts/aws/{Region}/{Volume ID}/index.html
```
You should now be able to query your web server:
```shell
  $ curl <Pod IP address>
  $ Hello World
````


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/aws_ebs/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
