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

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/aws_ebs/README.md?pixel)]()
