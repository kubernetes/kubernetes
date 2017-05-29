This is a simple web server pod which serves HTML from an Cinder volume.

Create a volume in the same tenant and zone as your node.

Add your volume information in the pod description file cinder-web.yaml then create the pod:

```shell
  $ kubectl create -f examples/volumes/cinder/cinder-web.yaml
```

Add some data to the volume if is empty:

```sh
  $ echo  "Hello World" >& /var/lib/kubelet/plugins/kubernetes.io/cinder/mounts/{Volume ID}/index.html
```

You should now be able to query your web server:

```sh
  $ curl <Pod IP address>
  $ Hello World
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/cinder/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
