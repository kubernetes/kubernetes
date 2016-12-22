# API Server

This is a work in progress example for an API Server.
We are working on isolating the generic api server code from kubernetes specific
API objects. Some relevant issues:

* https://github.com/kubernetes/kubernetes/issues/17412
* https://github.com/kubernetes/kubernetes/issues/2742
* https://github.com/kubernetes/kubernetes/issues/13541

This code here is to examplify what it takes to write your own API server.

To start this example api server, run:

```
$ go run examples/apiserver/server/main.go
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/apiserver/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
