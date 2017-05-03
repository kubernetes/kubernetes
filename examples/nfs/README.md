Hello, PersistentDisks!
-----------------

This example will serve an http response of "Hello Openshift!" to [http://localhost:6061](http://localhost:6061).  To create the pod run:

        $ openshift kube create pods -c examples/hello-openshift/hello-pod.json

Contribute
----------

For any updates to hello_openshift.go, the hello_openshift binary should be rebuilt using:

        $ CGO_ENABLED=0 go build -a -ldflags '-s' hello_openshift.go
