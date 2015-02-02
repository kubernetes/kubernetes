Hello, World!
-----------------

This is the simplest way to test persistent volumes.


This example will serve an http response of "Hello Openshift!" to [http://localhost:6061](http://localhost:6061).  To create the pod run:

        $ openshift cli create -f examples/hello-openshift/hello-pod.json

Contribute
----------

For any updates to hello_persistence.go, the hello_persistence binary should be rebuilt using:

        $ CGO_ENABLED=0 go build -a -ldflags '-s' hello_persistence.go




## Step 1: Create Persistent Volumes

As an administrator, create persistent volumes for the cluster.




