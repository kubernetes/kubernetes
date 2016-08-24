# client-go

Go clients for talking to a [kubernetes](http://kubernetes.io/) cluster.

### What's included

* The `kuberentes` package contains the clientset to access Kubernetes API.
* The `discovery` package is used to discovery API supported on the Kubernetes API server.
* The `dynamic` package contains a dynamic client that can handle API objects without knowing the scheme.
* The `transport` package is used to set up auth and start a connection.
* The `tools/cache` package is useful for writing controllers.

### Releases

Each top-level folder (e.g., 1.4) contains a release of clients and their dependencies.

client-go has the same release cycle as the Kubernetes main repository. For example, in the 1.4 release cycle, the contents in `1.4/` folder are subjected to changes. The `1.4/` folder will be stable once the 1.4 release is final, and new changes will go into the `1.5/` folder.

### How to get it

You can `go get` a release of client-go, e.g., `go get k8s.io/client-go/1.4`.

### Reporting bugs

Please report bugs to the main Kubernetes repository.

### Contributing code
Please send pull requests against the client packages in the Kubernetes main repository, and run the `/staging/src/k8s.io/client-go/copy.sh` script to update the staging area in the main repository. Changes in the staging area will be published to this repository every day.
