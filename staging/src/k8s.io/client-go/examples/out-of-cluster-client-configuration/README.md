# Authenticating outside the cluster

This example shows you how to configure a client with client-go to authenticate
to the Kubernetes API from an application running outside the Kubernetes
cluster.

You can use your kubeconfig file that contains the context information
of your cluster to initialize a client. The kubeconfig file is also used
by the `kubectl` command to authenticate to the clusters.

## Running this example

Make sure your `kubectl` is configured and pointed to a cluster. Run
`kubectl get nodes` to confirm.

Run this application with:

    cd out-of-cluster-client-configuration
    go build -o app .
    ./app

Running this application will use the kubeconfig file and then authenticate to the
cluster, and print the number of nodes in the cluster every 10 seconds:

    $ ./app
    Used filesystem-hosted kubeconfig for configuration
    There are 3 pods in the cluster
    There are 3 pods in the cluster
    There are 3 pods in the cluster
    ...

Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to quit this application.

> **Note:** You can use the `-kubeconfig` option to use a different config file. By default
this program picks up the default file used by kubectl (when `KUBECONFIG`
environment variable is not set).

> **Note:** This example also can show how you can authenticate using the contents
of a kubeconfig file stored in an environmental variable.  A similar approach could be used to
generate configurations stored in external services (databases, etcd.)
>
> To try it out do the following:

    $ export KUBECONFIG_CONTENTS=$(cat ~/.kube/config|base64)
    $ ./app
    Used environmental variable for rest configuration
    There are 3 pods in the cluster
    There are 3 pods in the cluster
    There are 3 pods in the cluster
    ...

