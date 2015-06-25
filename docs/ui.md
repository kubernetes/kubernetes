# Kubernetes UI Instructions

## Kubernetes User Interface
Kubernetes has an extensible user interface with default functionality that describes the current cluster. See the [README](../www/README.md) in the www directory for more information.

### Running locally
Assuming that you have a cluster running locally at `localhost:8080`, as described [here](getting-started-guides/locally.md), you can run the UI against it with kubectl:

```sh
kubectl proxy --www=www/app --www-prefix=/
```

You should now be able to access it by visiting [localhost:8001](http://localhost:8001/).

You can also use other web servers to serve the contents of the www/app directory, as described [here](../www/README.md#serving-the-app-during-development). 

### Running remotely
When Kubernetes is deployed remotely, the api server deploys the UI. To access it, visit `/static/app/` or `/ui`, which redirects to `/static/app/`, on your master server.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/ui.md?pixel)]()
