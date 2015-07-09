# Kubernetes User Interface
Kubernetes has an extensible user interface with default functionality that describes the current cluster. It is accessible via `https://<kubernetes-master>/ui`, which redirects to `https://<kubernetes-master>/api/v1/proxy/namespaces/kube-system/services/kube-ui/#/dashboard/`.

## Running the UI
The UI is run by default as a [cluster addon](../cluster/addons/README.md) through the [kube-ui](../cluster/addons/kube-ui) service. It is accessible via `https://<kubernetes-master>/ui`, which redirects to `https://<kubernetes-master>/api/v1/proxy/namespaces/kube-system/services/kube-ui/#/dashboard/`.

If the [`kube-addons.sh`](../cluster/saltbase/salt/kube-addons/kube-addons.sh) script is not running, the kube-ui service will not be started. In this case, it can be started manually with:
```sh
kubectl create -f cluster/addons/kube-ui/kube-ui-rc.yaml
kubectl create -f cluster/addons/kube-ui/kube-ui-svc.yaml
```

### Running locally
Assuming that you have a cluster running locally at `localhost:8080`, as described [here](getting-started-guides/locally.md), you can run the UI against it with kubectl:

```sh
kubectl proxy --www=www/app --www-prefix=/
```

You should now be able to access it by visiting [localhost:8001](http://localhost:8001/).

You can also use other web servers to serve the contents of the www/app directory, as described [here](../www/README.md#serving-the-app-during-development).

## Development
Kubernetes UI development is described [here](../www/README.md).

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/ui.md?pixel)]()
