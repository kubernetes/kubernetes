# Kubernetes Worker

## Usage

This charm deploys a container runtime, and additionally stands up the Kubernetes
worker applications: kubelet, and kube-proxy.

In order for this charm to be useful, it should be deployed with its companion
charm [kubernetes-master](https://jujucharms.com/u/containers/kubernetes-master)
and linked with an SDN-Plugin.

This charm has also been bundled up for your convenience so you can skip the
above steps, and deploy it with a single command:

```shell
juju deploy canonical-kubernetes
```

For more information about [Canonical Kubernetes](https://jujucharms.com/canonical-kubernetes)
consult the bundle `README.md` file.


## Scale out

To add additional compute capacity to your Kubernetes workers, you may
`juju add-unit` scale the cluster of applications. They will automatically
join any related kubernetes-master, and enlist themselves as ready once the
deployment is complete.

## Operational actions

The kubernetes-worker charm supports the following Operational Actions:

#### Pause

Pausing the workload enables administrators to both [drain](http://kubernetes.io/docs/user-guide/kubectl/kubectl_drain/) and [cordon](http://kubernetes.io/docs/user-guide/kubectl/kubectl_cordon/)
a unit for maintenance.


#### Resume

Resuming the workload will [uncordon](http://kubernetes.io/docs/user-guide/kubectl/kubectl_uncordon/) a paused unit. Workloads will automatically migrate unless otherwise directed via their application declaration.

## Known Limitations

Kubernetes workers currently only support 'phaux' HA scenarios. Even when configured with an HA cluster string, they will only ever contact the first unit in the cluster map. To enalbe a proper HA story, kubernetes-worker units are encouraged to proxy through a [kubeapi-load-balancer](https://jujucharms.com/kubeapi-load-balancer)
application. This enables a HA deployment without the need to
re-render configuration and disrupt the worker services.

External access to pods must be performed through a [Kubernetes
Ingress Resource](http://kubernetes.io/docs/user-guide/ingress/). More
information
