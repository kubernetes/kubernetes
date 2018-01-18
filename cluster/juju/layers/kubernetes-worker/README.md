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

## Private registry

With the "registry" action that is part for the kubernetes-worker charm, you can very easily create a private docker registry, with authentication, and available over TLS. Please note that the registry deployed with the action is not HA, and uses storage tied to the kubernetes node where the pod is running. So if the registry pod changes is migrated from one node to another for whatever reason, you will need to re-publish the images.

### Example usage

Create the relevant authentication files. Let's say you want user `userA` to authenticate with the password `passwordA`. Then you'll do :

    echo -n "userA:passwordA" > htpasswd-plain
    htpasswd -c -b -B htpasswd userA passwordA

(the `htpasswd` program comes with the `apache2-utils` package)

Supposing your registry will be reachable at `myregistry.company.com`, and that you already have your TLS key in the `registry.key` file, and your TLS certificate (with `myregistry.company.com` as Common Name) in the `registry.crt` file, you would then run :

    juju run-action kubernetes-worker/0 registry domain=myregistry.company.com htpasswd="$(base64 -w0 htpasswd)" htpasswd-plain="$(base64 -w0 htpasswd-plain)" tlscert="$(base64 -w0 registry.crt)" tlskey="$(base64 -w0 registry.key)" ingress=true

If you then decide that you want do delete the registry, just run :

    juju run-action kubernetes-worker/0 registry delete=true ingress=true

## Known Limitations

Kubernetes workers currently only support 'phaux' HA scenarios. Even when configured with an HA cluster string, they will only ever contact the first unit in the cluster map. To enable a proper HA story, kubernetes-worker units are encouraged to proxy through a [kubeapi-load-balancer](https://jujucharms.com/kubeapi-load-balancer)
application. This enables a HA deployment without the need to
re-render configuration and disrupt the worker services.

External access to pods must be performed through a [Kubernetes
Ingress Resource](http://kubernetes.io/docs/user-guide/ingress/).

When using NodePort type networking, there is no automation in exposing the
ports selected by kubernetes or chosen by the user. They will need to be
opened manually and can be performed across an entire worker pool.

If your NodePort service port selected is `30510` you can open this across all
members of a worker pool named `kubernetes-worker` like so:

```
juju run --application kubernetes-worker open-port 30510/tcp
```

Don't forget to expose the kubernetes-worker application if its not already
exposed, as this can cause confusion once the port has been opened and the
service is not reachable.

Note: When debugging connection issues with NodePort services, its important
to first check the kube-proxy service on the worker units. If kube-proxy is not
running, the associated port-mapping will not be configured in the iptables
rulechains. 

If you need to close the NodePort once a workload has been terminated, you can
follow the same steps inversely.

```
juju run --application kubernetes-worker close-port 30510
```

