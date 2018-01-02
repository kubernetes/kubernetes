Configuring sources
===================

Heapster can get data from multiple sources (although at this momeent we support only one kind - Kubernetes).
They are specified in the command line via the `--source` flag. The flag takes an argument of the form `PREFIX:CONFIG[?OPTIONS]`.
Options (optional!) are specified as URL query parameters, separated by `&` as normal.
This allows each source to have custom configuration passed to it without needing to
continually add new flags to Heapster as new sources are added. This also means
Heapster can capture metrics from multiple sources at once, potentially even multiple
Kubernetes clusters.

## Current sources
### Kubernetes
To use the kubernetes source add the following flag:

	--source=kubernetes:<KUBERNETES_MASTER>[?<KUBERNETES_OPTIONS>]

If you're running Heapster in a Kubernetes pod you can use the following flag:

	--source=kubernetes

Heapster requires an authentication token to connect with the apiserver securely. By default, Heapster will use the inClusterConfig system to configure the secure connection. This requires kubernetes version `v1.0.3` or higher and a couple extra kubernetes configuration steps. Firstly, for your apiserver you must create a SSL certificate pair with a SAN that includes the ClusterIP of the kubernetes service. Look [here](https://github.com/kubernetes/kubernetes/blob/e4fde6d2cae2d924a4eb72d1e3b2639f057bb8c1/cluster/gce/util.sh#L497-L559) for an example of how to properly generate certs. Secondly, you need to pass the `ca.crt` that you generated to the `--root-ca-file` option of the controller-manager. This will distribute the root CA to `/var/run/secrets/kubernetes.io/serviceaccount/` of all pods. If you are using `ABAC` authorization (as opposed to `AllowAll` which is the default), you will also need to give the `system:serviceaccount:<namespace-of-heapster>:default` readonly access to the cluster (look [here](https://github.com/kubernetes/kubernetes/blob/master/docs/admin/authorization.md#a-quick-note-on-service-accounts) for more info).

If you don't want to setup inClusterConfig, you can still use Heapster! To run without auth, use the following config:

	--source=kubernetes:http://<address-of-kubernetes-master>:<http-port>?inClusterConfig=false

This requires the apiserver to be setup completely without auth, which can be done by binding the insecure port to all interfaces (see the apiserver `--insecure-bind-address` option) but *WARNING* be aware of the security repercussions. Only do this if you trust *EVERYONE* on your network.

*Note: Remove "monitoring-token" volume from heaspter controller config if you are running without auth.*

Alternatively, you can use a heapster-only serviceaccount like this:

```shell
cat <EOF | kubectl create -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: heapster
EOF
```

This will generate a token on the API server. You will then need to reference the service account in your Heapster pod spec like this:

```yaml
apiVersion: "v1"
kind: "ReplicationController"
metadata:
  labels:
    name: "heapster"
  name: "monitoring-heapster-controller"
spec:
  replicas: 1
  selector:
    name: "heapster"
  template:
    metadata:
      labels:
        name: "heapster"
    spec:
      serviceAccount: "heapster"
      containers:
        -
          image: "kubernetes/heapster:v0.13.0"
          name: "heapster"
          command:
            - "/heapster"
            - "--source=kubernetes:http://kubernetes-ro?inClusterConfig=false&useServiceAccount=true&auth="
            - "--sink=influxdb:http://monitoring-influxdb:80"
```

This will mount the generated token at `/var/run/secrets/kubernetes.io/serviceaccount/token` in the Heapster container.


The following options are available:
* `inClusterConfig` - Use kube config in service accounts associated with Heapster's namesapce. (default: true)
* `kubeletPort` - kubelet port to use (default: `10255`)
* `kubeletHttps` - whether to use https to connect to kubelets (default: `false`)
* `apiVersion` - API version to use to talk to Kubernetes. Defaults to the version in kubeConfig.
* `insecure` - whether to trust kubernetes certificates (default: `false`)
* `auth` - client auth file to use. Set auth if the service accounts are not usable.
* `useServiceAccount` - whether to use the service account token if one is mounted at `/var/run/secrets/kubernetes.io/serviceaccount/token` (default: `false`)

There is also a sub-source for metrics - `kubernetes.summary_api` - that uses a slightly different, memory-efficient API for passing data from Kubelet/cAdvisor to Heapster. It supports the same set of options as `kubernetes`. Sample usage:
```
 - --source=kubernetes.summary_api:'' 
```