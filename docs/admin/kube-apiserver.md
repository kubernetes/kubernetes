<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/admin/kube-apiserver.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## kube-apiserver



### Synopsis


The Kubernetes API server validates and configures data
for the api objects which include pods, services, replicationcontrollers, and
others. The API Server services REST operations and provides the frontend to the
cluster's shared state through which all other components interact.


### Options

```
      --address=<nil>: DEPRECATED: see --insecure-bind-address instead
      --admission-control="": Ordered list of plug-ins to do admission control of resources into cluster. Comma-delimited list of: AlwaysAdmit, AlwaysDeny, DenyExecOnPrivileged, DenyEscalatingExec, LimitRanger, NamespaceAutoProvision, NamespaceExists, NamespaceLifecycle, ResourceQuota, SecurityContextDeny, ServiceAccount
      --admission-control-config-file="": File with admission control configuration.
      --advertise-address=<nil>: The IP address on which to advertise the apiserver to members of the cluster. This address must be reachable by the rest of the cluster. If blank, the --bind-address will be used. If --bind-address is unspecified, the host's default interface will be used.
      --allow-privileged=false: If true, allow privileged containers.
      --api-prefix="": The prefix for API requests on the server. Default '/api'.
      --authorization-mode="": Selects how to do authorization on the secure port.  One of: AlwaysAllow,AlwaysDeny,ABAC
      --authorization-policy-file="": File with authorization policy in csv format, used with --authorization-mode=ABAC, on the secure port.
      --basic-auth-file="": If set, the file that will be used to admit requests to the secure port of the API server via http basic authentication.
      --bind-address=<nil>: The IP address on which to serve the --read-only-port and --secure-port ports. The associated interface(s) must be reachable by the rest of the cluster, and by CLI/web clients. If blank, all interfaces will be used (0.0.0.0).
      --cert-dir="": The directory where the TLS certs are located (by default /var/run/kubernetes). If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored.
      --client-ca-file="": If set, any request presenting a client certificate signed by one of the authorities in the client-ca-file is authenticated with an identity corresponding to the CommonName of the client certificate.
      --cloud-config="": The path to the cloud provider configuration file.  Empty string for no configuration file.
      --cloud-provider="": The provider for cloud services.  Empty string for no provider.
      --cluster-name="": The instance prefix for the cluster
      --cors-allowed-origins=[]: List of allowed origins for CORS, comma separated.  An allowed origin can be a regular expression to support subdomain matching.  If this list is empty CORS will not be enabled.
      --etcd-config="": The config file for the etcd client. Mutually exclusive with -etcd-servers.
      --etcd-prefix="": The prefix for all resource paths in etcd.
      --etcd-servers=[]: List of etcd servers to watch (http://ip:port), comma separated. Mutually exclusive with -etcd-config
      --event-ttl=0: Amount of time to retain events. Default 1 hour.
      --external-hostname="": The hostname to use when generating externalized URLs for this master (e.g. Swagger API Docs.)
  -h, --help=false: help for kube-apiserver
      --insecure-bind-address=<nil>: The IP address on which to serve the --insecure-port (set to 0.0.0.0 for all interfaces). Defaults to localhost.
      --insecure-port=0: The port on which to serve unsecured, unauthenticated access. Default 8080. It is assumed that firewall rules are set up such that this port is not reachable from outside of the cluster and that port 443 on the cluster's public address is proxied to this port. This is performed by nginx in the default setup.
      --kubelet-certificate-authority="": Path to a cert. file for the certificate authority.
      --kubelet-client-certificate="": Path to a client key file for TLS.
      --kubelet-client-key="": Path to a client key file for TLS.
      --kubelet-https=false: Use https for kubelet connections
      --kubelet-port=0: Kubelet port
      --kubelet-timeout=0: Timeout for kubelet operations
      --long-running-request-regexp="(/|^)((watch|proxy)(/|$)|(logs|portforward|exec)/?$)": A regular expression matching long running requests which should be excluded from maximum inflight request handling.
      --master-service-namespace="": The namespace from which the Kubernetes master services should be injected into pods
      --max-requests-inflight=400: The maximum number of requests in flight at a given time.  When the server exceeds this, it rejects requests.  Zero for no limit.
      --min-request-timeout=1800: An optional field indicating the minimum number of seconds a handler must keep a request open before timing it out. Currently only honored by the watch request handler, which picks a randomized value above this number as the connection timeout, to spread out load.
      --old-etcd-prefix="": The previous prefix for all resource paths in etcd, if any.
      --port=0: DEPRECATED: see --insecure-port instead
      --profiling=true: Enable profiling via web interface host:port/debug/pprof/
      --public-address-override=<nil>: DEPRECATED: see --bind-address instead
      --runtime-config=: A set of key=value pairs that describe runtime configuration that may be passed to the apiserver. api/<version> key can be used to turn on/off specific api versions. api/all and api/legacy are special keys to control all and legacy api versions respectively.
      --secure-port=0: The port on which to serve HTTPS with authentication and authorization. If 0, don't serve HTTPS at all.
      --service-account-key-file="": File containing PEM-encoded x509 RSA private or public key, used to verify ServiceAccount tokens. If unspecified, --tls-private-key-file is used.
      --service-account-lookup=false: If true, validate ServiceAccount tokens exist in etcd as part of authentication.
      --service-cluster-ip-range=<nil>: A CIDR notation IP range from which to assign service cluster IPs. This must not overlap with any IP ranges assigned to nodes for pods.
      --service-node-port-range=: A port range to reserve for services with NodePort visibility.  Example: '30000-32767'.  Inclusive at both ends of the range.
      --ssh-keyfile="": If non-empty, use secure SSH proxy to the nodes, using this user keyfile
      --ssh-user="": If non-empty, use secure SSH proxy to the nodes, using this user name
      --storage-version="": The version to store resources with. Defaults to server preferred
      --tls-cert-file="": File containing x509 Certificate for HTTPS.  (CA cert, if any, concatenated after server cert). If HTTPS serving is enabled, and --tls-cert-file and --tls-private-key-file are not provided, a self-signed certificate and key are generated for the public address and saved to /var/run/kubernetes.
      --tls-private-key-file="": File containing x509 private key matching --tls-cert-file.
      --token-auth-file="": If set, the file that will be used to secure the secure port of the API server via token authentication.
```

###### Auto generated by spf13/cobra at 2015-07-06 18:03:28.852677626 +0000 UTC


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/kube-apiserver.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
