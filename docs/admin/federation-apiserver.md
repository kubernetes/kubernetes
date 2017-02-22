## federation-apiserver



### Synopsis


The Kubernetes federation API server validates and configures data
for the api objects which include pods, services, replicationcontrollers, and
others. The API Server services REST operations and provides the frontend to the
cluster's shared state through which all other components interact.

```
federation-apiserver
```

### Options

```
      --admission-control string                                Ordered list of plug-ins to do admission control of resources into cluster. Comma-delimited list of: AlwaysAdmit, AlwaysDeny, NamespaceLifecycle, OwnerReferencesPermissionEnforcement. (default "AlwaysAdmit")
      --admission-control-config-file string                    File with admission control configuration.
      --advertise-address ip                                    The IP address on which to advertise the apiserver to members of the cluster. This address must be reachable by the rest of the cluster. If blank, the --bind-address will be used. If --bind-address is unspecified, the host's default interface will be used.
      --anonymous-auth                                          Enables anonymous requests to the secure port of the API server. Requests that are not rejected by another authentication method are treated as anonymous requests. Anonymous requests have a username of system:anonymous, and a group name of system:unauthenticated. (default true)
      --audit-log-maxage int                                    The maximum number of days to retain old audit log files based on the timestamp encoded in their filename.
      --audit-log-maxbackup int                                 The maximum number of old audit log files to retain.
      --audit-log-maxsize int                                   The maximum size in megabytes of the audit log file before it gets rotated. Defaults to 100MB.
      --audit-log-path string                                   If set, all requests coming to the apiserver will be logged to this file.
      --authentication-token-webhook-cache-ttl duration         The duration to cache responses from the webhook token authenticator. Default is 2m. (default 2m0s)
      --authentication-token-webhook-config-file string         File with webhook configuration for token authentication in kubeconfig format. The API server will query the remote service to determine authentication for bearer tokens.
      --authorization-mode string                               Ordered list of plug-ins to do authorization on secure port. Comma-delimited list of: AlwaysAllow,AlwaysDeny,ABAC,Webhook,RBAC. (default "AlwaysAllow")
      --authorization-policy-file string                        File with authorization policy in csv format, used with --authorization-mode=ABAC, on the secure port.
      --authorization-webhook-cache-authorized-ttl duration     The duration to cache 'authorized' responses from the webhook authorizer. Default is 5m. (default 5m0s)
      --authorization-webhook-cache-unauthorized-ttl duration   The duration to cache 'unauthorized' responses from the webhook authorizer. Default is 30s. (default 30s)
      --authorization-webhook-config-file string                File with webhook configuration in kubeconfig format, used with --authorization-mode=Webhook. The API server will query the remote service to determine access on the API server's secure port.
      --basic-auth-file string                                  If set, the file that will be used to admit requests to the secure port of the API server via http basic authentication.
      --bind-address ip                                         The IP address on which to listen for the --secure-port port. The associated interface(s) must be reachable by the rest of the cluster, and by CLI/web clients. If blank, all interfaces will be used (0.0.0.0). (default 0.0.0.0)
      --cert-dir string                                         The directory where the TLS certs are located (by default /var/run/kubernetes). If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored. (default "/var/run/kubernetes")
      --client-ca-file string                                   If set, any request presenting a client certificate signed by one of the authorities in the client-ca-file is authenticated with an identity corresponding to the CommonName of the client certificate.
      --cloud-config string                                     The path to the cloud provider configuration file. Empty string for no configuration file.
      --cloud-provider string                                   The provider for cloud services. Empty string for no provider.
      --contention-profiling                                    Enable contention profiling. Requires --profiling to be set to work.
      --cors-allowed-origins stringSlice                        List of allowed origins for CORS, comma separated.  An allowed origin can be a regular expression to support subdomain matching. If this list is empty CORS will not be enabled.
      --delete-collection-workers int                           Number of workers spawned for DeleteCollection call. These are used to speed up namespace cleanup. (default 1)
      --deserialization-cache-size int                          Number of deserialized json objects to cache in memory.
      --enable-garbage-collector                                Enables the generic garbage collector. MUST be synced with the corresponding flag of the kube-controller-manager. (default true)
      --enable-swagger-ui                                       Enables swagger ui on the apiserver at /swagger-ui
      --etcd-cafile string                                      SSL Certificate Authority file used to secure etcd communication.
      --etcd-certfile string                                    SSL certification file used to secure etcd communication.
      --etcd-keyfile string                                     SSL key file used to secure etcd communication.
      --etcd-prefix string                                      The prefix to prepend to all resource paths in etcd. (default "/registry")
      --etcd-quorum-read                                        If true, enable quorum read.
      --etcd-servers stringSlice                                List of etcd servers to connect with (scheme://ip:port), comma separated.
      --etcd-servers-overrides stringSlice                      Per-resource etcd servers overrides, comma separated. The individual override format: group/resource#servers, where servers are http://ip:port, semicolon separated.
      --event-ttl duration                                      Amount of time to retain events. Default is 1h. (default 1h0m0s)
      --experimental-keystone-ca-file string                    If set, the Keystone server's certificate will be verified by one of the authorities in the experimental-keystone-ca-file, otherwise the host's root CA set will be used.
      --experimental-keystone-url string                        If passed, activates the keystone authentication plugin.
      --external-hostname string                                The hostname to use when generating externalized URLs for this master (e.g. Swagger API Docs).
      --feature-gates mapStringBool                             A set of key=value pairs that describe feature gates for alpha/experimental features. Options are:
AffinityInAnnotations=true|false (ALPHA - default=false)
AllAlpha=true|false (ALPHA - default=false)
AllowExtTrafficLocalEndpoints=true|false (BETA - default=true)
AppArmor=true|false (BETA - default=true)
DynamicKubeletConfig=true|false (ALPHA - default=false)
DynamicVolumeProvisioning=true|false (ALPHA - default=true)
ExperimentalCriticalPodAnnotation=true|false (ALPHA - default=false)
ExperimentalHostUserNamespaceDefaulting=true|false (BETA - default=false)
StreamingProxyRedirects=true|false (BETA - default=true)
      --insecure-allow-any-token username/group1,group2         If set, your server will be INSECURE.  Any token will be allowed and user information will be parsed from the token as username/group1,group2
      --insecure-bind-address ip                                The IP address on which to serve the --insecure-port (set to 0.0.0.0 for all interfaces). Defaults to localhost. (default 127.0.0.1)
      --insecure-port int                                       The port on which to serve unsecured, unauthenticated access. Default 8080. It is assumed that firewall rules are set up such that this port is not reachable from outside of the cluster and that port 443 on the cluster's public address is proxied to this port. This is performed by nginx in the default setup. (default 8080)
      --master-service-namespace string                         DEPRECATED: the namespace from which the kubernetes master services should be injected into pods. (default "default")
      --max-mutating-requests-inflight int                      The maximum number of mutating requests in flight at a given time. When the server exceeds this, it rejects requests. Zero for no limit. (default 200)
      --max-requests-inflight int                               The maximum number of non-mutating requests in flight at a given time. When the server exceeds this, it rejects requests. Zero for no limit. (default 400)
      --min-request-timeout int                                 An optional field indicating the minimum number of seconds a handler must keep a request open before timing it out. Currently only honored by the watch request handler, which picks a randomized value above this number as the connection timeout, to spread out load. (default 1800)
      --oidc-ca-file string                                     If set, the OpenID server's certificate will be verified by one of the authorities in the oidc-ca-file, otherwise the host's root CA set will be used.
      --oidc-client-id string                                   The client ID for the OpenID Connect client, must be set if oidc-issuer-url is set.
      --oidc-groups-claim string                                If provided, the name of a custom OpenID Connect claim for specifying user groups. The claim value is expected to be a string or array of strings. This flag is experimental, please see the authentication documentation for further details.
      --oidc-issuer-url string                                  The URL of the OpenID issuer, only HTTPS scheme will be accepted. If set, it will be used to verify the OIDC JSON Web Token (JWT).
      --oidc-username-claim string                              The OpenID claim to use as the user name. Note that claims other than the default ('sub') is not guaranteed to be unique and immutable. This flag is experimental, please see the authentication documentation for further details. (default "sub")
      --profiling                                               Enable profiling via web interface host:port/debug/pprof/ (default true)
      --requestheader-allowed-names stringSlice                 List of client certificate common names to allow to provide usernames in headers specified by --requestheader-username-headers. If empty, any client certificate validated by the authorities in --requestheader-client-ca-file is allowed.
      --requestheader-client-ca-file string                     Root certificate bundle to use to verify client certificates on incoming requests before trusting usernames in headers specified by --requestheader-username-headers
      --requestheader-extra-headers-prefix stringSlice          List of request header prefixes to inspect. X-Remote-Extra- is suggested.
      --requestheader-group-headers stringSlice                 List of request headers to inspect for groups. X-Remote-Group is suggested.
      --requestheader-username-headers stringSlice              List of request headers to inspect for usernames. X-Remote-User is common.
      --runtime-config mapStringString                          A set of key=value pairs that describe runtime configuration that may be passed to apiserver. apis/<groupVersion> key can be used to turn on/off specific api versions. apis/<groupVersion>/<resource> can be used to turn on/off specific resources. api/all and api/legacy are special keys to control all and legacy api versions respectively.
      --secure-port int                                         The port on which to serve HTTPS with authentication and authorization. If 0, don't serve HTTPS at all. (default 6443)
      --service-account-key-file stringArray                    File containing PEM-encoded x509 RSA or ECDSA private or public keys, used to verify ServiceAccount tokens. If unspecified, --tls-private-key-file is used. The specified file can contain multiple keys, and the flag can be specified multiple times with different files.
      --service-account-lookup                                  If true, validate ServiceAccount tokens exist in etcd as part of authentication.
      --storage-backend string                                  The storage backend for persistence. Options: 'etcd3' (default), 'etcd2'.
      --storage-media-type string                               The media type to use to store objects in storage. Defaults to application/json. Some resources may only support a specific media type and will ignore this setting. (default "application/vnd.kubernetes.protobuf")
      --storage-versions string                                 The per-group version to store resources in. Specified in the format "group1/version1,group2/version2,...". In the case where objects are moved from one group to the other, you may specify the format "group1=group2/v1beta1,group3/v1beta1,...". You only need to pass the groups you wish to change from the defaults. It defaults to a list of preferred versions of all registered groups, which is derived from the KUBE_API_VERSIONS environment variable. (default "apps/v1beta1,authentication.k8s.io/v1,authorization.k8s.io/v1,autoscaling/v1,batch/v1,certificates.k8s.io/v1beta1,componentconfig/v1alpha1,extensions/v1beta1,federation/v1beta1,policy/v1beta1,rbac.authorization.k8s.io/v1beta1,storage.k8s.io/v1beta1,v1")
      --target-ram-mb int                                       Memory limit for apiserver in MB (used to configure sizes of caches, etc.)
      --tls-ca-file string                                      If set, this certificate authority will used for secure access from Admission Controllers. This must be a valid PEM-encoded CA bundle. Altneratively, the certificate authority can be appended to the certificate provided by --tls-cert-file.
      --tls-cert-file string                                    File containing the default x509 Certificate for HTTPS. (CA cert, if any, concatenated after server cert). If HTTPS serving is enabled, and --tls-cert-file and --tls-private-key-file are not provided, a self-signed certificate and key are generated for the public address and saved to /var/run/kubernetes.
      --tls-private-key-file string                             File containing the default x509 private key matching --tls-cert-file.
      --tls-sni-cert-key namedCertKey                           A pair of x509 certificate and private key file paths, optionally suffixed with a list of domain patterns which are fully qualified domain names, possibly with prefixed wildcard segments. If no domain patterns are provided, the names of the certificate are extracted. Non-wildcard matches trump over wildcard matches, explicit domain patterns trump over extracted names. For multiple key/certificate pairs, use the --tls-sni-cert-key multiple times. Examples: "example.key,example.crt" or "*.foo.com,foo.com:foo.key,foo.crt". (default [])
      --token-auth-file string                                  If set, the file that will be used to secure the secure port of the API server via token authentication.
      --watch-cache                                             Enable watch caching in the apiserver (default true)
      --watch-cache-sizes stringSlice                           List of watch cache sizes for every resource (pods, nodes, etc.), comma separated. The individual override format: resource#size, where size is a number. It takes effect when watch-cache is enabled.
```

###### Auto generated by spf13/cobra on 21-Feb-2017
