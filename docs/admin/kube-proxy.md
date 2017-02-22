## kube-proxy



### Synopsis


The Kubernetes network proxy runs on each node. This
reflects services as defined in the Kubernetes API on each node and can do simple
TCP,UDP stream forwarding or round robin TCP,UDP forwarding across a set of backends.
Service cluster ips and ports are currently found through Docker-links-compatible
environment variables specifying ports opened by the service proxy. There is an optional
addon that provides cluster DNS for these cluster IPs. The user must create a service
with the apiserver API to configure the proxy.

```
kube-proxy
```

### Options

```
      --azure-container-registry-config string       Path to the file container Azure container registry configuration information.
      --bind-address ip                              The IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces) (default 0.0.0.0)
      --cleanup-iptables                             If true cleanup iptables rules and exit.
      --cluster-cidr string                          The CIDR range of pods in the cluster. It is used to bridge traffic coming from outside of the cluster. If not provided, no off-cluster bridging will be performed.
      --config-sync-period duration                  How often configuration from the apiserver is refreshed.  Must be greater than 0. (default 15m0s)
      --conntrack-max-per-core int32                 Maximum number of NAT connections to track per CPU core (0 to leave the limit as-is and ignore conntrack-min). (default 32768)
      --conntrack-min int32                          Minimum number of conntrack entries to allocate, regardless of conntrack-max-per-core (set conntrack-max-per-core=0 to leave the limit as-is). (default 131072)
      --conntrack-tcp-timeout-close-wait duration    NAT timeout for TCP connections in the CLOSE_WAIT state (default 1h0m0s)
      --conntrack-tcp-timeout-established duration   Idle timeout for established TCP connections (0 to leave as-is) (default 24h0m0s)
      --feature-gates mapStringBool                  A set of key=value pairs that describe feature gates for alpha/experimental features. Options are:
AffinityInAnnotations=true|false (ALPHA - default=false)
AllAlpha=true|false (ALPHA - default=false)
AllowExtTrafficLocalEndpoints=true|false (BETA - default=true)
AppArmor=true|false (BETA - default=true)
DynamicKubeletConfig=true|false (ALPHA - default=false)
DynamicVolumeProvisioning=true|false (ALPHA - default=true)
ExperimentalCriticalPodAnnotation=true|false (ALPHA - default=false)
ExperimentalHostUserNamespaceDefaulting=true|false (BETA - default=false)
StreamingProxyRedirects=true|false (BETA - default=true)
      --google-json-key string                       The Google Cloud Platform Service Account JSON Key to use for authentication.
      --healthz-bind-address ip                      The IP address for the health check server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces) (default 127.0.0.1)
      --healthz-port int32                           The port to bind the health check server. Use 0 to disable. (default 10249)
      --hostname-override string                     If non-empty, will use this string as identification instead of the actual hostname.
      --iptables-masquerade-bit int32                If using the pure iptables proxy, the bit of the fwmark space to mark packets requiring SNAT with.  Must be within the range [0, 31]. (default 14)
      --iptables-min-sync-period duration            The minimum interval of how often the iptables rules can be refreshed as endpoints and services change (e.g. '5s', '1m', '2h22m').
      --iptables-sync-period duration                The maximum interval of how often iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0. (default 30s)
      --kube-api-burst int32                         Burst to use while talking with kubernetes apiserver (default 10)
      --kube-api-content-type string                 Content type of requests sent to apiserver. (default "application/vnd.kubernetes.protobuf")
      --kube-api-qps float32                         QPS to use while talking with kubernetes apiserver (default 5)
      --kubeconfig string                            Path to kubeconfig file with authorization information (the master location is set by the master flag).
      --masquerade-all                               If using the pure iptables proxy, SNAT everything
      --master string                                The address of the Kubernetes API server (overrides any value in kubeconfig)
      --oom-score-adj int32                          The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000] (default -999)
      --proxy-mode ProxyMode                         Which proxy mode to use: 'userspace' (older) or 'iptables' (faster). If blank, use the best-available proxy (currently iptables).  If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are insufficient, this always falls back to the userspace proxy.
      --proxy-port-range port-range                  Range of host ports (beginPort-endPort, inclusive) that may be consumed in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.
      --udp-timeout duration                         How long an idle UDP connection will be kept open (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxy-mode=userspace (default 250ms)
```

###### Auto generated by spf13/cobra on 21-Feb-2017
