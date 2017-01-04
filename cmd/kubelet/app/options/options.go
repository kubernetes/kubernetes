/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package options contains all of the primary arguments for a kubelet.
package options

import (
	_ "net/http/pprof"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	utilconfig "k8s.io/kubernetes/pkg/util/config"
	"k8s.io/kubernetes/pkg/util/flag"
	utiltaints "k8s.io/kubernetes/pkg/util/taints"

	"github.com/spf13/pflag"
)

const (
	DefaultKubeletPodsDirName       = "pods"
	DefaultKubeletVolumesDirName    = "volumes"
	DefaultKubeletPluginsDirName    = "plugins"
	DefaultKubeletContainersDirName = "containers"
)

// KubeletServer encapsulates all of the parameters necessary for starting up
// a kubelet. These can either be set via command line or directly.
type KubeletServer struct {
	componentconfig.KubeletConfiguration

	KubeConfig          flag.StringFlag
	BootstrapKubeconfig string

	// If true, an invalid KubeConfig will result in the Kubelet exiting with an error.
	RequireKubeConfig bool
	AuthPath          flag.StringFlag // Deprecated -- use KubeConfig instead
	APIServerList     []string        // Deprecated -- use KubeConfig instead

	// Insert a probability of random errors during calls to the master.
	ChaosChance float64
	// Crash immediately, rather than eating panics.
	ReallyCrashForTesting bool

	// TODO(mtaufen): It is increasingly looking like nobody actually uses the
	//                Kubelet's runonce mode anymore, so it may be a candidate
	//                for deprecation and removal.
	// If runOnce is true, the Kubelet will check the API server once for pods,
	// run those in addition to the pods specified by the local manifest, and exit.
	RunOnce bool
}

// NewKubeletServer will create a new KubeletServer with default values.
func NewKubeletServer() *KubeletServer {
	versioned := &v1alpha1.KubeletConfiguration{}
	api.Scheme.Default(versioned)
	config := componentconfig.KubeletConfiguration{}
	api.Scheme.Convert(versioned, &config, nil)
	return &KubeletServer{
		KubeConfig:           flag.NewStringFlag("/var/lib/kubelet/kubeconfig"),
		RequireKubeConfig:    false, // in 1.5, default to true
		KubeletConfiguration: config,
	}
}

// AddFlags adds flags for a specific KubeletServer to the specified FlagSet
func (s *KubeletServer) AddFlags(fs *pflag.FlagSet) {
	// TODO(#34726:1.8.0): Remove the opt-in for failing when swap is enabled.
	fs.BoolVar(&s.ExperimentalFailSwapOn, "experimental-fail-swap-on", s.ExperimentalFailSwapOn, "Makes the Kubelet fail to start if swap is enabled on the node. This is a temporary opton to maintain legacy behavior, failing due to swap enabled will happen by default in v1.6.")

	fs.Var(&s.KubeConfig, "kubeconfig", "Path to a kubeconfig file, specifying how to connect to the API server. --api-servers will be used for the location unless --require-kubeconfig is set.")
	fs.BoolVar(&s.RequireKubeConfig, "require-kubeconfig", s.RequireKubeConfig, "If true the Kubelet will exit if there are configuration errors, and will ignore the value of --api-servers in favor of the server defined in the kubeconfig file.")

	// DEPRECATED: Remove these flags at the beginning of 1.5.
	fs.Var(&s.AuthPath, "auth-path", "Path to .kubernetes_auth file, specifying how to authenticate to API server.")
	fs.MarkDeprecated("auth-path", "will be removed in a future version")
	fs.StringSliceVar(&s.APIServerList, "api-servers", []string{}, "List of Kubernetes API servers for publishing events, and reading pods and services. (ip:port), comma separated.")
	fs.MarkDeprecated("api-servers", "Use --kubeconfig instead. Will be removed in a future version.")

	fs.StringVar(&s.PodManifestPath, "config", s.PodManifestPath, "Path to to the directory containing pod manifest files to run, or the path to a single pod manifest file.")
	fs.MarkDeprecated("config", "Use --pod-manifest-path instead. Will be removed in a future version.")
	fs.StringVar(&s.PodManifestPath, "pod-manifest-path", s.PodManifestPath, "Path to to the directory containing pod manifest files to run, or the path to a single pod manifest file.")
	fs.DurationVar(&s.SyncFrequency.Duration, "sync-frequency", s.SyncFrequency.Duration, "Max period between synchronizing running containers and config")
	fs.DurationVar(&s.FileCheckFrequency.Duration, "file-check-frequency", s.FileCheckFrequency.Duration, "Duration between checking config files for new data")
	fs.DurationVar(&s.HTTPCheckFrequency.Duration, "http-check-frequency", s.HTTPCheckFrequency.Duration, "Duration between checking http for new data")
	fs.StringVar(&s.ManifestURL, "manifest-url", s.ManifestURL, "URL for accessing the container manifest")
	fs.StringVar(&s.ManifestURLHeader, "manifest-url-header", s.ManifestURLHeader, "HTTP header to use when accessing the manifest URL, with the key separated from the value with a ':', as in 'key:value'")
	fs.BoolVar(&s.EnableServer, "enable-server", s.EnableServer, "Enable the Kubelet's server")
	fs.Var(componentconfig.IPVar{Val: &s.Address}, "address", "The IP address for the Kubelet to serve on (set to 0.0.0.0 for all interfaces)")
	fs.Int32Var(&s.Port, "port", s.Port, "The port for the Kubelet to serve on.")
	fs.Int32Var(&s.ReadOnlyPort, "read-only-port", s.ReadOnlyPort, "The read-only port for the Kubelet to serve on with no authentication/authorization (set to 0 to disable)")

	// Authentication
	fs.BoolVar(&s.Authentication.Anonymous.Enabled, "anonymous-auth", s.Authentication.Anonymous.Enabled, ""+
		"Enables anonymous requests to the Kubelet server. Requests that are not rejected by another "+
		"authentication method are treated as anonymous requests. Anonymous requests have a username "+
		"of system:anonymous, and a group name of system:unauthenticated.")
	fs.BoolVar(&s.Authentication.Webhook.Enabled, "authentication-token-webhook", s.Authentication.Webhook.Enabled, ""+
		"Use the TokenReview API to determine authentication for bearer tokens.")
	fs.DurationVar(&s.Authentication.Webhook.CacheTTL.Duration, "authentication-token-webhook-cache-ttl", s.Authentication.Webhook.CacheTTL.Duration, ""+
		"The duration to cache responses from the webhook token authenticator.")
	fs.StringVar(&s.Authentication.X509.ClientCAFile, "client-ca-file", s.Authentication.X509.ClientCAFile, ""+
		"If set, any request presenting a client certificate signed by one of the authorities in the client-ca-file "+
		"is authenticated with an identity corresponding to the CommonName of the client certificate.")

	// Authorization
	fs.StringVar((*string)(&s.Authorization.Mode), "authorization-mode", string(s.Authorization.Mode), ""+
		"Authorization mode for Kubelet server. Valid options are AlwaysAllow or Webhook. "+
		"Webhook mode uses the SubjectAccessReview API to determine authorization.")
	fs.DurationVar(&s.Authorization.Webhook.CacheAuthorizedTTL.Duration, "authorization-webhook-cache-authorized-ttl", s.Authorization.Webhook.CacheAuthorizedTTL.Duration, ""+
		"The duration to cache 'authorized' responses from the webhook authorizer.")
	fs.DurationVar(&s.Authorization.Webhook.CacheUnauthorizedTTL.Duration, "authorization-webhook-cache-unauthorized-ttl", s.Authorization.Webhook.CacheUnauthorizedTTL.Duration, ""+
		"The duration to cache 'unauthorized' responses from the webhook authorizer.")

	fs.StringVar(&s.TLSCertFile, "tls-cert-file", s.TLSCertFile, ""+
		"File containing x509 Certificate for HTTPS.  (CA cert, if any, concatenated after server cert). "+
		"If --tls-cert-file and --tls-private-key-file are not provided, a self-signed certificate and key "+
		"are generated for the public address and saved to the directory passed to --cert-dir.")
	fs.StringVar(&s.TLSPrivateKeyFile, "tls-private-key-file", s.TLSPrivateKeyFile, "File containing x509 private key matching --tls-cert-file.")
	fs.StringVar(&s.CertDirectory, "cert-dir", s.CertDirectory, "The directory where the TLS certs are located (by default /var/run/kubernetes). "+
		"If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored.")
	fs.StringVar(&s.BootstrapKubeconfig, "experimental-bootstrap-kubeconfig", s.BootstrapKubeconfig, "<Warning: Experimental feature> Path to a kubeconfig file that will be used to get client certificate for kubelet. "+
		"If the file specified by --kubeconfig does not exist, the bootstrap kubeconfig is used to request a client certificate from the API server. "+
		"On success, a kubeconfig file referencing the generated key and obtained certificate is written to the path specified by --kubeconfig. "+
		"The certificate and key file will be stored in the directory pointed by --cert-dir.")
	fs.StringVar(&s.HostnameOverride, "hostname-override", s.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.StringVar(&s.PodInfraContainerImage, "pod-infra-container-image", s.PodInfraContainerImage, "The image whose network/ipc namespaces containers in each pod will use.")
	fs.StringVar(&s.DockerEndpoint, "docker-endpoint", s.DockerEndpoint, "Use this for the docker endpoint to communicate with")
	fs.StringVar(&s.RootDirectory, "root-dir", s.RootDirectory, "Directory path for managing kubelet files (volume mounts,etc).")
	fs.StringVar(&s.SeccompProfileRoot, "seccomp-profile-root", s.SeccompProfileRoot, "Directory path for seccomp profiles.")
	fs.BoolVar(&s.AllowPrivileged, "allow-privileged", s.AllowPrivileged, "If true, allow containers to request privileged mode. [default=false]")
	fs.StringSliceVar(&s.HostNetworkSources, "host-network-sources", s.HostNetworkSources, "Comma-separated list of sources from which the Kubelet allows pods to use of host network. [default=\"*\"]")
	fs.StringSliceVar(&s.HostPIDSources, "host-pid-sources", s.HostPIDSources, "Comma-separated list of sources from which the Kubelet allows pods to use the host pid namespace. [default=\"*\"]")
	fs.StringSliceVar(&s.HostIPCSources, "host-ipc-sources", s.HostIPCSources, "Comma-separated list of sources from which the Kubelet allows pods to use the host ipc namespace. [default=\"*\"]")
	fs.Int32Var(&s.RegistryPullQPS, "registry-qps", s.RegistryPullQPS, "If > 0, limit registry pull QPS to this value.  If 0, unlimited. [default=5.0]")
	fs.Int32Var(&s.RegistryBurst, "registry-burst", s.RegistryBurst, "Maximum size of a bursty pulls, temporarily allows pulls to burst to this number, while still not exceeding registry-qps.  Only used if --registry-qps > 0")
	fs.Int32Var(&s.EventRecordQPS, "event-qps", s.EventRecordQPS, "If > 0, limit event creations per second to this value. If 0, unlimited.")
	fs.Int32Var(&s.EventBurst, "event-burst", s.EventBurst, "Maximum size of a bursty event records, temporarily allows event records to burst to this number, while still not exceeding event-qps. Only used if --event-qps > 0")
	fs.BoolVar(&s.RunOnce, "runonce", s.RunOnce, "If true, exit after spawning pods from local manifests or remote urls. Exclusive with --api-servers, and --enable-server")
	fs.BoolVar(&s.EnableDebuggingHandlers, "enable-debugging-handlers", s.EnableDebuggingHandlers, "Enables server endpoints for log collection and local running of containers and commands")
	fs.DurationVar(&s.MinimumGCAge.Duration, "minimum-container-ttl-duration", s.MinimumGCAge.Duration, "Minimum age for a finished container before it is garbage collected.  Examples: '300ms', '10s' or '2h45m'")
	fs.MarkDeprecated("minimum-container-ttl-duration", "Use --eviction-hard or --eviction-soft instead. Will be removed in a future version.")
	fs.Int32Var(&s.MaxPerPodContainerCount, "maximum-dead-containers-per-container", s.MaxPerPodContainerCount, "Maximum number of old instances to retain per container.  Each container takes up some disk space.  Default: 1.")
	fs.MarkDeprecated("maximum-dead-containers-per-container", "Use --eviction-hard or --eviction-soft instead. Will be removed in a future version.")
	fs.Int32Var(&s.MaxContainerCount, "maximum-dead-containers", s.MaxContainerCount, "Maximum number of old instances of containers to retain globally.  Each container takes up some disk space. To disable, set to a negative number.  Default: -1.")
	fs.MarkDeprecated("maximum-dead-containers", "Use --eviction-hard or --eviction-soft instead. Will be removed in a future version.")
	fs.Int32Var(&s.CAdvisorPort, "cadvisor-port", s.CAdvisorPort, "The port of the localhost cAdvisor endpoint")
	fs.Int32Var(&s.HealthzPort, "healthz-port", s.HealthzPort, "The port of the localhost healthz endpoint")
	fs.Var(componentconfig.IPVar{Val: &s.HealthzBindAddress}, "healthz-bind-address", "The IP address for the healthz server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)")
	fs.Int32Var(&s.OOMScoreAdj, "oom-score-adj", s.OOMScoreAdj, "The oom-score-adj value for kubelet process. Values must be within the range [-1000, 1000]")
	fs.BoolVar(&s.RegisterNode, "register-node", s.RegisterNode, "Register the node with the apiserver (defaults to true if --api-servers is set)")
	fs.StringVar(&s.ClusterDomain, "cluster-domain", s.ClusterDomain, "Domain for this cluster.  If set, kubelet will configure all containers to search this domain in addition to the host's search domains")
	fs.StringVar(&s.MasterServiceNamespace, "master-service-namespace", s.MasterServiceNamespace, "The namespace from which the kubernetes master services should be injected into pods")
	fs.StringVar(&s.ClusterDNS, "cluster-dns", s.ClusterDNS, "IP address for a cluster DNS server.  This value is used for containers' DNS server in case of Pods with \"dnsPolicy=ClusterFirst\"")
	fs.DurationVar(&s.StreamingConnectionIdleTimeout.Duration, "streaming-connection-idle-timeout", s.StreamingConnectionIdleTimeout.Duration, "Maximum time a streaming connection can be idle before the connection is automatically closed. 0 indicates no timeout. Example: '5m'")
	fs.DurationVar(&s.NodeStatusUpdateFrequency.Duration, "node-status-update-frequency", s.NodeStatusUpdateFrequency.Duration, "Specifies how often kubelet posts node status to master. Note: be cautious when changing the constant, it must work with nodeMonitorGracePeriod in nodecontroller. Default: 10s")
	s.NodeLabels = make(map[string]string)
	bindableNodeLabels := utilconfig.ConfigurationMap(s.NodeLabels)
	fs.Var(&bindableNodeLabels, "node-labels", "<Warning: Alpha feature> Labels to add when registering the node in the cluster.  Labels must be key=value pairs separated by ','.")
	fs.DurationVar(&s.ImageMinimumGCAge.Duration, "minimum-image-ttl-duration", s.ImageMinimumGCAge.Duration, "Minimum age for an unused image before it is garbage collected.  Examples: '300ms', '10s' or '2h45m'. Default: '2m'")
	fs.Int32Var(&s.ImageGCHighThresholdPercent, "image-gc-high-threshold", s.ImageGCHighThresholdPercent, "The percent of disk usage after which image garbage collection is always run. Default: 90%")
	fs.Int32Var(&s.ImageGCLowThresholdPercent, "image-gc-low-threshold", s.ImageGCLowThresholdPercent, "The percent of disk usage before which image garbage collection is never run. Lowest disk usage to garbage collect to. Default: 80%")
	fs.Int32Var(&s.LowDiskSpaceThresholdMB, "low-diskspace-threshold-mb", s.LowDiskSpaceThresholdMB, "The absolute free disk space, in MB, to maintain. When disk space falls below this threshold, new pods would be rejected. Default: 256")
	fs.DurationVar(&s.VolumeStatsAggPeriod.Duration, "volume-stats-agg-period", s.VolumeStatsAggPeriod.Duration, "Specifies interval for kubelet to calculate and cache the volume disk usage for all pods and volumes.  To disable volume calculations, set to 0.  Default: '1m'")
	fs.StringVar(&s.NetworkPluginName, "network-plugin", s.NetworkPluginName, "<Warning: Alpha feature> The name of the network plugin to be invoked for various events in kubelet/pod lifecycle")
	fs.StringVar(&s.NetworkPluginDir, "network-plugin-dir", s.NetworkPluginDir, "<Warning: Alpha feature> The full path of the directory in which to search for network plugins or CNI config")
	fs.StringVar(&s.CNIConfDir, "cni-conf-dir", s.CNIConfDir, "<Warning: Alpha feature> The full path of the directory in which to search for CNI config files. Default: /etc/cni/net.d")
	fs.StringVar(&s.CNIBinDir, "cni-bin-dir", s.CNIBinDir, "<Warning: Alpha feature> The full path of the directory in which to search for CNI plugin binaries. Default: /opt/cni/bin")
	fs.Int32Var(&s.NetworkPluginMTU, "network-plugin-mtu", s.NetworkPluginMTU, "<Warning: Alpha feature> The MTU to be passed to the network plugin, to override the default. Set to 0 to use the default 1460 MTU.")
	fs.StringVar(&s.VolumePluginDir, "volume-plugin-dir", s.VolumePluginDir, "<Warning: Alpha feature> The full path of the directory in which to search for additional third party volume plugins")
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider, "The provider for cloud services. By default, kubelet will attempt to auto-detect the cloud provider. Specify empty string for running with no cloud provider. [default=auto-detect]")
	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	fs.StringVar(&s.FeatureGates, "feature-gates", s.FeatureGates, "A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(utilconfig.DefaultFeatureGate.KnownFeatures(), "\n"))

	fs.StringVar(&s.KubeletCgroups, "resource-container", s.KubeletCgroups, "Optional absolute name of the resource-only container to create and run the Kubelet in.")
	fs.MarkDeprecated("resource-container", "Use --kubelet-cgroups instead. Will be removed in a future version.")
	fs.StringVar(&s.KubeletCgroups, "kubelet-cgroups", s.KubeletCgroups, "Optional absolute name of cgroups to create and run the Kubelet in.")

	fs.StringVar(&s.SystemCgroups, "system-container", s.SystemCgroups, "Optional resource-only container in which to place all non-kernel processes that are not already in a container. Empty for no container. Rolling back the flag requires a reboot. (Default: \"\").")
	fs.MarkDeprecated("system-container", "Use --system-cgroups instead. Will be removed in a future version.")
	fs.StringVar(&s.SystemCgroups, "system-cgroups", s.SystemCgroups, "Optional absolute name of cgroups in which to place all non-kernel processes that are not already inside a cgroup under `/`. Empty for no container. Rolling back the flag requires a reboot. (Default: \"\").")

	fs.BoolVar(&s.ExperimentalCgroupsPerQOS, "experimental-cgroups-per-qos", s.ExperimentalCgroupsPerQOS, "Enable creation of QoS cgroup hierarchy, if true top level QoS and pod cgroups are created.")
	fs.StringVar(&s.CgroupDriver, "cgroup-driver", s.CgroupDriver, "Driver that the kubelet uses to manipulate cgroups on the host.  Possible values: 'cgroupfs', 'systemd'")
	fs.StringVar(&s.CgroupRoot, "cgroup-root", s.CgroupRoot, "Optional root cgroup to use for pods. This is handled by the container runtime on a best effort basis. Default: '', which means use the container runtime default.")
	fs.StringVar(&s.ContainerRuntime, "container-runtime", s.ContainerRuntime, "The container runtime to use. Possible values: 'docker', 'rkt'. Default: 'docker'.")
	fs.DurationVar(&s.RuntimeRequestTimeout.Duration, "runtime-request-timeout", s.RuntimeRequestTimeout.Duration, "Timeout of all runtime requests except long running request - pull, logs, exec and attach. When timeout exceeded, kubelet will cancel the request, throw out an error and retry later. Default: 2m0s")
	fs.DurationVar(&s.ImagePullProgressDeadline.Duration, "image-pull-progress-deadline", s.ImagePullProgressDeadline.Duration, "If no pulling progress is made before this deadline, the image pulling will be cancelled. Default: 1m0s.")
	fs.StringVar(&s.LockFilePath, "lock-file", s.LockFilePath, "<Warning: Alpha feature> The path to file for kubelet to use as a lock file.")
	fs.BoolVar(&s.ExitOnLockContention, "exit-on-lock-contention", s.ExitOnLockContention, "Whether kubelet should exit upon lock-file contention.")
	fs.StringVar(&s.RktPath, "rkt-path", s.RktPath, "Path of rkt binary. Leave empty to use the first rkt in $PATH.  Only used if --container-runtime='rkt'.")
	fs.StringVar(&s.ExperimentalMounterPath, "experimental-mounter-path", s.ExperimentalMounterPath, "[Experimental] Path of mounter binary. Leave empty to use the default mount.")
	fs.StringVar(&s.RktAPIEndpoint, "rkt-api-endpoint", s.RktAPIEndpoint, "The endpoint of the rkt API service to communicate with. Only used if --container-runtime='rkt'.")
	fs.StringVar(&s.RktStage1Image, "rkt-stage1-image", s.RktStage1Image, "image to use as stage1. Local paths and http/https URLs are supported. If empty, the 'stage1.aci' in the same directory as '--rkt-path' will be used.")
	fs.MarkDeprecated("rkt-stage1-image", "Will be removed in a future version. The default stage1 image will be specified by the rkt configurations, see https://github.com/coreos/rkt/blob/master/Documentation/configuration.md for more details.")
	fs.StringVar(&s.HairpinMode, "hairpin-mode", s.HairpinMode, "How should the kubelet setup hairpin NAT. This allows endpoints of a Service to loadbalance back to themselves if they should try to access their own Service. Valid values are \"promiscuous-bridge\", \"hairpin-veth\" and \"none\".")
	fs.BoolVar(&s.BabysitDaemons, "babysit-daemons", s.BabysitDaemons, "If true, the node has babysitter process monitoring docker and kubelet.")
	fs.MarkDeprecated("babysit-daemons", "Will be removed in a future version.")
	fs.Int32Var(&s.MaxPods, "max-pods", s.MaxPods, "Number of Pods that can run on this Kubelet.")
	fs.Int32Var(&s.NvidiaGPUs, "experimental-nvidia-gpus", s.NvidiaGPUs, "Number of NVIDIA GPU devices on this node. Only 0 (default) and 1 are currently supported.")
	fs.StringVar(&s.DockerExecHandlerName, "docker-exec-handler", s.DockerExecHandlerName, "Handler to use when executing a command in a container. Valid values are 'native' and 'nsenter'. Defaults to 'native'.")
	fs.StringVar(&s.NonMasqueradeCIDR, "non-masquerade-cidr", s.NonMasqueradeCIDR, "Traffic to IPs outside this range will use IP masquerade.")
	fs.StringVar(&s.PodCIDR, "pod-cidr", "", "The CIDR to use for pod IP addresses, only used in standalone mode.  In cluster mode, this is obtained from the master.")
	fs.StringVar(&s.ResolverConfig, "resolv-conf", s.ResolverConfig, "Resolver configuration file used as the basis for the container DNS resolution configuration.")
	fs.BoolVar(&s.CPUCFSQuota, "cpu-cfs-quota", s.CPUCFSQuota, "Enable CPU CFS quota enforcement for containers that specify CPU limits")
	fs.BoolVar(&s.EnableControllerAttachDetach, "enable-controller-attach-detach", s.EnableControllerAttachDetach, "Enables the Attach/Detach controller to manage attachment/detachment of volumes scheduled to this node, and disables kubelet from executing any attach/detach operations")
	fs.BoolVar(&s.MakeIPTablesUtilChains, "make-iptables-util-chains", s.MakeIPTablesUtilChains, "If true, kubelet will ensure iptables utility rules are present on host.")
	fs.Int32Var(&s.IPTablesMasqueradeBit, "iptables-masquerade-bit", s.IPTablesMasqueradeBit, "The bit of the fwmark space to mark packets for SNAT. Must be within the range [0, 31]. Please match this parameter with corresponding parameter in kube-proxy.")
	fs.Int32Var(&s.IPTablesDropBit, "iptables-drop-bit", s.IPTablesDropBit, "The bit of the fwmark space to mark packets for dropping. Must be within the range [0, 31].")
	fs.StringSliceVar(&s.AllowedUnsafeSysctls, "experimental-allowed-unsafe-sysctls", s.AllowedUnsafeSysctls, "Comma-separated whitelist of unsafe sysctls or unsafe sysctl patterns (ending in *). Use these at your own risk.")

	// Flags intended for testing, not recommended used in production environments.
	fs.BoolVar(&s.ReallyCrashForTesting, "really-crash-for-testing", s.ReallyCrashForTesting, "If true, when panics occur crash. Intended for testing.")
	fs.Float64Var(&s.ChaosChance, "chaos-chance", s.ChaosChance, "If > 0.0, introduce random client errors and latency. Intended for testing. [default=0.0]")
	fs.BoolVar(&s.Containerized, "containerized", s.Containerized, "Experimental support for running kubelet in a container.  Intended for testing. [default=false]")
	fs.Int64Var(&s.MaxOpenFiles, "max-open-files", s.MaxOpenFiles, "Number of files that can be opened by Kubelet process. [default=1000000]")
	fs.Var(&s.SystemReserved, "system-reserved", "A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs that describe resources reserved for non-kubernetes components. Currently only cpu and memory are supported. See http://kubernetes.io/docs/user-guide/compute-resources for more detail. [default=none]")
	fs.Var(&s.KubeReserved, "kube-reserved", "A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs that describe resources reserved for kubernetes system components. Currently only cpu and memory are supported. See http://kubernetes.io/docs/user-guide/compute-resources for more detail. [default=none]")
	fs.BoolVar(&s.RegisterSchedulable, "register-schedulable", s.RegisterSchedulable, "Register the node as schedulable. Won't have any effect if register-node is false. [default=true]")
	fs.MarkDeprecated("register-schedulable", "will be removed in a future version")
	fs.Var(utiltaints.NewTaintsVar(&s.RegisterWithTaints), "register-with-taints", "Register the node with the given list of taints (comma seperated \"<key>=<value>:<effect>\"). No-op if register-node is false.")
	fs.StringVar(&s.ContentType, "kube-api-content-type", s.ContentType, "Content type of requests sent to apiserver.")
	fs.Int32Var(&s.KubeAPIQPS, "kube-api-qps", s.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&s.KubeAPIBurst, "kube-api-burst", s.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver")
	fs.BoolVar(&s.SerializeImagePulls, "serialize-image-pulls", s.SerializeImagePulls, "Pull images one at a time. We recommend *not* changing the default value on nodes that run docker daemon with version < 1.9 or an Aufs storage backend. Issue #10959 has more details. [default=true]")
	fs.DurationVar(&s.OutOfDiskTransitionFrequency.Duration, "outofdisk-transition-frequency", s.OutOfDiskTransitionFrequency.Duration, "Duration for which the kubelet has to wait before transitioning out of out-of-disk node condition status. Default: 5m0s")
	fs.StringVar(&s.NodeIP, "node-ip", s.NodeIP, "IP address of the node. If set, kubelet will use this IP address for the node")
	fs.BoolVar(&s.EnableCustomMetrics, "enable-custom-metrics", s.EnableCustomMetrics, "Support for gathering custom metrics.")
	fs.StringVar(&s.RuntimeCgroups, "runtime-cgroups", s.RuntimeCgroups, "Optional absolute name of cgroups to create and run the runtime in.")
	fs.StringVar(&s.EvictionHard, "eviction-hard", s.EvictionHard, "A set of eviction thresholds (e.g. memory.available<1Gi) that if met would trigger a pod eviction.")
	fs.StringVar(&s.EvictionSoft, "eviction-soft", s.EvictionSoft, "A set of eviction thresholds (e.g. memory.available<1.5Gi) that if met over a corresponding grace period would trigger a pod eviction.")
	fs.StringVar(&s.EvictionSoftGracePeriod, "eviction-soft-grace-period", s.EvictionSoftGracePeriod, "A set of eviction grace periods (e.g. memory.available=1m30s) that correspond to how long a soft eviction threshold must hold before triggering a pod eviction.")
	fs.DurationVar(&s.EvictionPressureTransitionPeriod.Duration, "eviction-pressure-transition-period", s.EvictionPressureTransitionPeriod.Duration, "Duration for which the kubelet has to wait before transitioning out of an eviction pressure condition.")
	fs.Int32Var(&s.EvictionMaxPodGracePeriod, "eviction-max-pod-grace-period", s.EvictionMaxPodGracePeriod, "Maximum allowed grace period (in seconds) to use when terminating pods in response to a soft eviction threshold being met.  If negative, defer to pod specified value.")
	fs.StringVar(&s.EvictionMinimumReclaim, "eviction-minimum-reclaim", s.EvictionMinimumReclaim, "A set of minimum reclaims (e.g. imagefs.available=2Gi) that describes the minimum amount of resource the kubelet will reclaim when performing a pod eviction if that resource is under pressure.")
	fs.BoolVar(&s.ExperimentalKernelMemcgNotification, "experimental-kernel-memcg-notification", s.ExperimentalKernelMemcgNotification, "If enabled, the kubelet will integrate with the kernel memcg notification to determine if memory eviction thresholds are crossed rather than polling.")
	fs.Int32Var(&s.PodsPerCore, "pods-per-core", s.PodsPerCore, "Number of Pods per core that can run on this Kubelet. The total number of Pods on this Kubelet cannot exceed max-pods, so max-pods will be used if this calculation results in a larger number of Pods allowed on the Kubelet. A value of 0 disables this limit.")
	fs.BoolVar(&s.ProtectKernelDefaults, "protect-kernel-defaults", s.ProtectKernelDefaults, "Default kubelet behaviour for kernel tuning. If set, kubelet errors if any of kernel tunables is different than kubelet defaults.")

	// CRI flags.
	fs.BoolVar(&s.EnableCRI, "experimental-cri", s.EnableCRI, "[Experimental] Enable the Container Runtime Interface (CRI) integration. If --container-runtime is set to \"remote\", Kubelet will communicate with the runtime/image CRI server listening on the endpoint specified by --remote-runtime-endpoint/--remote-image-endpoint. If --container-runtime is set to \"docker\", Kubelet will launch a in-process CRI server on behalf of docker, and communicate over a default endpoint.")
	fs.StringVar(&s.RemoteRuntimeEndpoint, "container-runtime-endpoint", s.RemoteRuntimeEndpoint, "[Experimental] The unix socket endpoint of remote runtime service. The endpoint is used only when CRI integration is enabled (--experimental-cri)")
	fs.StringVar(&s.RemoteImageEndpoint, "image-service-endpoint", s.RemoteImageEndpoint, "[Experimental] The unix socket endpoint of remote image service. If not specified, it will be the same with container-runtime-endpoint by default. The endpoint is used only when CRI integration is enabled (--experimental-cri)")

	fs.BoolVar(&s.ExperimentalCheckNodeCapabilitiesBeforeMount, "experimental-check-node-capabilities-before-mount", s.ExperimentalCheckNodeCapabilitiesBeforeMount, "[Experimental] if set true, the kubelet will check the underlying node for required componenets (binaries, etc.) before performing the mount")
}
