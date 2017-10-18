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
	"fmt"
	_ "net/http/pprof"
	"runtime"
	"strings"

	"github.com/spf13/pflag"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1alpha1"
	kubeletconfigvalidation "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/validation"
	"k8s.io/kubernetes/pkg/kubelet/config"
	utiltaints "k8s.io/kubernetes/pkg/util/taints"
)

// A configuration field should go in KubeletFlags instead of KubeletConfiguration if any of these are true:
// - its value will never, or cannot safely be changed during the lifetime of a node
// - its value cannot be safely shared between nodes at the same time (e.g. a hostname)
//   KubeletConfiguration is intended to be shared between nodes
// In general, please try to avoid adding flags or configuration fields,
// we already have a confusingly large amount of them.
type KubeletFlags struct {
	KubeConfig          flag.StringFlag
	BootstrapKubeconfig string
	RotateCertificates  bool

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

	// HostnameOverride is the hostname used to identify the kubelet instead
	// of the actual hostname.
	HostnameOverride string
	// NodeIP is IP address of the node.
	// If set, kubelet will use this IP address for the node.
	NodeIP string

	// This flag, if set, sets the unique id of the instance that an external provider (i.e. cloudprovider)
	// can use to identify a specific node
	ProviderID string

	// Container-runtime-specific options.
	config.ContainerRuntimeOptions

	// certDirectory is the directory where the TLS certs are located (by
	// default /var/run/kubernetes). If tlsCertFile and tlsPrivateKeyFile
	// are provided, this flag will be ignored.
	CertDirectory string

	// cloudProvider is the provider for cloud services.
	// +optional
	CloudProvider string

	// cloudConfigFile is the path to the cloud provider configuration file.
	// +optional
	CloudConfigFile string

	// rootDirectory is the directory path to place kubelet files (volume
	// mounts,etc).
	RootDirectory string

	// The Kubelet will use this directory for checkpointing downloaded configurations and tracking configuration health.
	// The Kubelet will create this directory if it does not already exist.
	// The path may be absolute or relative; relative paths are under the Kubelet's current working directory.
	// Providing this flag enables dynamic kubelet configuration.
	// To use this flag, the DynamicKubeletConfig feature gate must be enabled.
	DynamicConfigDir flag.StringFlag

	// The Kubelet will look in this directory for an init configuration.
	// The path may be absolute or relative; relative paths are under the Kubelet's current working directory.
	// Omit this flag to use the combination of built-in default configuration values and flags.
	// To use this flag, the DynamicKubeletConfig feature gate must be enabled.
	InitConfigDir flag.StringFlag

	// EXPERIMENTAL FLAGS
	// Whitelist of unsafe sysctls or sysctl patterns (ending in *).
	// +optional
	AllowedUnsafeSysctls []string
	// containerized should be set to true if kubelet is running in a container.
	Containerized bool
	// remoteRuntimeEndpoint is the endpoint of remote runtime service
	RemoteRuntimeEndpoint string
	// remoteImageEndpoint is the endpoint of remote image service
	RemoteImageEndpoint string
	// experimentalMounterPath is the path of mounter binary. Leave empty to use the default mount path
	ExperimentalMounterPath string
	// If enabled, the kubelet will integrate with the kernel memcg notification to determine if memory eviction thresholds are crossed rather than polling.
	// +optional
	ExperimentalKernelMemcgNotification bool
	// This flag, if set, enables a check prior to mount operations to verify that the required components
	// (binaries, etc.) to mount the volume are available on the underlying node. If the check is enabled
	// and fails the mount operation fails.
	ExperimentalCheckNodeCapabilitiesBeforeMount bool
	// This flag, if set, will avoid including `EvictionHard` limits while computing Node Allocatable.
	// Refer to [Node Allocatable](https://git.k8s.io/community/contributors/design-proposals/node-allocatable.md) doc for more information.
	ExperimentalNodeAllocatableIgnoreEvictionThreshold bool
	// A set of ResourceName=Percentage (e.g. memory=50%) pairs that describe
	// how pod resource requests are reserved at the QoS level.
	// Currently only memory is supported. [default=none]"
	ExperimentalQOSReserved kubeletconfig.ConfigurationMap

	// DEPRECATED FLAGS
	// minimumGCAge is the minimum age for a finished container before it is
	// garbage collected.
	MinimumGCAge metav1.Duration
	// maxPerPodContainerCount is the maximum number of old instances to
	// retain per container. Each container takes up some disk space.
	MaxPerPodContainerCount int32
	// maxContainerCount is the maximum number of old instances of containers
	// to retain globally. Each container takes up some disk space.
	MaxContainerCount int32
	// masterServiceNamespace is The namespace from which the kubernetes
	// master services should be injected into pods.
	MasterServiceNamespace string
	// registerSchedulable tells the kubelet to register the node as
	// schedulable. Won't have any effect if register-node is false.
	// DEPRECATED: use registerWithTaints instead
	RegisterSchedulable bool
	// RequireKubeConfig is deprecated! A valid KubeConfig is now required if --kubeconfig is provided.
	RequireKubeConfig bool
	// nonMasqueradeCIDR configures masquerading: traffic to IPs outside this range will use IP masquerade.
	NonMasqueradeCIDR string
	// This flag, if set, instructs the kubelet to keep volumes from terminated pods mounted to the node.
	// This can be useful for debugging volume related issues.
	KeepTerminatedPodVolumes bool
	// enable gathering custom metrics.
	EnableCustomMetrics bool
}

// NewKubeletFlags will create a new KubeletFlags with default values
func NewKubeletFlags() *KubeletFlags {
	remoteRuntimeEndpoint := ""
	if runtime.GOOS == "linux" {
		remoteRuntimeEndpoint = "unix:///var/run/dockershim.sock"
	} else if runtime.GOOS == "windows" {
		remoteRuntimeEndpoint = "tcp://localhost:3735"
	}

	return &KubeletFlags{
		// TODO(#41161:v1.10.0): Remove the default kubeconfig path and --require-kubeconfig.
		RequireKubeConfig:                   false,
		KubeConfig:                          flag.NewStringFlag("/var/lib/kubelet/kubeconfig"),
		ContainerRuntimeOptions:             *NewContainerRuntimeOptions(),
		CertDirectory:                       "/var/lib/kubelet/pki",
		RootDirectory:                       v1alpha1.DefaultRootDir,
		MasterServiceNamespace:              metav1.NamespaceDefault,
		MaxContainerCount:                   -1,
		MaxPerPodContainerCount:             1,
		MinimumGCAge:                        metav1.Duration{Duration: 0},
		NonMasqueradeCIDR:                   "10.0.0.0/8",
		RegisterSchedulable:                 true,
		ExperimentalKernelMemcgNotification: false,
		ExperimentalQOSReserved:             make(map[string]string),
		RemoteRuntimeEndpoint:               remoteRuntimeEndpoint,
		RotateCertificates:                  false,
		// TODO(#54161:v1.11.0): Remove --enable-custom-metrics flag, it is deprecated.
		EnableCustomMetrics: false,
	}
}

func ValidateKubeletFlags(f *KubeletFlags) error {
	// ensure that nobody sets DynamicConfigDir if the dynamic config feature gate is turned off
	if f.DynamicConfigDir.Provided() && !utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
		return fmt.Errorf("the DynamicKubeletConfig feature gate must be enabled in order to use the --dynamic-config-dir flag")
	}
	// ensure that nobody sets InitConfigDir if the dynamic config feature gate is turned off
	if f.InitConfigDir.Provided() && !utilfeature.DefaultFeatureGate.Enabled(features.KubeletConfigFile) {
		return fmt.Errorf("the KubeletConfigFile feature gate must be enabled in order to use the --init-config-dir flag")
	}
	return nil
}

// NewKubeletConfiguration will create a new KubeletConfiguration with default values
func NewKubeletConfiguration() (*kubeletconfig.KubeletConfiguration, error) {
	scheme, _, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}
	versioned := &v1alpha1.KubeletConfiguration{}
	scheme.Default(versioned)
	config := &kubeletconfig.KubeletConfiguration{}
	if err := scheme.Convert(versioned, config, nil); err != nil {
		return nil, err
	}
	return config, nil
}

// KubeletServer encapsulates all of the parameters necessary for starting up
// a kubelet. These can either be set via command line or directly.
type KubeletServer struct {
	KubeletFlags
	kubeletconfig.KubeletConfiguration
}

// NewKubeletServer will create a new KubeletServer with default values.
func NewKubeletServer() (*KubeletServer, error) {
	config, err := NewKubeletConfiguration()
	if err != nil {
		return nil, err
	}
	return &KubeletServer{
		KubeletFlags:         *NewKubeletFlags(),
		KubeletConfiguration: *config,
	}, nil
}

// validateKubeletServer validates configuration of KubeletServer and returns an error if the input configuration is invalid
func ValidateKubeletServer(s *KubeletServer) error {
	// please add any KubeletConfiguration validation to the kubeletconfigvalidation.ValidateKubeletConfiguration function
	if err := kubeletconfigvalidation.ValidateKubeletConfiguration(&s.KubeletConfiguration); err != nil {
		return err
	}
	if err := ValidateKubeletFlags(&s.KubeletFlags); err != nil {
		return err
	}
	return nil
}

// AddFlags adds flags for a specific KubeletServer to the specified FlagSet
func (s *KubeletServer) AddFlags(fs *pflag.FlagSet) {
	s.KubeletFlags.AddFlags(fs)
	AddKubeletConfigFlags(fs, &s.KubeletConfiguration)
}

// AddFlags adds flags for a specific KubeletFlags to the specified FlagSet
func (f *KubeletFlags) AddFlags(fs *pflag.FlagSet) {
	f.ContainerRuntimeOptions.AddFlags(fs)

	fs.Var(&f.KubeConfig, "kubeconfig", "Path to a kubeconfig file, specifying how to connect to the API server.")
	// TODO(#41161:v1.10.0): Remove the default kubeconfig path and --require-kubeconfig.
	fs.BoolVar(&f.RequireKubeConfig, "require-kubeconfig", f.RequireKubeConfig, "This flag is no longer necessary. It has been deprecated and will be removed in a future version.")
	fs.MarkDeprecated("require-kubeconfig", "You no longer need to use --require-kubeconfig. This will be removed in a future version. Providing --kubeconfig enables API server mode, omitting --kubeconfig enables standalone mode unless --require-kubeconfig=true is also set. In the latter case, the legacy default kubeconfig path will be used until --require-kubeconfig is removed.")

	fs.MarkDeprecated("experimental-bootstrap-kubeconfig", "Use --bootstrap-kubeconfig")
	fs.StringVar(&f.BootstrapKubeconfig, "experimental-bootstrap-kubeconfig", f.BootstrapKubeconfig, "deprecated: use --bootstrap-kubeconfig")
	fs.StringVar(&f.BootstrapKubeconfig, "bootstrap-kubeconfig", f.BootstrapKubeconfig, "Path to a kubeconfig file that will be used to get client certificate for kubelet. "+
		"If the file specified by --kubeconfig does not exist, the bootstrap kubeconfig is used to request a client certificate from the API server. "+
		"On success, a kubeconfig file referencing the generated client certificate and key is written to the path specified by --kubeconfig. "+
		"The client certificate and key file will be stored in the directory pointed by --cert-dir.")
	fs.BoolVar(&f.RotateCertificates, "rotate-certificates", f.RotateCertificates, "<Warning: Beta feature> Auto rotate the kubelet client certificates by requesting new certificates from the kube-apiserver when the certificate expiration approaches.")

	fs.BoolVar(&f.ReallyCrashForTesting, "really-crash-for-testing", f.ReallyCrashForTesting, "If true, when panics occur crash. Intended for testing.")
	fs.Float64Var(&f.ChaosChance, "chaos-chance", f.ChaosChance, "If > 0.0, introduce random client errors and latency. Intended for testing.")

	fs.BoolVar(&f.RunOnce, "runonce", f.RunOnce, "If true, exit after spawning pods from local manifests or remote urls. Exclusive with --enable-server")

	fs.StringVar(&f.HostnameOverride, "hostname-override", f.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")

	fs.StringVar(&f.NodeIP, "node-ip", f.NodeIP, "IP address of the node. If set, kubelet will use this IP address for the node")

	fs.StringVar(&f.ProviderID, "provider-id", f.ProviderID, "Unique identifier for identifying the node in a machine database, i.e cloudprovider")

	fs.StringVar(&f.CertDirectory, "cert-dir", f.CertDirectory, "The directory where the TLS certs are located. "+
		"If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored.")

	fs.StringVar(&f.CloudProvider, "cloud-provider", f.CloudProvider, "The provider for cloud services. Specify empty string for running with no cloud provider.")
	fs.StringVar(&f.CloudConfigFile, "cloud-config", f.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")

	fs.StringVar(&f.RootDirectory, "root-dir", f.RootDirectory, "Directory path for managing kubelet files (volume mounts,etc).")

	fs.Var(&f.DynamicConfigDir, "dynamic-config-dir", "The Kubelet will use this directory for checkpointing downloaded configurations and tracking configuration health. The Kubelet will create this directory if it does not already exist. The path may be absolute or relative; relative paths start at the Kubelet's current working directory. Providing this flag enables dynamic Kubelet configuration. Presently, you must also enable the DynamicKubeletConfig feature gate to pass this flag.")
	fs.Var(&f.InitConfigDir, "init-config-dir", "The Kubelet will look in this directory for the init configuration. The path may be absolute or relative; relative paths start at the Kubelet's current working directory. Omit this argument to use the built-in default configuration values. Presently, you must also enable the DynamicKubeletConfig feature gate to pass this flag.")

	// EXPERIMENTAL FLAGS
	fs.StringVar(&f.ExperimentalMounterPath, "experimental-mounter-path", f.ExperimentalMounterPath, "[Experimental] Path of mounter binary. Leave empty to use the default mount.")
	fs.StringSliceVar(&f.AllowedUnsafeSysctls, "experimental-allowed-unsafe-sysctls", f.AllowedUnsafeSysctls, "Comma-separated whitelist of unsafe sysctls or unsafe sysctl patterns (ending in *). Use these at your own risk.")
	fs.BoolVar(&f.Containerized, "containerized", f.Containerized, "Experimental support for running kubelet in a container.  Intended for testing.")
	fs.BoolVar(&f.ExperimentalKernelMemcgNotification, "experimental-kernel-memcg-notification", f.ExperimentalKernelMemcgNotification, "If enabled, the kubelet will integrate with the kernel memcg notification to determine if memory eviction thresholds are crossed rather than polling.")
	fs.StringVar(&f.RemoteRuntimeEndpoint, "container-runtime-endpoint", f.RemoteRuntimeEndpoint, "[Experimental] The endpoint of remote runtime service. Currently unix socket is supported on Linux, and tcp is supported on windows.  Examples:'unix:///var/run/dockershim.sock', 'tcp://localhost:3735'")
	fs.StringVar(&f.RemoteImageEndpoint, "image-service-endpoint", f.RemoteImageEndpoint, "[Experimental] The endpoint of remote image service. If not specified, it will be the same with container-runtime-endpoint by default. Currently unix socket is supported on Linux, and tcp is supported on windows.  Examples:'unix:///var/run/dockershim.sock', 'tcp://localhost:3735'")
	fs.BoolVar(&f.ExperimentalCheckNodeCapabilitiesBeforeMount, "experimental-check-node-capabilities-before-mount", f.ExperimentalCheckNodeCapabilitiesBeforeMount, "[Experimental] if set true, the kubelet will check the underlying node for required componenets (binaries, etc.) before performing the mount")
	fs.BoolVar(&f.ExperimentalNodeAllocatableIgnoreEvictionThreshold, "experimental-allocatable-ignore-eviction", f.ExperimentalNodeAllocatableIgnoreEvictionThreshold, "When set to 'true', Hard Eviction Thresholds will be ignored while calculating Node Allocatable. See https://kubernetes.io/docs/tasks/administer-cluster/reserve-compute-resources/ for more details. [default=false]")
	fs.Var(&f.ExperimentalQOSReserved, "experimental-qos-reserved", "A set of ResourceName=Percentage (e.g. memory=50%) pairs that describe how pod resource requests are reserved at the QoS level. Currently only memory is supported. [default=none]")

	// DEPRECATED FLAGS
	fs.DurationVar(&f.MinimumGCAge.Duration, "minimum-container-ttl-duration", f.MinimumGCAge.Duration, "Minimum age for a finished container before it is garbage collected.  Examples: '300ms', '10s' or '2h45m'")
	fs.MarkDeprecated("minimum-container-ttl-duration", "Use --eviction-hard or --eviction-soft instead. Will be removed in a future version.")
	fs.Int32Var(&f.MaxPerPodContainerCount, "maximum-dead-containers-per-container", f.MaxPerPodContainerCount, "Maximum number of old instances to retain per container.  Each container takes up some disk space.")
	fs.MarkDeprecated("maximum-dead-containers-per-container", "Use --eviction-hard or --eviction-soft instead. Will be removed in a future version.")
	fs.Int32Var(&f.MaxContainerCount, "maximum-dead-containers", f.MaxContainerCount, "Maximum number of old instances of containers to retain globally.  Each container takes up some disk space. To disable, set to a negative number.")
	fs.MarkDeprecated("maximum-dead-containers", "Use --eviction-hard or --eviction-soft instead. Will be removed in a future version.")
	fs.StringVar(&f.MasterServiceNamespace, "master-service-namespace", f.MasterServiceNamespace, "The namespace from which the kubernetes master services should be injected into pods")
	fs.MarkDeprecated("master-service-namespace", "This flag will be removed in a future version.")
	fs.BoolVar(&f.RegisterSchedulable, "register-schedulable", f.RegisterSchedulable, "Register the node as schedulable. Won't have any effect if register-node is false.")
	fs.MarkDeprecated("register-schedulable", "will be removed in a future version")
	fs.StringVar(&f.NonMasqueradeCIDR, "non-masquerade-cidr", f.NonMasqueradeCIDR, "Traffic to IPs outside this range will use IP masquerade. Set to '0.0.0.0/0' to never masquerade.")
	fs.MarkDeprecated("non-masquerade-cidr", "will be removed in a future version")
	fs.BoolVar(&f.KeepTerminatedPodVolumes, "keep-terminated-pod-volumes", f.KeepTerminatedPodVolumes, "Keep terminated pod volumes mounted to the node after the pod terminates.  Can be useful for debugging volume related issues.")
	fs.MarkDeprecated("keep-terminated-pod-volumes", "will be removed in a future version")
	// TODO(#54161:v1.11.0): Remove --enable-custom-metrics flag, it is deprecated.
	fs.BoolVar(&f.EnableCustomMetrics, "enable-custom-metrics", f.EnableCustomMetrics, "Support for gathering custom metrics.")
	fs.MarkDeprecated("enable-custom-metrics", "will be removed in a future version")

}

// AddKubeletConfigFlags adds flags for a specific kubeletconfig.KubeletConfiguration to the specified FlagSet
func AddKubeletConfigFlags(fs *pflag.FlagSet, c *kubeletconfig.KubeletConfiguration) {
	fs.BoolVar(&c.FailSwapOn, "fail-swap-on", true, "Makes the Kubelet fail to start if swap is enabled on the node. ")
	fs.BoolVar(&c.FailSwapOn, "experimental-fail-swap-on", true, "DEPRECATED: please use --fail-swap-on instead.")
	fs.MarkDeprecated("experimental-fail-swap-on", "This flag is deprecated and will be removed in future releases. please use --fail-swap-on instead.")

	fs.StringVar(&c.PodManifestPath, "pod-manifest-path", c.PodManifestPath, "Path to the directory containing pod manifest files to run, or the path to a single pod manifest file. Files starting with dots will be ignored.")
	fs.DurationVar(&c.SyncFrequency.Duration, "sync-frequency", c.SyncFrequency.Duration, "Max period between synchronizing running containers and config")
	fs.DurationVar(&c.FileCheckFrequency.Duration, "file-check-frequency", c.FileCheckFrequency.Duration, "Duration between checking config files for new data")
	fs.DurationVar(&c.HTTPCheckFrequency.Duration, "http-check-frequency", c.HTTPCheckFrequency.Duration, "Duration between checking http for new data")
	fs.StringVar(&c.ManifestURL, "manifest-url", c.ManifestURL, "URL for accessing the container manifest")
	fs.StringVar(&c.ManifestURLHeader, "manifest-url-header", c.ManifestURLHeader, "HTTP header to use when accessing the manifest URL, with the key separated from the value with a ':', as in 'key:value'")
	fs.BoolVar(&c.EnableServer, "enable-server", c.EnableServer, "Enable the Kubelet's server")
	fs.Var(componentconfig.IPVar{Val: &c.Address}, "address", "The IP address for the Kubelet to serve on (set to 0.0.0.0 for all interfaces)")
	fs.Int32Var(&c.Port, "port", c.Port, "The port for the Kubelet to serve on.")
	fs.Int32Var(&c.ReadOnlyPort, "read-only-port", c.ReadOnlyPort, "The read-only port for the Kubelet to serve on with no authentication/authorization (set to 0 to disable)")

	// Authentication
	fs.BoolVar(&c.Authentication.Anonymous.Enabled, "anonymous-auth", c.Authentication.Anonymous.Enabled, ""+
		"Enables anonymous requests to the Kubelet server. Requests that are not rejected by another "+
		"authentication method are treated as anonymous requests. Anonymous requests have a username "+
		"of system:anonymous, and a group name of system:unauthenticated.")
	fs.BoolVar(&c.Authentication.Webhook.Enabled, "authentication-token-webhook", c.Authentication.Webhook.Enabled, ""+
		"Use the TokenReview API to determine authentication for bearer tokens.")
	fs.DurationVar(&c.Authentication.Webhook.CacheTTL.Duration, "authentication-token-webhook-cache-ttl", c.Authentication.Webhook.CacheTTL.Duration, ""+
		"The duration to cache responses from the webhook token authenticator.")
	fs.StringVar(&c.Authentication.X509.ClientCAFile, "client-ca-file", c.Authentication.X509.ClientCAFile, ""+
		"If set, any request presenting a client certificate signed by one of the authorities in the client-ca-file "+
		"is authenticated with an identity corresponding to the CommonName of the client certificate.")

	// Authorization
	fs.StringVar((*string)(&c.Authorization.Mode), "authorization-mode", string(c.Authorization.Mode), ""+
		"Authorization mode for Kubelet server. Valid options are AlwaysAllow or Webhook. "+
		"Webhook mode uses the SubjectAccessReview API to determine authorization.")
	fs.DurationVar(&c.Authorization.Webhook.CacheAuthorizedTTL.Duration, "authorization-webhook-cache-authorized-ttl", c.Authorization.Webhook.CacheAuthorizedTTL.Duration, ""+
		"The duration to cache 'authorized' responses from the webhook authorizer.")
	fs.DurationVar(&c.Authorization.Webhook.CacheUnauthorizedTTL.Duration, "authorization-webhook-cache-unauthorized-ttl", c.Authorization.Webhook.CacheUnauthorizedTTL.Duration, ""+
		"The duration to cache 'unauthorized' responses from the webhook authorizer.")

	fs.StringVar(&c.TLSCertFile, "tls-cert-file", c.TLSCertFile, ""+
		"File containing x509 Certificate used for serving HTTPS (with intermediate certs, if any, concatenated after server cert). "+
		"If --tls-cert-file and --tls-private-key-file are not provided, a self-signed certificate and key "+
		"are generated for the public address and saved to the directory passed to --cert-dir.")
	fs.StringVar(&c.TLSPrivateKeyFile, "tls-private-key-file", c.TLSPrivateKeyFile, "File containing x509 private key matching --tls-cert-file.")

	fs.StringVar(&c.SeccompProfileRoot, "seccomp-profile-root", c.SeccompProfileRoot, "Directory path for seccomp profiles.")
	fs.BoolVar(&c.AllowPrivileged, "allow-privileged", c.AllowPrivileged, "If true, allow containers to request privileged mode.")
	fs.StringSliceVar(&c.HostNetworkSources, "host-network-sources", c.HostNetworkSources, "Comma-separated list of sources from which the Kubelet allows pods to use of host network.")
	fs.StringSliceVar(&c.HostPIDSources, "host-pid-sources", c.HostPIDSources, "Comma-separated list of sources from which the Kubelet allows pods to use the host pid namespace.")
	fs.StringSliceVar(&c.HostIPCSources, "host-ipc-sources", c.HostIPCSources, "Comma-separated list of sources from which the Kubelet allows pods to use the host ipc namespace.")
	fs.Int32Var(&c.RegistryPullQPS, "registry-qps", c.RegistryPullQPS, "If > 0, limit registry pull QPS to this value.  If 0, unlimited.")
	fs.Int32Var(&c.RegistryBurst, "registry-burst", c.RegistryBurst, "Maximum size of a bursty pulls, temporarily allows pulls to burst to this number, while still not exceeding registry-qps. Only used if --registry-qps > 0")
	fs.Int32Var(&c.EventRecordQPS, "event-qps", c.EventRecordQPS, "If > 0, limit event creations per second to this value. If 0, unlimited.")
	fs.Int32Var(&c.EventBurst, "event-burst", c.EventBurst, "Maximum size of a bursty event records, temporarily allows event records to burst to this number, while still not exceeding event-qps. Only used if --event-qps > 0")

	fs.BoolVar(&c.EnableDebuggingHandlers, "enable-debugging-handlers", c.EnableDebuggingHandlers, "Enables server endpoints for log collection and local running of containers and commands")
	fs.BoolVar(&c.EnableContentionProfiling, "contention-profiling", false, "Enable lock contention profiling, if profiling is enabled")
	fs.Int32Var(&c.CAdvisorPort, "cadvisor-port", c.CAdvisorPort, "The port of the localhost cAdvisor endpoint (set to 0 to disable)")
	fs.Int32Var(&c.HealthzPort, "healthz-port", c.HealthzPort, "The port of the localhost healthz endpoint (set to 0 to disable)")
	fs.Var(componentconfig.IPVar{Val: &c.HealthzBindAddress}, "healthz-bind-address", "The IP address for the healthz server to serve on. (set to 0.0.0.0 for all interfaces)")
	fs.Int32Var(&c.OOMScoreAdj, "oom-score-adj", c.OOMScoreAdj, "The oom-score-adj value for kubelet process. Values must be within the range [-1000, 1000]")
	fs.BoolVar(&c.RegisterNode, "register-node", c.RegisterNode, "Register the node with the apiserver. If --kubeconfig is not provided, this flag is irrelevant, as the Kubelet won't have an apiserver to register with. Default=true.")
	fs.StringVar(&c.ClusterDomain, "cluster-domain", c.ClusterDomain, "Domain for this cluster.  If set, kubelet will configure all containers to search this domain in addition to the host's search domains")

	fs.StringSliceVar(&c.ClusterDNS, "cluster-dns", c.ClusterDNS, "Comma-separated list of DNS server IP address.  This value is used for containers DNS server in case of Pods with \"dnsPolicy=ClusterFirst\". Note: all DNS servers appearing in the list MUST serve the same set of records otherwise name resolution within the cluster may not work correctly. There is no guarantee as to which DNS server may be contacted for name resolution.")
	fs.DurationVar(&c.StreamingConnectionIdleTimeout.Duration, "streaming-connection-idle-timeout", c.StreamingConnectionIdleTimeout.Duration, "Maximum time a streaming connection can be idle before the connection is automatically closed. 0 indicates no timeout. Example: '5m'")
	fs.DurationVar(&c.NodeStatusUpdateFrequency.Duration, "node-status-update-frequency", c.NodeStatusUpdateFrequency.Duration, "Specifies how often kubelet posts node status to master. Note: be cautious when changing the constant, it must work with nodeMonitorGracePeriod in nodecontroller.")
	c.NodeLabels = make(map[string]string)
	bindableNodeLabels := flag.ConfigurationMap(c.NodeLabels)
	fs.Var(&bindableNodeLabels, "node-labels", "<Warning: Alpha feature> Labels to add when registering the node in the cluster.  Labels must be key=value pairs separated by ','.")
	fs.DurationVar(&c.ImageMinimumGCAge.Duration, "minimum-image-ttl-duration", c.ImageMinimumGCAge.Duration, "Minimum age for an unused image before it is garbage collected.  Examples: '300ms', '10s' or '2h45m'.")
	fs.Int32Var(&c.ImageGCHighThresholdPercent, "image-gc-high-threshold", c.ImageGCHighThresholdPercent, "The percent of disk usage after which image garbage collection is always run.")
	fs.Int32Var(&c.ImageGCLowThresholdPercent, "image-gc-low-threshold", c.ImageGCLowThresholdPercent, "The percent of disk usage before which image garbage collection is never run. Lowest disk usage to garbage collect to.")
	fs.DurationVar(&c.VolumeStatsAggPeriod.Duration, "volume-stats-agg-period", c.VolumeStatsAggPeriod.Duration, "Specifies interval for kubelet to calculate and cache the volume disk usage for all pods and volumes.  To disable volume calculations, set to 0.")
	fs.StringVar(&c.VolumePluginDir, "volume-plugin-dir", c.VolumePluginDir, "<Warning: Alpha feature> The full path of the directory in which to search for additional third party volume plugins")
	fs.Var(flag.MapStringBool(c.FeatureGates), "feature-gates", "A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(utilfeature.DefaultFeatureGate.KnownFeatures(), "\n"))
	fs.StringVar(&c.KubeletCgroups, "kubelet-cgroups", c.KubeletCgroups, "Optional absolute name of cgroups to create and run the Kubelet in.")
	fs.StringVar(&c.SystemCgroups, "system-cgroups", c.SystemCgroups, "Optional absolute name of cgroups in which to place all non-kernel processes that are not already inside a cgroup under `/`. Empty for no container. Rolling back the flag requires a reboot.")

	fs.BoolVar(&c.CgroupsPerQOS, "cgroups-per-qos", c.CgroupsPerQOS, "Enable creation of QoS cgroup hierarchy, if true top level QoS and pod cgroups are created.")
	fs.StringVar(&c.CgroupDriver, "cgroup-driver", c.CgroupDriver, "Driver that the kubelet uses to manipulate cgroups on the host.  Possible values: 'cgroupfs', 'systemd'")
	fs.StringVar(&c.CgroupRoot, "cgroup-root", c.CgroupRoot, "Optional root cgroup to use for pods. This is handled by the container runtime on a best effort basis. Default: '', which means use the container runtime default.")
	fs.StringVar(&c.CPUManagerPolicy, "cpu-manager-policy", c.CPUManagerPolicy, "<Warning: Alpha feature> CPU Manager policy to use. Possible values: 'none', 'static'. Default: 'none'")
	fs.DurationVar(&c.CPUManagerReconcilePeriod.Duration, "cpu-manager-reconcile-period", c.CPUManagerReconcilePeriod.Duration, "<Warning: Alpha feature> CPU Manager reconciliation period. Examples: '10s', or '1m'. If not supplied, defaults to `NodeStatusUpdateFrequency`")
	fs.StringVar(&c.ContainerRuntime, "container-runtime", c.ContainerRuntime, "The container runtime to use. Possible values: 'docker', 'rkt'.")
	fs.DurationVar(&c.RuntimeRequestTimeout.Duration, "runtime-request-timeout", c.RuntimeRequestTimeout.Duration, "Timeout of all runtime requests except long running request - pull, logs, exec and attach. When timeout exceeded, kubelet will cancel the request, throw out an error and retry later.")
	fs.StringVar(&c.LockFilePath, "lock-file", c.LockFilePath, "<Warning: Alpha feature> The path to file for kubelet to use as a lock file.")
	fs.BoolVar(&c.ExitOnLockContention, "exit-on-lock-contention", c.ExitOnLockContention, "Whether kubelet should exit upon lock-file contention.")
	fs.StringVar(&c.HairpinMode, "hairpin-mode", c.HairpinMode, "How should the kubelet setup hairpin NAT. This allows endpoints of a Service to loadbalance back to themselves if they should try to access their own Service. Valid values are \"promiscuous-bridge\", \"hairpin-veth\" and \"none\".")
	fs.Int32Var(&c.MaxPods, "max-pods", c.MaxPods, "Number of Pods that can run on this Kubelet.")

	fs.StringVar(&c.PodCIDR, "pod-cidr", "", "The CIDR to use for pod IP addresses, only used in standalone mode.  In cluster mode, this is obtained from the master.")
	fs.StringVar(&c.ResolverConfig, "resolv-conf", c.ResolverConfig, "Resolver configuration file used as the basis for the container DNS resolution configuration.")
	fs.BoolVar(&c.CPUCFSQuota, "cpu-cfs-quota", c.CPUCFSQuota, "Enable CPU CFS quota enforcement for containers that specify CPU limits")
	fs.BoolVar(&c.EnableControllerAttachDetach, "enable-controller-attach-detach", c.EnableControllerAttachDetach, "Enables the Attach/Detach controller to manage attachment/detachment of volumes scheduled to this node, and disables kubelet from executing any attach/detach operations")
	fs.BoolVar(&c.MakeIPTablesUtilChains, "make-iptables-util-chains", c.MakeIPTablesUtilChains, "If true, kubelet will ensure iptables utility rules are present on host.")
	fs.Int32Var(&c.IPTablesMasqueradeBit, "iptables-masquerade-bit", c.IPTablesMasqueradeBit, "The bit of the fwmark space to mark packets for SNAT. Must be within the range [0, 31]. Please match this parameter with corresponding parameter in kube-proxy.")
	fs.Int32Var(&c.IPTablesDropBit, "iptables-drop-bit", c.IPTablesDropBit, "The bit of the fwmark space to mark packets for dropping. Must be within the range [0, 31].")

	// Flags intended for testing, not recommended used in production environments.
	fs.Int64Var(&c.MaxOpenFiles, "max-open-files", c.MaxOpenFiles, "Number of files that can be opened by Kubelet process.")

	fs.Var(utiltaints.NewTaintsVar(&c.RegisterWithTaints), "register-with-taints", "Register the node with the given list of taints (comma separated \"<key>=<value>:<effect>\"). No-op if register-node is false.")
	fs.StringVar(&c.ContentType, "kube-api-content-type", c.ContentType, "Content type of requests sent to apiserver.")
	fs.Int32Var(&c.KubeAPIQPS, "kube-api-qps", c.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&c.KubeAPIBurst, "kube-api-burst", c.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver")
	fs.BoolVar(&c.SerializeImagePulls, "serialize-image-pulls", c.SerializeImagePulls, "Pull images one at a time. We recommend *not* changing the default value on nodes that run docker daemon with version < 1.9 or an Aufs storage backend. Issue #10959 has more details.")

	fs.StringVar(&c.RuntimeCgroups, "runtime-cgroups", c.RuntimeCgroups, "Optional absolute name of cgroups to create and run the runtime in.")
	fs.StringVar(&c.EvictionHard, "eviction-hard", c.EvictionHard, "A set of eviction thresholds (e.g. memory.available<1Gi) that if met would trigger a pod eviction.")
	fs.StringVar(&c.EvictionSoft, "eviction-soft", c.EvictionSoft, "A set of eviction thresholds (e.g. memory.available<1.5Gi) that if met over a corresponding grace period would trigger a pod eviction.")
	fs.StringVar(&c.EvictionSoftGracePeriod, "eviction-soft-grace-period", c.EvictionSoftGracePeriod, "A set of eviction grace periods (e.g. memory.available=1m30s) that correspond to how long a soft eviction threshold must hold before triggering a pod eviction.")
	fs.DurationVar(&c.EvictionPressureTransitionPeriod.Duration, "eviction-pressure-transition-period", c.EvictionPressureTransitionPeriod.Duration, "Duration for which the kubelet has to wait before transitioning out of an eviction pressure condition.")
	fs.Int32Var(&c.EvictionMaxPodGracePeriod, "eviction-max-pod-grace-period", c.EvictionMaxPodGracePeriod, "Maximum allowed grace period (in seconds) to use when terminating pods in response to a soft eviction threshold being met.  If negative, defer to pod specified value.")
	fs.StringVar(&c.EvictionMinimumReclaim, "eviction-minimum-reclaim", c.EvictionMinimumReclaim, "A set of minimum reclaims (e.g. imagefs.available=2Gi) that describes the minimum amount of resource the kubelet will reclaim when performing a pod eviction if that resource is under pressure.")
	fs.Int32Var(&c.PodsPerCore, "pods-per-core", c.PodsPerCore, "Number of Pods per core that can run on this Kubelet. The total number of Pods on this Kubelet cannot exceed max-pods, so max-pods will be used if this calculation results in a larger number of Pods allowed on the Kubelet. A value of 0 disables this limit.")
	fs.BoolVar(&c.ProtectKernelDefaults, "protect-kernel-defaults", c.ProtectKernelDefaults, "Default kubelet behaviour for kernel tuning. If set, kubelet errors if any of kernel tunables is different than kubelet defaults.")

	// Node Allocatable Flags
	fs.Var(&c.SystemReserved, "system-reserved", "A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=500Mi) pairs that describe resources reserved for non-kubernetes components. Currently only cpu and memory are supported. See http://kubernetes.io/docs/user-guide/compute-resources for more detail. [default=none]")
	fs.Var(&c.KubeReserved, "kube-reserved", "A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=500Mi, ephemeral-storage=1Gi) pairs that describe resources reserved for kubernetes system components. Currently cpu, memory and local ephemeral storage for root file system are supported. See http://kubernetes.io/docs/user-guide/compute-resources for more detail. [default=none]")
	fs.StringSliceVar(&c.EnforceNodeAllocatable, "enforce-node-allocatable", c.EnforceNodeAllocatable, "A comma separated list of levels of node allocatable enforcement to be enforced by kubelet. Acceptible options are 'pods', 'system-reserved' & 'kube-reserved'. If the latter two options are specified, '--system-reserved-cgroup' & '--kube-reserved-cgroup' must also be set respectively. See https://kubernetes.io/docs/tasks/administer-cluster/reserve-compute-resources/ for more details.")
	fs.StringVar(&c.SystemReservedCgroup, "system-reserved-cgroup", c.SystemReservedCgroup, "Absolute name of the top level cgroup that is used to manage non-kubernetes components for which compute resources were reserved via '--system-reserved' flag. Ex. '/system-reserved'. [default='']")
	fs.StringVar(&c.KubeReservedCgroup, "kube-reserved-cgroup", c.KubeReservedCgroup, "Absolute name of the top level cgroup that is used to manage kubernetes components for which compute resources were reserved via '--kube-reserved' flag. Ex. '/kube-reserved'. [default='']")
}
