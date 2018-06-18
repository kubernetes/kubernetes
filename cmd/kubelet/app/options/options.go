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
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/pflag"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1beta1"
	kubeletconfigvalidation "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/validation"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/master/ports"
	utiltaints "k8s.io/kubernetes/pkg/util/taints"
)

const defaultRootDir = "/var/lib/kubelet"

// A configuration field should go in KubeletFlags instead of KubeletConfiguration if any of these are true:
// - its value will never, or cannot safely be changed during the lifetime of a node
// - its value cannot be safely shared between nodes at the same time (e.g. a hostname)
//   KubeletConfiguration is intended to be shared between nodes
// In general, please try to avoid adding flags or configuration fields,
// we already have a confusingly large amount of them.
type KubeletFlags struct {
	KubeConfig          string
	BootstrapKubeconfig string

	// Insert a probability of random errors during calls to the master.
	ChaosChance float64
	// Crash immediately, rather than eating panics.
	ReallyCrashForTesting bool

	// TODO(mtaufen): It is increasingly looking like nobody actually uses the
	//                Kubelet's runonce mode anymore, so it may be a candidate
	//                for deprecation and removal.
	// If runOnce is true, the Kubelet will check the API server once for pods,
	// run those in addition to the pods specified by static pod files, and exit.
	RunOnce bool

	// enableServer enables the Kubelet's server
	EnableServer bool

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

	// The Kubelet will load its initial configuration from this file.
	// The path may be absolute or relative; relative paths are under the Kubelet's current working directory.
	// Omit this flag to use the combination of built-in default configuration values and flags.
	KubeletConfigFile string

	// registerNode enables automatic registration with the apiserver.
	RegisterNode bool

	// registerWithTaints are an array of taints to add to a node object when
	// the kubelet registers itself. This only takes effect when registerNode
	// is true and upon the initial registration of the node.
	RegisterWithTaints []core.Taint

	// cAdvisorPort is the port of the localhost cAdvisor endpoint (set to 0 to disable)
	CAdvisorPort int32

	// WindowsService should be set to true if kubelet is running as a service on Windows.
	// Its corresponding flag only gets registered in Windows builds.
	WindowsService bool

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
	// Node Labels are the node labels to add when registering the node in the cluster
	NodeLabels map[string]string
	// volumePluginDir is the full path of the directory in which to search
	// for additional third party volume plugins
	VolumePluginDir string
	// lockFilePath is the path that kubelet will use to as a lock file.
	// It uses this file as a lock to synchronize with other kubelet processes
	// that may be running.
	LockFilePath string
	// ExitOnLockContention is a flag that signifies to the kubelet that it is running
	// in "bootstrap" mode. This requires that 'LockFilePath' has been set.
	// This will cause the kubelet to listen to inotify events on the lock file,
	// releasing it and exiting when another process tries to open that file.
	ExitOnLockContention bool
	// seccompProfileRoot is the directory path for seccomp profiles.
	SeccompProfileRoot string
	// bootstrapCheckpointPath is the path to the directory containing pod checkpoints to
	// run on restore
	BootstrapCheckpointPath string
	// NodeStatusMaxImages caps the number of images reported in Node.Status.Images.
	// This is an experimental, short-term flag to help with node scalability.
	NodeStatusMaxImages int32

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
	// nonMasqueradeCIDR configures masquerading: traffic to IPs outside this range will use IP masquerade.
	NonMasqueradeCIDR string
	// This flag, if set, instructs the kubelet to keep volumes from terminated pods mounted to the node.
	// This can be useful for debugging volume related issues.
	KeepTerminatedPodVolumes bool
	// allowPrivileged enables containers to request privileged mode.
	// Defaults to true.
	AllowPrivileged bool
	// hostNetworkSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use of host network. Defaults to "*". Valid
	// options are "file", "http", "api", and "*" (all sources).
	HostNetworkSources []string
	// hostPIDSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use the host pid namespace. Defaults to "*".
	HostPIDSources []string
	// hostIPCSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use the host ipc namespace. Defaults to "*".
	HostIPCSources []string
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
		EnableServer:                        true,
		ContainerRuntimeOptions:             *NewContainerRuntimeOptions(),
		CertDirectory:                       "/var/lib/kubelet/pki",
		RootDirectory:                       defaultRootDir,
		MasterServiceNamespace:              metav1.NamespaceDefault,
		MaxContainerCount:                   -1,
		MaxPerPodContainerCount:             1,
		MinimumGCAge:                        metav1.Duration{Duration: 0},
		NonMasqueradeCIDR:                   "10.0.0.0/8",
		RegisterSchedulable:                 true,
		ExperimentalKernelMemcgNotification: false,
		RemoteRuntimeEndpoint:               remoteRuntimeEndpoint,
		NodeLabels:                          make(map[string]string),
		VolumePluginDir:                     "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/",
		RegisterNode:                        true,
		SeccompProfileRoot:                  filepath.Join(defaultRootDir, "seccomp"),
		HostNetworkSources:                  []string{kubetypes.AllSource},
		HostPIDSources:                      []string{kubetypes.AllSource},
		HostIPCSources:                      []string{kubetypes.AllSource},
		// TODO(#56523:v1.12.0): Remove --cadvisor-port, it has been deprecated since v1.10
		CAdvisorPort: 0,
		// TODO(#58010:v1.13.0): Remove --allow-privileged, it is deprecated
		AllowPrivileged: true,
		// prior to the introduction of this flag, there was a hardcoded cap of 50 images
		NodeStatusMaxImages: 50,
	}
}

func ValidateKubeletFlags(f *KubeletFlags) error {
	// ensure that nobody sets DynamicConfigDir if the dynamic config feature gate is turned off
	if f.DynamicConfigDir.Provided() && !utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
		return fmt.Errorf("the DynamicKubeletConfig feature gate must be enabled in order to use the --dynamic-config-dir flag")
	}
	if f.CAdvisorPort != 0 && utilvalidation.IsValidPortNum(int(f.CAdvisorPort)) != nil {
		return fmt.Errorf("invalid configuration: CAdvisorPort (--cadvisor-port) %v must be between 0 and 65535, inclusive", f.CAdvisorPort)
	}
	if f.NodeStatusMaxImages < -1 {
		return fmt.Errorf("invalid configuration: NodeStatusMaxImages (--node-status-max-images) must be -1 or greater")
	}
	return nil
}

// NewKubeletConfiguration will create a new KubeletConfiguration with default values
func NewKubeletConfiguration() (*kubeletconfig.KubeletConfiguration, error) {
	scheme, _, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}
	versioned := &v1beta1.KubeletConfiguration{}
	scheme.Default(versioned)
	config := &kubeletconfig.KubeletConfiguration{}
	if err := scheme.Convert(versioned, config, nil); err != nil {
		return nil, err
	}
	applyLegacyDefaults(config)
	return config, nil
}

// applyLegacyDefaults applies legacy default values to the KubeletConfiguration in order to
// preserve the command line API. This is used to construct the baseline default KubeletConfiguration
// before the first round of flag parsing.
func applyLegacyDefaults(kc *kubeletconfig.KubeletConfiguration) {
	// --anonymous-auth
	kc.Authentication.Anonymous.Enabled = true
	// --authentication-token-webhook
	kc.Authentication.Webhook.Enabled = false
	// --authorization-mode
	kc.Authorization.Mode = kubeletconfig.KubeletAuthorizationModeAlwaysAllow
	// --read-only-port
	kc.ReadOnlyPort = ports.KubeletReadOnlyPort
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
func (f *KubeletFlags) AddFlags(mainfs *pflag.FlagSet) {
	fs := pflag.NewFlagSet("", pflag.ExitOnError)
	defer func() {
		// Unhide deprecated flags. We want deprecated flags to show in Kubelet help.
		// We have some hidden flags, but we might as well unhide these when they are deprecated,
		// as silently deprecating and removing (even hidden) things is unkind to people who use them.
		fs.VisitAll(func(f *pflag.Flag) {
			if len(f.Deprecated) > 0 {
				f.Hidden = false
			}
		})
		mainfs.AddFlagSet(fs)
	}()

	f.ContainerRuntimeOptions.AddFlags(fs)
	f.addOSFlags(fs)

	fs.StringVar(&f.KubeletConfigFile, "config", f.KubeletConfigFile, "The Kubelet will load its initial configuration from this file. The path may be absolute or relative; relative paths start at the Kubelet's current working directory. Omit this flag to use the built-in default configuration values. Command-line flags override configuration from this file.")
	fs.StringVar(&f.KubeConfig, "kubeconfig", f.KubeConfig, "Path to a kubeconfig file, specifying how to connect to the API server. Providing --kubeconfig enables API server mode, omitting --kubeconfig enables standalone mode.")

	fs.StringVar(&f.BootstrapKubeconfig, "bootstrap-kubeconfig", f.BootstrapKubeconfig, "Path to a kubeconfig file that will be used to get client certificate for kubelet. "+
		"If the file specified by --kubeconfig does not exist, the bootstrap kubeconfig is used to request a client certificate from the API server. "+
		"On success, a kubeconfig file referencing the generated client certificate and key is written to the path specified by --kubeconfig. "+
		"The client certificate and key file will be stored in the directory pointed by --cert-dir.")

	fs.BoolVar(&f.ReallyCrashForTesting, "really-crash-for-testing", f.ReallyCrashForTesting, "If true, when panics occur crash. Intended for testing.")
	fs.Float64Var(&f.ChaosChance, "chaos-chance", f.ChaosChance, "If > 0.0, introduce random client errors and latency. Intended for testing.")

	fs.BoolVar(&f.RunOnce, "runonce", f.RunOnce, "If true, exit after spawning pods from static pod files or remote urls. Exclusive with --enable-server")
	fs.BoolVar(&f.EnableServer, "enable-server", f.EnableServer, "Enable the Kubelet's server")

	fs.StringVar(&f.HostnameOverride, "hostname-override", f.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname. If --cloud-provider is set, the cloud provider determines the name of the node (consult cloud provider documentation to determine if and how the hostname is used).")

	fs.StringVar(&f.NodeIP, "node-ip", f.NodeIP, "IP address of the node. If set, kubelet will use this IP address for the node")

	fs.StringVar(&f.ProviderID, "provider-id", f.ProviderID, "Unique identifier for identifying the node in a machine database, i.e cloudprovider")

	fs.StringVar(&f.CertDirectory, "cert-dir", f.CertDirectory, "The directory where the TLS certs are located. "+
		"If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored.")

	fs.StringVar(&f.CloudProvider, "cloud-provider", f.CloudProvider, "The provider for cloud services. Specify empty string for running with no cloud provider. If set, the cloud provider determines the name of the node (consult cloud provider documentation to determine if and how the hostname is used).")
	fs.StringVar(&f.CloudConfigFile, "cloud-config", f.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")

	fs.StringVar(&f.RootDirectory, "root-dir", f.RootDirectory, "Directory path for managing kubelet files (volume mounts,etc).")

	fs.Var(&f.DynamicConfigDir, "dynamic-config-dir", "The Kubelet will use this directory for checkpointing downloaded configurations and tracking configuration health. The Kubelet will create this directory if it does not already exist. The path may be absolute or relative; relative paths start at the Kubelet's current working directory. Providing this flag enables dynamic Kubelet configuration. The DynamicKubeletConfig feature gate must be enabled to pass this flag; this gate currently defaults to true because the feature is beta.")

	fs.BoolVar(&f.RegisterNode, "register-node", f.RegisterNode, "Register the node with the apiserver. If --kubeconfig is not provided, this flag is irrelevant, as the Kubelet won't have an apiserver to register with. Default=true.")
	fs.Var(utiltaints.NewTaintsVar(&f.RegisterWithTaints), "register-with-taints", "Register the node with the given list of taints (comma separated \"<key>=<value>:<effect>\"). No-op if register-node is false.")
	fs.BoolVar(&f.Containerized, "containerized", f.Containerized, "Running kubelet in a container.")

	// EXPERIMENTAL FLAGS
	fs.StringVar(&f.ExperimentalMounterPath, "experimental-mounter-path", f.ExperimentalMounterPath, "[Experimental] Path of mounter binary. Leave empty to use the default mount.")
	fs.StringSliceVar(&f.AllowedUnsafeSysctls, "allowed-unsafe-sysctls", f.AllowedUnsafeSysctls, "Comma-separated whitelist of unsafe sysctls or unsafe sysctl patterns (ending in *). Use these at your own risk. Sysctls feature gate is enabled by default.")
	fs.BoolVar(&f.ExperimentalKernelMemcgNotification, "experimental-kernel-memcg-notification", f.ExperimentalKernelMemcgNotification, "If enabled, the kubelet will integrate with the kernel memcg notification to determine if memory eviction thresholds are crossed rather than polling.")
	fs.StringVar(&f.RemoteRuntimeEndpoint, "container-runtime-endpoint", f.RemoteRuntimeEndpoint, "[Experimental] The endpoint of remote runtime service. Currently unix socket is supported on Linux, and tcp is supported on windows.  Examples:'unix:///var/run/dockershim.sock', 'tcp://localhost:3735'")
	fs.StringVar(&f.RemoteImageEndpoint, "image-service-endpoint", f.RemoteImageEndpoint, "[Experimental] The endpoint of remote image service. If not specified, it will be the same with container-runtime-endpoint by default. Currently unix socket is supported on Linux, and tcp is supported on windows.  Examples:'unix:///var/run/dockershim.sock', 'tcp://localhost:3735'")
	fs.BoolVar(&f.ExperimentalCheckNodeCapabilitiesBeforeMount, "experimental-check-node-capabilities-before-mount", f.ExperimentalCheckNodeCapabilitiesBeforeMount, "[Experimental] if set true, the kubelet will check the underlying node for required components (binaries, etc.) before performing the mount")
	fs.BoolVar(&f.ExperimentalNodeAllocatableIgnoreEvictionThreshold, "experimental-allocatable-ignore-eviction", f.ExperimentalNodeAllocatableIgnoreEvictionThreshold, "When set to 'true', Hard Eviction Thresholds will be ignored while calculating Node Allocatable. See https://kubernetes.io/docs/tasks/administer-cluster/reserve-compute-resources/ for more details. [default=false]")
	bindableNodeLabels := flag.ConfigurationMap(f.NodeLabels)
	fs.Var(&bindableNodeLabels, "node-labels", "<Warning: Alpha feature> Labels to add when registering the node in the cluster.  Labels must be key=value pairs separated by ','.")
	fs.StringVar(&f.VolumePluginDir, "volume-plugin-dir", f.VolumePluginDir, "The full path of the directory in which to search for additional third party volume plugins")
	fs.StringVar(&f.LockFilePath, "lock-file", f.LockFilePath, "<Warning: Alpha feature> The path to file for kubelet to use as a lock file.")
	fs.BoolVar(&f.ExitOnLockContention, "exit-on-lock-contention", f.ExitOnLockContention, "Whether kubelet should exit upon lock-file contention.")
	fs.StringVar(&f.SeccompProfileRoot, "seccomp-profile-root", f.SeccompProfileRoot, "<Warning: Alpha feature> Directory path for seccomp profiles.")
	fs.StringVar(&f.BootstrapCheckpointPath, "bootstrap-checkpoint-path", f.BootstrapCheckpointPath, "<Warning: Alpha feature> Path to to the directory where the checkpoints are stored")
	fs.Int32Var(&f.NodeStatusMaxImages, "node-status-max-images", f.NodeStatusMaxImages, "<Warning: Alpha feature> The maximum number of images to report in Node.Status.Images. If -1 is specified, no cap will be applied. Default: 50")

	// DEPRECATED FLAGS
	fs.StringVar(&f.BootstrapKubeconfig, "experimental-bootstrap-kubeconfig", f.BootstrapKubeconfig, "")
	fs.MarkDeprecated("experimental-bootstrap-kubeconfig", "Use --bootstrap-kubeconfig")
	fs.Int32Var(&f.CAdvisorPort, "cadvisor-port", f.CAdvisorPort, "The port of the localhost cAdvisor endpoint (set to 0 to disable)")
	fs.MarkDeprecated("cadvisor-port", "The default will change to 0 (disabled) in 1.11, and the cadvisor port will be removed entirely in 1.12")
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
	// TODO(#58010:v1.13.0): Remove --allow-privileged, it is deprecated
	fs.BoolVar(&f.AllowPrivileged, "allow-privileged", f.AllowPrivileged, "If true, allow containers to request privileged mode. Default: true")
	fs.MarkDeprecated("allow-privileged", "will be removed in a future version")
	// TODO(#58010:v1.12.0): Remove --host-network-sources, it is deprecated
	fs.StringSliceVar(&f.HostNetworkSources, "host-network-sources", f.HostNetworkSources, "Comma-separated list of sources from which the Kubelet allows pods to use of host network.")
	fs.MarkDeprecated("host-network-sources", "will be removed in a future version")
	// TODO(#58010:v1.12.0): Remove --host-pid-sources, it is deprecated
	fs.StringSliceVar(&f.HostPIDSources, "host-pid-sources", f.HostPIDSources, "Comma-separated list of sources from which the Kubelet allows pods to use the host pid namespace.")
	fs.MarkDeprecated("host-pid-sources", "will be removed in a future version")
	// TODO(#58010:v1.12.0): Remove --host-ipc-sources, it is deprecated
	fs.StringSliceVar(&f.HostIPCSources, "host-ipc-sources", f.HostIPCSources, "Comma-separated list of sources from which the Kubelet allows pods to use the host ipc namespace.")
	fs.MarkDeprecated("host-ipc-sources", "will be removed in a future version")

}

// AddKubeletConfigFlags adds flags for a specific kubeletconfig.KubeletConfiguration to the specified FlagSet
func AddKubeletConfigFlags(mainfs *pflag.FlagSet, c *kubeletconfig.KubeletConfiguration) {
	fs := pflag.NewFlagSet("", pflag.ExitOnError)
	defer func() {
		// All KubeletConfiguration flags are now deprecated, and any new flags that point to
		// KubeletConfiguration fields are deprecated-on-creation. When removing flags at the end
		// of their deprecation period, be careful to check that they have *actually* been deprecated
		// members of the KubeletConfiguration for the entire deprecation period:
		// e.g. if a flag was added after this deprecation function, it may not be at the end
		// of its lifetime yet, even if the rest are.
		deprecated := "This parameter should be set via the config file specified by the Kubelet's --config flag. See https://kubernetes.io/docs/tasks/administer-cluster/kubelet-config-file/ for more information."
		fs.VisitAll(func(f *pflag.Flag) {
			f.Deprecated = deprecated
		})
		mainfs.AddFlagSet(fs)
	}()

	// Flags marked deprecated in v1.11, remove in v1.13:
	// Flag marked deprecated in: #63912
	fs.BoolVar(&c.RotateCertificates, "rotate-certificates", c.RotateCertificates, "<Warning: Beta feature> Auto rotate the kubelet client certificates by requesting new certificates from the kube-apiserver when the certificate expiration approaches.")
	// Flag marked deprecated in: #62509
	fs.Var(flag.NewMapStringString(&c.QOSReserved), "qos-reserved", "<Warning: Alpha feature> A set of ResourceName=Percentage (e.g. memory=50%) pairs that describe how pod resource requests are reserved at the QoS level. Currently only memory is supported. Requires the QOSReserved feature gate to be enabled.")
}
