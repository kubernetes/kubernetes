/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Package app makes it easy to create a kubelet server for various contexts.
package app

// Note: if you change code in this file, you might need to change code in
// contrib/mesos/pkg/executor/service/.

import (
	"crypto/tls"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	_ "net/http/pprof"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/chaosclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/healthz"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/config"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/network"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	nodeutil "github.com/GoogleCloudPlatform/kubernetes/pkg/util/node"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

const defaultRootDir = "/var/lib/kubelet"

// KubeletServer encapsulates all of the parameters necessary for starting up
// a kubelet. These can either be set via command line or directly.
type KubeletServer struct {
	Config                         string
	SyncFrequency                  time.Duration
	FileCheckFrequency             time.Duration
	HTTPCheckFrequency             time.Duration
	ManifestURL                    string
	EnableServer                   bool
	Address                        util.IP
	Port                           uint
	ReadOnlyPort                   uint
	HostnameOverride               string
	PodInfraContainerImage         string
	DockerEndpoint                 string
	RootDirectory                  string
	AllowPrivileged                bool
	HostNetworkSources             string
	RegistryPullQPS                float64
	RegistryBurst                  int
	RunOnce                        bool
	EnableDebuggingHandlers        bool
	MinimumGCAge                   time.Duration
	MaxPerPodContainerCount        int
	MaxContainerCount              int
	AuthPath                       util.StringFlag // Deprecated -- use KubeConfig instead
	KubeConfig                     util.StringFlag
	CadvisorPort                   uint
	HealthzPort                    int
	HealthzBindAddress             util.IP
	OOMScoreAdj                    int
	APIServerList                  util.StringList
	RegisterNode                   bool
	StandaloneMode                 bool
	ClusterDomain                  string
	MasterServiceNamespace         string
	ClusterDNS                     util.IP
	StreamingConnectionIdleTimeout time.Duration
	ImageGCHighThresholdPercent    int
	ImageGCLowThresholdPercent     int
	LowDiskSpaceThresholdMB        int
	NetworkPluginName              string
	CloudProvider                  string
	CloudConfigFile                string
	TLSCertFile                    string
	TLSPrivateKeyFile              string
	CertDirectory                  string
	NodeStatusUpdateFrequency      time.Duration
	ResourceContainer              string
	CgroupRoot                     string
	ContainerRuntime               string
	DockerDaemonContainer          string
	SystemContainer                string
	ConfigureCBR0                  bool
	PodCIDR                        string
	MaxPods                        int
	DockerExecHandlerName          string

	// Flags intended for testing

	// Crash immediately, rather than eating panics.
	ReallyCrashForTesting bool
	// Insert a probability of random errors during calls to the master.
	ChaosChance float64
	// Is the kubelet containerized?
	Containerized bool
}

// bootstrapping interface for kubelet, targets the initialization protocol
type KubeletBootstrap interface {
	BirthCry()
	StartGarbageCollection()
	ListenAndServe(net.IP, uint, *kubelet.TLSOptions, bool)
	ListenAndServeReadOnly(net.IP, uint)
	Run(<-chan kubelet.PodUpdate)
	RunOnce(<-chan kubelet.PodUpdate) ([]kubelet.RunPodResult, error)
}

// create and initialize a Kubelet instance
type KubeletBuilder func(kc *KubeletConfig) (KubeletBootstrap, *config.PodConfig, error)

// NewKubeletServer will create a new KubeletServer with default values.
func NewKubeletServer() *KubeletServer {
	return &KubeletServer{
		SyncFrequency:               10 * time.Second,
		FileCheckFrequency:          20 * time.Second,
		HTTPCheckFrequency:          20 * time.Second,
		EnableServer:                true,
		Address:                     util.IP(net.ParseIP("0.0.0.0")),
		Port:                        ports.KubeletPort,
		ReadOnlyPort:                ports.KubeletReadOnlyPort,
		PodInfraContainerImage:      dockertools.PodInfraContainerImage,
		RootDirectory:               defaultRootDir,
		RegistryBurst:               10,
		EnableDebuggingHandlers:     true,
		MinimumGCAge:                1 * time.Minute,
		MaxPerPodContainerCount:     2,
		MaxContainerCount:           100,
		AuthPath:                    util.NewStringFlag("/var/lib/kubelet/kubernetes_auth"), // deprecated
		KubeConfig:                  util.NewStringFlag("/var/lib/kubelet/kubeconfig"),
		CadvisorPort:                4194,
		HealthzPort:                 10248,
		HealthzBindAddress:          util.IP(net.ParseIP("127.0.0.1")),
		RegisterNode:                true, // will be ignored if no apiserver is configured
		OOMScoreAdj:                 -900,
		MasterServiceNamespace:      api.NamespaceDefault,
		ImageGCHighThresholdPercent: 90,
		ImageGCLowThresholdPercent:  80,
		LowDiskSpaceThresholdMB:     256,
		NetworkPluginName:           "",
		HostNetworkSources:          kubelet.FileSource,
		CertDirectory:               "/var/run/kubernetes",
		NodeStatusUpdateFrequency:   10 * time.Second,
		ResourceContainer:           "/kubelet",
		CgroupRoot:                  "",
		ContainerRuntime:            "docker",
		DockerDaemonContainer:       "/docker-daemon",
		SystemContainer:             "",
		ConfigureCBR0:               false,
		DockerExecHandlerName:       "native",
	}
}

// AddFlags adds flags for a specific KubeletServer to the specified FlagSet
func (s *KubeletServer) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.Config, "config", s.Config, "Path to the config file or directory of files")
	fs.DurationVar(&s.SyncFrequency, "sync-frequency", s.SyncFrequency, "Max period between synchronizing running containers and config")
	fs.DurationVar(&s.FileCheckFrequency, "file-check-frequency", s.FileCheckFrequency, "Duration between checking config files for new data")
	fs.DurationVar(&s.HTTPCheckFrequency, "http-check-frequency", s.HTTPCheckFrequency, "Duration between checking http for new data")
	fs.StringVar(&s.ManifestURL, "manifest-url", s.ManifestURL, "URL for accessing the container manifest")
	fs.BoolVar(&s.EnableServer, "enable-server", s.EnableServer, "Enable the Kubelet's server")
	fs.Var(&s.Address, "address", "The IP address for the Kubelet to serve on (set to 0.0.0.0 for all interfaces)")
	fs.UintVar(&s.Port, "port", s.Port, "The port for the Kubelet to serve on. Note that \"kubectl logs\" will not work if you set this flag.") // see #9325
	fs.UintVar(&s.ReadOnlyPort, "read-only-port", s.ReadOnlyPort, "The read-only port for the Kubelet to serve on (set to 0 to disable)")
	fs.StringVar(&s.TLSCertFile, "tls-cert-file", s.TLSCertFile, ""+
		"File containing x509 Certificate for HTTPS.  (CA cert, if any, concatenated after server cert). "+
		"If --tls_cert_file and --tls_private_key_file are not provided, a self-signed certificate and key "+
		"are generated for the public address and saved to the directory passed to --cert_dir.")
	fs.StringVar(&s.TLSPrivateKeyFile, "tls-private-key-file", s.TLSPrivateKeyFile, "File containing x509 private key matching --tls_cert_file.")
	fs.StringVar(&s.CertDirectory, "cert-dir", s.CertDirectory, "The directory where the TLS certs are located (by default /var/run/kubernetes). "+
		"If --tls_cert_file and --tls_private_key_file are provided, this flag will be ignored.")
	fs.StringVar(&s.HostnameOverride, "hostname-override", s.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.StringVar(&s.PodInfraContainerImage, "pod-infra-container-image", s.PodInfraContainerImage, "The image whose network/ipc namespaces containers in each pod will use.")
	fs.StringVar(&s.DockerEndpoint, "docker-endpoint", s.DockerEndpoint, "If non-empty, use this for the docker endpoint to communicate with")
	fs.StringVar(&s.RootDirectory, "root-dir", s.RootDirectory, "Directory path for managing kubelet files (volume mounts,etc).")
	fs.BoolVar(&s.AllowPrivileged, "allow-privileged", s.AllowPrivileged, "If true, allow containers to request privileged mode. [default=false]")
	fs.StringVar(&s.HostNetworkSources, "host-network-sources", s.HostNetworkSources, "Comma-separated list of sources from which the Kubelet allows pods to use of host network. For all sources use \"*\" [default=\"file\"]")
	fs.Float64Var(&s.RegistryPullQPS, "registry-qps", s.RegistryPullQPS, "If > 0, limit registry pull QPS to this value.  If 0, unlimited. [default=0.0]")
	fs.IntVar(&s.RegistryBurst, "registry-burst", s.RegistryBurst, "Maximum size of a bursty pulls, temporarily allows pulls to burst to this number, while still not exceeding registry_qps.  Only used if --registry_qps > 0")
	fs.BoolVar(&s.RunOnce, "runonce", s.RunOnce, "If true, exit after spawning pods from local manifests or remote urls. Exclusive with --api_servers, and --enable-server")
	fs.BoolVar(&s.EnableDebuggingHandlers, "enable-debugging-handlers", s.EnableDebuggingHandlers, "Enables server endpoints for log collection and local running of containers and commands")
	fs.DurationVar(&s.MinimumGCAge, "minimum-container-ttl-duration", s.MinimumGCAge, "Minimum age for a finished container before it is garbage collected.  Examples: '300ms', '10s' or '2h45m'")
	fs.IntVar(&s.MaxPerPodContainerCount, "maximum-dead-containers-per-container", s.MaxPerPodContainerCount, "Maximum number of old instances of a container to retain per container.  Each container takes up some disk space.  Default: 2.")
	fs.IntVar(&s.MaxContainerCount, "maximum-dead-containers", s.MaxContainerCount, "Maximum number of old instances of a containers to retain globally.  Each container takes up some disk space.  Default: 100.")
	fs.Var(&s.AuthPath, "auth-path", "Path to .kubernetes_auth file, specifying how to authenticate to API server.")
	fs.MarkDeprecated("auth-path", "will be removed in a future version")
	fs.Var(&s.KubeConfig, "kubeconfig", "Path to a kubeconfig file, specifying how to authenticate to API server (the master location is set by the api-servers flag).")
	fs.UintVar(&s.CadvisorPort, "cadvisor-port", s.CadvisorPort, "The port of the localhost cAdvisor endpoint")
	fs.IntVar(&s.HealthzPort, "healthz-port", s.HealthzPort, "The port of the localhost healthz endpoint")
	fs.Var(&s.HealthzBindAddress, "healthz-bind-address", "The IP address for the healthz server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)")
	fs.IntVar(&s.OOMScoreAdj, "oom-score-adj", s.OOMScoreAdj, "The oom_score_adj value for kubelet process. Values must be within the range [-1000, 1000]")
	fs.Var(&s.APIServerList, "api-servers", "List of Kubernetes API servers for publishing events, and reading pods and services. (ip:port), comma separated.")
	fs.BoolVar(&s.RegisterNode, "register-node", s.RegisterNode, "Register the node with the apiserver (defaults to true if --api-server is set)")
	fs.StringVar(&s.ClusterDomain, "cluster-domain", s.ClusterDomain, "Domain for this cluster.  If set, kubelet will configure all containers to search this domain in addition to the host's search domains")
	fs.StringVar(&s.MasterServiceNamespace, "master-service-namespace", s.MasterServiceNamespace, "The namespace from which the kubernetes master services should be injected into pods")
	fs.Var(&s.ClusterDNS, "cluster-dns", "IP address for a cluster DNS server.  If set, kubelet will configure all containers to use this for DNS resolution in addition to the host's DNS servers")
	fs.DurationVar(&s.StreamingConnectionIdleTimeout, "streaming-connection-idle-timeout", 0, "Maximum time a streaming connection can be idle before the connection is automatically closed.  Example: '5m'")
	fs.DurationVar(&s.NodeStatusUpdateFrequency, "node-status-update-frequency", s.NodeStatusUpdateFrequency, "Specifies how often kubelet posts node status to master. Note: be cautious when changing the constant, it must work with nodeMonitorGracePeriod in nodecontroller. Default: 10s")
	fs.IntVar(&s.ImageGCHighThresholdPercent, "image-gc-high-threshold", s.ImageGCHighThresholdPercent, "The percent of disk usage after which image garbage collection is always run. Default: 90%%")
	fs.IntVar(&s.ImageGCLowThresholdPercent, "image-gc-low-threshold", s.ImageGCLowThresholdPercent, "The percent of disk usage before which image garbage collection is never run. Lowest disk usage to garbage collect to. Default: 80%%")
	fs.IntVar(&s.LowDiskSpaceThresholdMB, "low-diskspace-threshold-mb", s.LowDiskSpaceThresholdMB, "The absolute free disk space, in MB, to maintain. When disk space falls below this threshold, new pods would be rejected. Default: 256")
	fs.StringVar(&s.NetworkPluginName, "network-plugin", s.NetworkPluginName, "<Warning: Alpha feature> The name of the network plugin to be invoked for various events in kubelet/pod lifecycle")
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider, "The provider for cloud services.  Empty string for no provider.")
	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	fs.StringVar(&s.ResourceContainer, "resource-container", s.ResourceContainer, "Absolute name of the resource-only container to create and run the Kubelet in (Default: /kubelet).")
	fs.StringVar(&s.CgroupRoot, "cgroup_root", s.CgroupRoot, "Optional root cgroup to use for pods. This is handled by the container runtime on a best effort basis. Default: '', which means use the container runtime default.")
	fs.StringVar(&s.ContainerRuntime, "container_runtime", s.ContainerRuntime, "The container runtime to use. Possible values: 'docker', 'rkt'. Default: 'docker'.")
	fs.StringVar(&s.SystemContainer, "system-container", s.SystemContainer, "Optional resource-only container in which to place all non-kernel processes that are not already in a container. Empty for no container. Rolling back the flag requires a reboot. (Default: \"\").")
	fs.BoolVar(&s.ConfigureCBR0, "configure-cbr0", s.ConfigureCBR0, "If true, kubelet will configure cbr0 based on Node.Spec.PodCIDR.")
	fs.IntVar(&s.MaxPods, "max-pods", 40, "Number of Pods that can run on this Kubelet.")
	fs.StringVar(&s.DockerExecHandlerName, "docker-exec-handler", s.DockerExecHandlerName, "Handler to use when executing a command in a container. Valid values are 'native' and 'nsenter'. Defaults to 'native'.")
	fs.StringVar(&s.PodCIDR, "pod-cidr", "", "The CIDR to use for pod IP addresses, only used in standalone mode.  In cluster mode, this is obtained from the master.")
	// Flags intended for testing, not recommended used in production environments.
	fs.BoolVar(&s.ReallyCrashForTesting, "really-crash-for-testing", s.ReallyCrashForTesting, "If true, when panics occur crash. Intended for testing.")
	fs.Float64Var(&s.ChaosChance, "chaos-chance", s.ChaosChance, "If > 0.0, introduce random client errors and latency. Intended for testing. [default=0.0]")
	fs.BoolVar(&s.Containerized, "containerized", s.Containerized, "Experimental support for running kubelet in a container.  Intended for testing. [default=false]")
}

// Run runs the specified KubeletServer.  This should never exit.
func (s *KubeletServer) Run(_ []string) error {
	util.ReallyCrash = s.ReallyCrashForTesting
	rand.Seed(time.Now().UTC().UnixNano())

	// TODO(vmarmol): Do this through container config.
	if err := util.ApplyOomScoreAdj(0, s.OOMScoreAdj); err != nil {
		glog.Warning(err)
	}

	var apiclient *client.Client
	clientConfig, err := s.CreateAPIServerClientConfig()
	if err == nil {
		apiclient, err = client.New(clientConfig)
	}
	if err != nil && len(s.APIServerList) > 0 {
		glog.Warningf("No API client: %v", err)
	}

	glog.V(2).Infof("Using root directory: %v", s.RootDirectory)

	credentialprovider.SetPreferredDockercfgPath(s.RootDirectory)

	cadvisorInterface, err := cadvisor.New(s.CadvisorPort)
	if err != nil {
		return err
	}

	imageGCPolicy := kubelet.ImageGCPolicy{
		HighThresholdPercent: s.ImageGCHighThresholdPercent,
		LowThresholdPercent:  s.ImageGCLowThresholdPercent,
	}

	diskSpacePolicy := kubelet.DiskSpacePolicy{
		DockerFreeDiskMB: s.LowDiskSpaceThresholdMB,
		RootFreeDiskMB:   s.LowDiskSpaceThresholdMB,
	}
	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		return err
	}
	glog.V(2).Infof("Successfully initialized cloud provider: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)

	hostNetworkSources, err := kubelet.GetValidatedSources(strings.Split(s.HostNetworkSources, ","))
	if err != nil {
		return err
	}

	tlsOptions, err := s.InitializeTLS()
	if err != nil {
		return err
	}

	mounter := mount.New()
	if s.Containerized {
		glog.V(2).Info("Running kubelet in containerized mode (experimental)")
		mounter = &mount.NsenterMounter{}
	}

	var dockerExecHandler dockertools.ExecHandler
	switch s.DockerExecHandlerName {
	case "native":
		dockerExecHandler = &dockertools.NativeExecHandler{}
	case "nsenter":
		dockerExecHandler = &dockertools.NsenterExecHandler{}
	default:
		glog.Warningf("Unknown Docker exec handler %q; defaulting to native", s.DockerExecHandlerName)
		dockerExecHandler = &dockertools.NativeExecHandler{}
	}

	kcfg := KubeletConfig{
		Address:                        s.Address,
		AllowPrivileged:                s.AllowPrivileged,
		HostNetworkSources:             hostNetworkSources,
		HostnameOverride:               s.HostnameOverride,
		RootDirectory:                  s.RootDirectory,
		ConfigFile:                     s.Config,
		ManifestURL:                    s.ManifestURL,
		FileCheckFrequency:             s.FileCheckFrequency,
		HTTPCheckFrequency:             s.HTTPCheckFrequency,
		PodInfraContainerImage:         s.PodInfraContainerImage,
		SyncFrequency:                  s.SyncFrequency,
		RegistryPullQPS:                s.RegistryPullQPS,
		RegistryBurst:                  s.RegistryBurst,
		MinimumGCAge:                   s.MinimumGCAge,
		MaxPerPodContainerCount:        s.MaxPerPodContainerCount,
		MaxContainerCount:              s.MaxContainerCount,
		RegisterNode:                   s.RegisterNode,
		StandaloneMode:                 (len(s.APIServerList) == 0),
		ClusterDomain:                  s.ClusterDomain,
		ClusterDNS:                     s.ClusterDNS,
		Runonce:                        s.RunOnce,
		Port:                           s.Port,
		ReadOnlyPort:                   s.ReadOnlyPort,
		CadvisorInterface:              cadvisorInterface,
		EnableServer:                   s.EnableServer,
		EnableDebuggingHandlers:        s.EnableDebuggingHandlers,
		DockerClient:                   dockertools.ConnectToDockerOrDie(s.DockerEndpoint),
		KubeClient:                     apiclient,
		MasterServiceNamespace:         s.MasterServiceNamespace,
		VolumePlugins:                  ProbeVolumePlugins(),
		NetworkPlugins:                 ProbeNetworkPlugins(),
		NetworkPluginName:              s.NetworkPluginName,
		StreamingConnectionIdleTimeout: s.StreamingConnectionIdleTimeout,
		TLSOptions:                     tlsOptions,
		ImageGCPolicy:                  imageGCPolicy,
		DiskSpacePolicy:                diskSpacePolicy,
		Cloud:                          cloud,
		NodeStatusUpdateFrequency: s.NodeStatusUpdateFrequency,
		ResourceContainer:         s.ResourceContainer,
		CgroupRoot:                s.CgroupRoot,
		ContainerRuntime:          s.ContainerRuntime,
		Mounter:                   mounter,
		DockerDaemonContainer:     s.DockerDaemonContainer,
		SystemContainer:           s.SystemContainer,
		ConfigureCBR0:             s.ConfigureCBR0,
		PodCIDR:                   s.PodCIDR,
		MaxPods:                   s.MaxPods,
		DockerExecHandler:         dockerExecHandler,
	}

	if err := RunKubelet(&kcfg, nil); err != nil {
		return err
	}

	if s.HealthzPort > 0 {
		healthz.DefaultHealthz()
		go util.Forever(func() {
			err := http.ListenAndServe(net.JoinHostPort(s.HealthzBindAddress.String(), strconv.Itoa(s.HealthzPort)), nil)
			if err != nil {
				glog.Errorf("Starting health server failed: %v", err)
			}
		}, 5*time.Second)
	}

	if s.RunOnce {
		return nil
	}

	// run forever
	select {}
}

// InitializeTLS checks for a configured TLSCertFile and TLSPrivateKeyFile: if unspecified a new self-signed
// certificate and key file are generated. Returns a configured kubelet.TLSOptions object.
func (s *KubeletServer) InitializeTLS() (*kubelet.TLSOptions, error) {
	if s.TLSCertFile == "" && s.TLSPrivateKeyFile == "" {
		s.TLSCertFile = path.Join(s.CertDirectory, "kubelet.crt")
		s.TLSPrivateKeyFile = path.Join(s.CertDirectory, "kubelet.key")
		if err := util.GenerateSelfSignedCert(nodeutil.GetHostname(s.HostnameOverride), s.TLSCertFile, s.TLSPrivateKeyFile, nil, nil); err != nil {
			return nil, fmt.Errorf("unable to generate self signed cert: %v", err)
		}
		glog.V(4).Infof("Using self-signed cert (%s, %s)", s.TLSCertFile, s.TLSPrivateKeyFile)
	}
	tlsOptions := &kubelet.TLSOptions{
		Config: &tls.Config{
			// Change default from SSLv3 to TLSv1.0 (because of POODLE vulnerability).
			MinVersion: tls.VersionTLS10,
			// Populate PeerCertificates in requests, but don't yet reject connections without certificates.
			ClientAuth: tls.RequestClientCert,
		},
		CertFile: s.TLSCertFile,
		KeyFile:  s.TLSPrivateKeyFile,
	}
	return tlsOptions, nil
}

func (s *KubeletServer) authPathClientConfig(useDefaults bool) (*client.Config, error) {
	authInfo, err := clientauth.LoadFromFile(s.AuthPath.Value())
	if err != nil && !useDefaults {
		return nil, err
	}
	// If loading the default auth path, for backwards compatibility keep going
	// with the default auth.
	if err != nil {
		glog.Warningf("Could not load kubernetes auth path %s: %v. Continuing with defaults.", s.AuthPath, err)
	}
	if authInfo == nil {
		// authInfo didn't load correctly - continue with defaults.
		authInfo = &clientauth.Info{}
	}
	authConfig, err := authInfo.MergeWithConfig(client.Config{})
	if err != nil {
		return nil, err
	}
	authConfig.Host = s.APIServerList[0]
	return &authConfig, nil
}

func (s *KubeletServer) kubeconfigClientConfig() (*client.Config, error) {
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: s.KubeConfig.Value()},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: s.APIServerList[0]}}).ClientConfig()
}

// createClientConfig creates a client configuration from the command line
// arguments. If either --auth-path or --kubeconfig is explicitly set, it
// will be used (setting both is an error). If neither are set first attempt
// to load the default kubeconfig file, then the default auth path file, and
// fall back to the default auth (none) without an error.
// TODO(roberthbailey): Remove support for --auth-path
func (s *KubeletServer) createClientConfig() (*client.Config, error) {
	if s.KubeConfig.Provided() && s.AuthPath.Provided() {
		return nil, fmt.Errorf("cannot specify both --kubeconfig and --auth-path")
	}
	if s.KubeConfig.Provided() {
		return s.kubeconfigClientConfig()
	} else if s.AuthPath.Provided() {
		return s.authPathClientConfig(false)
	}
	// Try the kubeconfig default first, falling back to the auth path default.
	clientConfig, err := s.kubeconfigClientConfig()
	if err != nil {
		glog.Warningf("Could not load kubeconfig file %s: %v. Trying auth path instead.", s.KubeConfig, err)
		return s.authPathClientConfig(true)
	}
	return clientConfig, nil
}

// CreateAPIServerClientConfig generates a client.Config from command line flags,
// including api-server-list, via createClientConfig and then injects chaos into
// the configuration via addChaosToClientConfig. This func is exported to support
// integration with third party kubelet extensions (e.g. kubernetes-mesos).
func (s *KubeletServer) CreateAPIServerClientConfig() (*client.Config, error) {
	if len(s.APIServerList) < 1 {
		return nil, fmt.Errorf("no api servers specified")
	}
	// TODO: adapt Kube client to support LB over several servers
	if len(s.APIServerList) > 1 {
		glog.Infof("Multiple api servers specified.  Picking first one")
	}

	clientConfig, err := s.createClientConfig()
	if err != nil {
		return nil, err
	}
	s.addChaosToClientConfig(clientConfig)
	return clientConfig, nil
}

// addChaosToClientConfig injects random errors into client connections if configured.
func (s *KubeletServer) addChaosToClientConfig(config *client.Config) {
	if s.ChaosChance != 0.0 {
		config.WrapTransport = func(rt http.RoundTripper) http.RoundTripper {
			seed := chaosclient.NewSeed(1)
			// TODO: introduce a standard chaos package with more tunables - this is just a proof of concept
			// TODO: introduce random latency and stalls
			return chaosclient.NewChaosRoundTripper(rt, chaosclient.LogChaos, seed.P(s.ChaosChance, chaosclient.ErrSimulatedConnectionResetByPeer))
		}
	}
}

// SimpleRunKubelet is a simple way to start a Kubelet talking to dockerEndpoint, using an API Client.
// Under the hood it calls RunKubelet (below)
func SimpleKubelet(client *client.Client,
	dockerClient dockertools.DockerInterface,
	hostname, rootDir, manifestURL, address string,
	port uint,
	masterServiceNamespace string,
	volumePlugins []volume.VolumePlugin,
	tlsOptions *kubelet.TLSOptions,
	cadvisorInterface cadvisor.Interface,
	configFilePath string,
	cloud cloudprovider.Interface,
	osInterface kubecontainer.OSInterface) *KubeletConfig {

	imageGCPolicy := kubelet.ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	diskSpacePolicy := kubelet.DiskSpacePolicy{
		DockerFreeDiskMB: 256,
		RootFreeDiskMB:   256,
	}
	kcfg := KubeletConfig{
		KubeClient:             client,
		DockerClient:           dockerClient,
		HostnameOverride:       hostname,
		RootDirectory:          rootDir,
		ManifestURL:            manifestURL,
		PodInfraContainerImage: dockertools.PodInfraContainerImage,
		Port:                    port,
		Address:                 util.IP(net.ParseIP(address)),
		EnableServer:            true,
		EnableDebuggingHandlers: true,
		HTTPCheckFrequency:      1 * time.Second,
		FileCheckFrequency:      1 * time.Second,
		SyncFrequency:           3 * time.Second,
		MinimumGCAge:            10 * time.Second,
		MaxPerPodContainerCount: 2,
		MaxContainerCount:       100,
		RegisterNode:            true,
		MasterServiceNamespace:  masterServiceNamespace,
		VolumePlugins:           volumePlugins,
		TLSOptions:              tlsOptions,
		CadvisorInterface:       cadvisorInterface,
		ConfigFile:              configFilePath,
		ImageGCPolicy:           imageGCPolicy,
		DiskSpacePolicy:         diskSpacePolicy,
		Cloud:                   cloud,
		NodeStatusUpdateFrequency: 10 * time.Second,
		ResourceContainer:         "/kubelet",
		OSInterface:               osInterface,
		CgroupRoot:                "",
		ContainerRuntime:          "docker",
		Mounter:                   mount.New(),
		DockerDaemonContainer:     "/docker-daemon",
		SystemContainer:           "",
		MaxPods:                   32,
		DockerExecHandler:         &dockertools.NativeExecHandler{},
	}
	return &kcfg
}

// RunKubelet is responsible for setting up and running a kubelet.  It is used in three different applications:
//   1 Integration tests
//   2 Kubelet binary
//   3 Standalone 'kubernetes' binary
// Eventually, #2 will be replaced with instances of #3
func RunKubelet(kcfg *KubeletConfig, builder KubeletBuilder) error {
	kcfg.Hostname = nodeutil.GetHostname(kcfg.HostnameOverride)

	if len(kcfg.NodeName) == 0 {
		// Query the cloud provider for our node name, default to Hostname
		nodeName := kcfg.Hostname
		if kcfg.Cloud != nil {
			var err error
			instances, ok := kcfg.Cloud.Instances()
			if !ok {
				return fmt.Errorf("failed to get instances from cloud provider")
			}

			nodeName, err = instances.CurrentNodeName(kcfg.Hostname)
			if err != nil {
				return fmt.Errorf("error fetching current instance name from cloud provider: %v", err)
			}

			glog.V(2).Infof("cloud provider determined current node name to be %s", nodeName)
		}

		kcfg.NodeName = nodeName
	}

	eventBroadcaster := record.NewBroadcaster()
	kcfg.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: "kubelet", Host: kcfg.NodeName})
	eventBroadcaster.StartLogging(glog.V(3).Infof)
	if kcfg.KubeClient != nil {
		glog.V(4).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(kcfg.KubeClient.Events(""))
	} else {
		glog.Warning("No api server defined - no events will be sent to API server.")
	}
	capabilities.Setup(kcfg.AllowPrivileged, kcfg.HostNetworkSources)

	credentialprovider.SetPreferredDockercfgPath(kcfg.RootDirectory)

	if builder == nil {
		builder = createAndInitKubelet
	}
	if kcfg.OSInterface == nil {
		kcfg.OSInterface = kubecontainer.RealOS{}
	}
	k, podCfg, err := builder(kcfg)
	if err != nil {
		return fmt.Errorf("failed to create kubelet: %v", err)
	}
	// process pods and exit.
	if kcfg.Runonce {
		if _, err := k.RunOnce(podCfg.Updates()); err != nil {
			return fmt.Errorf("runonce failed: %v", err)
		}
		glog.Infof("Started kubelet as runonce")
	} else {
		startKubelet(k, podCfg, kcfg)
		glog.Infof("Started kubelet")
	}
	return nil
}

func startKubelet(k KubeletBootstrap, podCfg *config.PodConfig, kc *KubeletConfig) {
	// start the kubelet
	go util.Forever(func() { k.Run(podCfg.Updates()) }, 0)

	// start the kubelet server
	if kc.EnableServer {
		go util.Forever(func() {
			k.ListenAndServe(net.IP(kc.Address), kc.Port, kc.TLSOptions, kc.EnableDebuggingHandlers)
		}, 0)
	}
	if kc.ReadOnlyPort > 0 {
		go util.Forever(func() {
			k.ListenAndServeReadOnly(net.IP(kc.Address), kc.ReadOnlyPort)
		}, 0)
	}
}

func makePodSourceConfig(kc *KubeletConfig) *config.PodConfig {
	// source of all configuration
	cfg := config.NewPodConfig(config.PodConfigNotificationSnapshotAndUpdates, kc.Recorder)

	// define file config source
	if kc.ConfigFile != "" {
		glog.Infof("Adding manifest file: %v", kc.ConfigFile)
		config.NewSourceFile(kc.ConfigFile, kc.NodeName, kc.FileCheckFrequency, cfg.Channel(kubelet.FileSource))
	}

	// define url config source
	if kc.ManifestURL != "" {
		glog.Infof("Adding manifest url: %v", kc.ManifestURL)
		config.NewSourceURL(kc.ManifestURL, kc.NodeName, kc.HTTPCheckFrequency, cfg.Channel(kubelet.HTTPSource))
	}
	if kc.KubeClient != nil {
		glog.Infof("Watching apiserver")
		config.NewSourceApiserver(kc.KubeClient, kc.NodeName, cfg.Channel(kubelet.ApiserverSource))
	}
	return cfg
}

// KubeletConfig is all of the parameters necessary for running a kubelet.
// TODO: This should probably be merged with KubeletServer.  The extra object is a consequence of refactoring.
type KubeletConfig struct {
	KubeClient                     *client.Client
	DockerClient                   dockertools.DockerInterface
	CadvisorInterface              cadvisor.Interface
	Address                        util.IP
	AllowPrivileged                bool
	HostNetworkSources             []string
	HostnameOverride               string
	RootDirectory                  string
	ConfigFile                     string
	ManifestURL                    string
	FileCheckFrequency             time.Duration
	HTTPCheckFrequency             time.Duration
	Hostname                       string
	NodeName                       string
	PodInfraContainerImage         string
	SyncFrequency                  time.Duration
	RegistryPullQPS                float64
	RegistryBurst                  int
	MinimumGCAge                   time.Duration
	MaxPerPodContainerCount        int
	MaxContainerCount              int
	RegisterNode                   bool
	StandaloneMode                 bool
	ClusterDomain                  string
	ClusterDNS                     util.IP
	EnableServer                   bool
	EnableDebuggingHandlers        bool
	Port                           uint
	ReadOnlyPort                   uint
	Runonce                        bool
	MasterServiceNamespace         string
	VolumePlugins                  []volume.VolumePlugin
	NetworkPlugins                 []network.NetworkPlugin
	NetworkPluginName              string
	StreamingConnectionIdleTimeout time.Duration
	Recorder                       record.EventRecorder
	TLSOptions                     *kubelet.TLSOptions
	ImageGCPolicy                  kubelet.ImageGCPolicy
	DiskSpacePolicy                kubelet.DiskSpacePolicy
	Cloud                          cloudprovider.Interface
	NodeStatusUpdateFrequency      time.Duration
	ResourceContainer              string
	OSInterface                    kubecontainer.OSInterface
	CgroupRoot                     string
	ContainerRuntime               string
	Mounter                        mount.Interface
	DockerDaemonContainer          string
	SystemContainer                string
	ConfigureCBR0                  bool
	PodCIDR                        string
	MaxPods                        int
	DockerExecHandler              dockertools.ExecHandler
}

func createAndInitKubelet(kc *KubeletConfig) (k KubeletBootstrap, pc *config.PodConfig, err error) {
	// TODO: block until all sources have delivered at least one update to the channel, or break the sync loop
	// up into "per source" synchronizations
	// TODO: KubeletConfig.KubeClient should be a client interface, but client interface misses certain methods
	// used by kubelet. Since NewMainKubelet expects a client interface, we need to make sure we are not passing
	// a nil pointer to it when what we really want is a nil interface.
	var kubeClient client.Interface
	if kc.KubeClient != nil {
		kubeClient = kc.KubeClient
	}

	gcPolicy := kubelet.ContainerGCPolicy{
		MinAge:             kc.MinimumGCAge,
		MaxPerPodContainer: kc.MaxPerPodContainerCount,
		MaxContainers:      kc.MaxContainerCount,
	}

	pc = makePodSourceConfig(kc)
	k, err = kubelet.NewMainKubelet(
		kc.Hostname,
		kc.NodeName,
		kc.DockerClient,
		kubeClient,
		kc.RootDirectory,
		kc.PodInfraContainerImage,
		kc.SyncFrequency,
		float32(kc.RegistryPullQPS),
		kc.RegistryBurst,
		gcPolicy,
		pc.SeenAllSources,
		kc.RegisterNode,
		kc.StandaloneMode,
		kc.ClusterDomain,
		net.IP(kc.ClusterDNS),
		kc.MasterServiceNamespace,
		kc.VolumePlugins,
		kc.NetworkPlugins,
		kc.NetworkPluginName,
		kc.StreamingConnectionIdleTimeout,
		kc.Recorder,
		kc.CadvisorInterface,
		kc.ImageGCPolicy,
		kc.DiskSpacePolicy,
		kc.Cloud,
		kc.NodeStatusUpdateFrequency,
		kc.ResourceContainer,
		kc.OSInterface,
		kc.CgroupRoot,
		kc.ContainerRuntime,
		kc.Mounter,
		kc.DockerDaemonContainer,
		kc.SystemContainer,
		kc.ConfigureCBR0,
		kc.PodCIDR,
		kc.MaxPods,
		kc.DockerExecHandler)

	if err != nil {
		return nil, nil, err
	}

	k.BirthCry()

	k.StartGarbageCollection()

	return k, pc, nil
}
