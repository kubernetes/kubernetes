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

// Package app makes it easy to create a kubelet server for various contexts.
package app

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	kubeExternal "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/client/chaosclient"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	clientauth "k8s.io/kubernetes/pkg/client/unversioned/auth"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/server"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	utilconfig "k8s.io/kubernetes/pkg/util/config"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/util/crypto"
	"k8s.io/kubernetes/pkg/util/flock"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/rlimit"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/volume"
)

// bootstrapping interface for kubelet, targets the initialization protocol
type KubeletBootstrap interface {
	BirthCry()
	StartGarbageCollection()
	ListenAndServe(address net.IP, port uint, tlsOptions *server.TLSOptions, auth server.AuthInterface, enableDebuggingHandlers bool)
	ListenAndServeReadOnly(address net.IP, port uint)
	Run(<-chan kubetypes.PodUpdate)
	RunOnce(<-chan kubetypes.PodUpdate) ([]kubelet.RunPodResult, error)
}

// create and initialize a Kubelet instance
type KubeletBuilder func(kc *KubeletConfig) (KubeletBootstrap, *config.PodConfig, error)

// NewKubeletCommand creates a *cobra.Command object with default parameters
func NewKubeletCommand() *cobra.Command {
	s := options.NewKubeletServer()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "kubelet",
		Long: `The kubelet is the primary "node agent" that runs on each
node. The kubelet works in terms of a PodSpec. A PodSpec is a YAML or JSON object
that describes a pod. The kubelet takes a set of PodSpecs that are provided through
various mechanisms (primarily through the apiserver) and ensures that the containers
described in those PodSpecs are running and healthy. The kubelet doesn't manage
containers which were not created by Kubernetes.

Other than from an PodSpec from the apiserver, there are three ways that a container
manifest can be provided to the Kubelet.

File: Path passed as a flag on the command line. This file is rechecked every 20
seconds (configurable with a flag).

HTTP endpoint: HTTP endpoint passed as a parameter on the command line. This endpoint
is checked every 20 seconds (also configurable with a flag).

HTTP server: The kubelet can also listen for HTTP and respond to a simple API
(underspec'd currently) to submit a new manifest.`,
		Run: func(cmd *cobra.Command, args []string) {
		},
	}

	return cmd
}

// UnsecuredKubeletConfig returns a KubeletConfig suitable for being run, or an error if the server setup
// is not valid.  It will not start any background processes, and does not include authentication/authorization
func UnsecuredKubeletConfig(s *options.KubeletServer) (*KubeletConfig, error) {
	hostNetworkSources, err := kubetypes.GetValidatedSources(s.HostNetworkSources)
	if err != nil {
		return nil, err
	}

	hostPIDSources, err := kubetypes.GetValidatedSources(s.HostPIDSources)
	if err != nil {
		return nil, err
	}

	hostIPCSources, err := kubetypes.GetValidatedSources(s.HostIPCSources)
	if err != nil {
		return nil, err
	}

	mounter := mount.New()
	var writer io.Writer = &io.StdWriter{}
	if s.Containerized {
		glog.V(2).Info("Running kubelet in containerized mode (experimental)")
		mounter = mount.NewNsenterMounter()
		writer = &io.NsenterWriter{}
	}

	tlsOptions, err := InitializeTLS(s)
	if err != nil {
		return nil, err
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

	imageGCPolicy := images.ImageGCPolicy{
		MinAge:               s.ImageMinimumGCAge.Duration,
		HighThresholdPercent: int(s.ImageGCHighThresholdPercent),
		LowThresholdPercent:  int(s.ImageGCLowThresholdPercent),
	}

	diskSpacePolicy := kubelet.DiskSpacePolicy{
		DockerFreeDiskMB: int(s.LowDiskSpaceThresholdMB),
		RootFreeDiskMB:   int(s.LowDiskSpaceThresholdMB),
	}

	manifestURLHeader := make(http.Header)
	if s.ManifestURLHeader != "" {
		pieces := strings.Split(s.ManifestURLHeader, ":")
		if len(pieces) != 2 {
			return nil, fmt.Errorf("manifest-url-header must have a single ':' key-value separator, got %q", s.ManifestURLHeader)
		}
		manifestURLHeader.Set(pieces[0], pieces[1])
	}

	reservation, err := parseReservation(s.KubeReserved, s.SystemReserved)
	if err != nil {
		return nil, err
	}

	thresholds, err := eviction.ParseThresholdConfig(s.EvictionHard, s.EvictionSoft, s.EvictionSoftGracePeriod, s.EvictionMinimumReclaim)
	if err != nil {
		return nil, err
	}
	evictionConfig := eviction.Config{
		PressureTransitionPeriod: s.EvictionPressureTransitionPeriod.Duration,
		MaxPodGracePeriodSeconds: int64(s.EvictionMaxPodGracePeriod),
		Thresholds:               thresholds,
	}

	return &KubeletConfig{
		Address:                      net.ParseIP(s.Address),
		AllowPrivileged:              s.AllowPrivileged,
		Auth:                         nil, // default does not enforce auth[nz]
		CAdvisorInterface:            nil, // launches background processes, not set here
		VolumeStatsAggPeriod:         s.VolumeStatsAggPeriod.Duration,
		CgroupRoot:                   s.CgroupRoot,
		Cloud:                        nil, // cloud provider might start background processes
		ClusterDNS:                   net.ParseIP(s.ClusterDNS),
		ClusterDomain:                s.ClusterDomain,
		ConfigFile:                   s.PodManifestPath,
		ConfigureCBR0:                s.ConfigureCBR0,
		ContainerManager:             nil,
		ContainerRuntime:             s.ContainerRuntime,
		RuntimeRequestTimeout:        s.RuntimeRequestTimeout.Duration,
		CPUCFSQuota:                  s.CPUCFSQuota,
		DiskSpacePolicy:              diskSpacePolicy,
		DockerClient:                 dockertools.ConnectToDockerOrDie(s.DockerEndpoint, s.RuntimeRequestTimeout.Duration), // TODO(random-liu): Set RuntimeRequestTimeout for rkt.
		RuntimeCgroups:               s.RuntimeCgroups,
		DockerExecHandler:            dockerExecHandler,
		EnableControllerAttachDetach: s.EnableControllerAttachDetach,
		EnableCustomMetrics:          s.EnableCustomMetrics,
		EnableDebuggingHandlers:      s.EnableDebuggingHandlers,
		CgroupsPerQOS:                s.CgroupsPerQOS,
		EnableServer:                 s.EnableServer,
		EventBurst:                   int(s.EventBurst),
		EventRecordQPS:               float32(s.EventRecordQPS),
		FileCheckFrequency:           s.FileCheckFrequency.Duration,
		HostnameOverride:             s.HostnameOverride,
		HostNetworkSources:           hostNetworkSources,
		HostPIDSources:               hostPIDSources,
		HostIPCSources:               hostIPCSources,
		HTTPCheckFrequency:           s.HTTPCheckFrequency.Duration,
		ImageGCPolicy:                imageGCPolicy,
		KubeClient:                   nil,
		ManifestURL:                  s.ManifestURL,
		ManifestURLHeader:            manifestURLHeader,
		MasterServiceNamespace:       s.MasterServiceNamespace,
		MaxContainerCount:            int(s.MaxContainerCount),
		MaxOpenFiles:                 uint64(s.MaxOpenFiles),
		MaxPerPodContainerCount:      int(s.MaxPerPodContainerCount),
		MaxPods:                      int(s.MaxPods),
		NvidiaGPUs:                   int(s.NvidiaGPUs),
		MinimumGCAge:                 s.MinimumGCAge.Duration,
		Mounter:                      mounter,
		NetworkPluginName:            s.NetworkPluginName,
		NetworkPlugins:               ProbeNetworkPlugins(s.NetworkPluginDir),
		NodeLabels:                   s.NodeLabels,
		NodeStatusUpdateFrequency:    s.NodeStatusUpdateFrequency.Duration,
		NonMasqueradeCIDR:            s.NonMasqueradeCIDR,
		OOMAdjuster:                  oom.NewOOMAdjuster(),
		OSInterface:                  kubecontainer.RealOS{},
		PodCIDR:                      s.PodCIDR,
		ReconcileCIDR:                s.ReconcileCIDR,
		PodInfraContainerImage:       s.PodInfraContainerImage,
		Port:                           uint(s.Port),
		ReadOnlyPort:                   uint(s.ReadOnlyPort),
		RegisterNode:                   s.RegisterNode,
		RegisterSchedulable:            s.RegisterSchedulable,
		RegistryBurst:                  int(s.RegistryBurst),
		RegistryPullQPS:                float64(s.RegistryPullQPS),
		ResolverConfig:                 s.ResolverConfig,
		Reservation:                    *reservation,
		KubeletCgroups:                 s.KubeletCgroups,
		RktPath:                        s.RktPath,
		RktAPIEndpoint:                 s.RktAPIEndpoint,
		RktStage1Image:                 s.RktStage1Image,
		RootDirectory:                  s.RootDirectory,
		SeccompProfileRoot:             s.SeccompProfileRoot,
		Runonce:                        s.RunOnce,
		SerializeImagePulls:            s.SerializeImagePulls,
		StandaloneMode:                 (len(s.APIServerList) == 0),
		StreamingConnectionIdleTimeout: s.StreamingConnectionIdleTimeout.Duration,
		SyncFrequency:                  s.SyncFrequency.Duration,
		SystemCgroups:                  s.SystemCgroups,
		TLSOptions:                     tlsOptions,
		Writer:                         writer,
		VolumePlugins:                  ProbeVolumePlugins(s.VolumePluginDir),
		OutOfDiskTransitionFrequency:   s.OutOfDiskTransitionFrequency.Duration,
		HairpinMode:                    s.HairpinMode,
		BabysitDaemons:                 s.BabysitDaemons,
		ExperimentalFlannelOverlay:     s.ExperimentalFlannelOverlay,
		NodeIP:                net.ParseIP(s.NodeIP),
		EvictionConfig:        evictionConfig,
		PodsPerCore:           int(s.PodsPerCore),
		ProtectKernelDefaults: s.ProtectKernelDefaults,
	}, nil
}

// Run runs the specified KubeletServer for the given KubeletConfig.  This should never exit.
// The kcfg argument may be nil - if so, it is initialized from the settings on KubeletServer.
// Otherwise, the caller is assumed to have set up the KubeletConfig object and all defaults
// will be ignored.
func Run(s *options.KubeletServer, kcfg *KubeletConfig) error {
	err := run(s, kcfg)
	if err != nil {
		glog.Errorf("Failed running kubelet: %v", err)
	}
	return err
}

func checkPermissions() error {
	if uid := os.Getuid(); uid != 0 {
		return fmt.Errorf("Kubelet needs to run as uid `0`. It is being run as %d", uid)
	}
	// TODO: Check if kubelet is running in the `initial` user namespace.
	// http://man7.org/linux/man-pages/man7/user_namespaces.7.html
	return nil
}

func run(s *options.KubeletServer, kcfg *KubeletConfig) (err error) {
	if s.ExitOnLockContention && s.LockFilePath == "" {
		return errors.New("cannot exit on lock file contention: no lock file specified")
	}
	if err := checkPermissions(); err != nil {
		glog.Error(err)
	}

	done := make(chan struct{})
	if s.LockFilePath != "" {
		glog.Infof("acquiring lock on %q", s.LockFilePath)
		if err := flock.Acquire(s.LockFilePath); err != nil {
			return fmt.Errorf("unable to acquire file lock on %q: %v", s.LockFilePath, err)
		}
		if s.ExitOnLockContention {
			glog.Infof("watching for inotify events for: %v", s.LockFilePath)
			if err := watchForLockfileContention(s.LockFilePath, done); err != nil {
				return err
			}
		}
	}
	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(s.KubeletConfiguration)
	} else {
		glog.Errorf("unable to register configz: %s", err)
	}

	if kcfg == nil {
		cfg, err := UnsecuredKubeletConfig(s)
		if err != nil {
			return err
		}
		kcfg = cfg

		clientConfig, err := CreateAPIServerClientConfig(s)
		if err == nil {
			kcfg.KubeClient, err = clientset.NewForConfig(clientConfig)

			// make a separate client for events
			eventClientConfig := *clientConfig
			eventClientConfig.QPS = float32(s.EventRecordQPS)
			eventClientConfig.Burst = int(s.EventBurst)
			kcfg.EventClient, err = clientset.NewForConfig(&eventClientConfig)
		}
		if err != nil && len(s.APIServerList) > 0 {
			glog.Warningf("No API client: %v", err)
		}

		if s.CloudProvider == kubeExternal.AutoDetectCloudProvider {
			kcfg.AutoDetectCloudProvider = true
		} else {
			cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
			if err != nil {
				return err
			}
			if cloud == nil {
				glog.V(2).Infof("No cloud provider specified: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)
			} else {
				glog.V(2).Infof("Successfully initialized cloud provider: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)
				kcfg.Cloud = cloud
			}
		}
	}

	if kcfg.CAdvisorInterface == nil {
		kcfg.CAdvisorInterface, err = cadvisor.New(uint(s.CAdvisorPort), kcfg.ContainerRuntime)
		if err != nil {
			return err
		}
	}

	if kcfg.ContainerManager == nil {
		if kcfg.SystemCgroups != "" && kcfg.CgroupRoot == "" {
			return fmt.Errorf("invalid configuration: system container was specified and cgroup root was not specified")
		}
		kcfg.ContainerManager, err = cm.NewContainerManager(kcfg.Mounter, kcfg.CAdvisorInterface, cm.NodeConfig{
			RuntimeCgroupsName:    kcfg.RuntimeCgroups,
			SystemCgroupsName:     kcfg.SystemCgroups,
			KubeletCgroupsName:    kcfg.KubeletCgroups,
			ContainerRuntime:      kcfg.ContainerRuntime,
			CgroupsPerQOS:         kcfg.CgroupsPerQOS,
			CgroupRoot:            kcfg.CgroupRoot,
			ProtectKernelDefaults: kcfg.ProtectKernelDefaults,
		})
		if err != nil {
			return err
		}
	}

	runtime.ReallyCrash = s.ReallyCrashForTesting
	rand.Seed(time.Now().UTC().UnixNano())

	// TODO(vmarmol): Do this through container config.
	oomAdjuster := kcfg.OOMAdjuster
	if err := oomAdjuster.ApplyOOMScoreAdj(0, int(s.OOMScoreAdj)); err != nil {
		glog.Warning(err)
	}

	if err := RunKubelet(kcfg); err != nil {
		return err
	}

	if s.HealthzPort > 0 {
		healthz.DefaultHealthz()
		go wait.Until(func() {
			err := http.ListenAndServe(net.JoinHostPort(s.HealthzBindAddress, strconv.Itoa(int(s.HealthzPort))), nil)
			if err != nil {
				glog.Errorf("Starting health server failed: %v", err)
			}
		}, 5*time.Second, wait.NeverStop)
	}

	if s.RunOnce {
		return nil
	}

	<-done
	return nil
}

// InitializeTLS checks for a configured TLSCertFile and TLSPrivateKeyFile: if unspecified a new self-signed
// certificate and key file are generated. Returns a configured server.TLSOptions object.
func InitializeTLS(s *options.KubeletServer) (*server.TLSOptions, error) {
	if s.TLSCertFile == "" && s.TLSPrivateKeyFile == "" {
		s.TLSCertFile = path.Join(s.CertDirectory, "kubelet.crt")
		s.TLSPrivateKeyFile = path.Join(s.CertDirectory, "kubelet.key")
		if crypto.ShouldGenSelfSignedCerts(s.TLSCertFile, s.TLSPrivateKeyFile) {
			if err := crypto.GenerateSelfSignedCert(nodeutil.GetHostname(s.HostnameOverride), s.TLSCertFile, s.TLSPrivateKeyFile, nil, nil); err != nil {
				return nil, fmt.Errorf("unable to generate self signed cert: %v", err)
			}
			glog.V(4).Infof("Using self-signed cert (%s, %s)", s.TLSCertFile, s.TLSPrivateKeyFile)
		}
	}
	tlsOptions := &server.TLSOptions{
		Config: &tls.Config{
			// Can't use SSLv3 because of POODLE and BEAST
			// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
			// Can't use TLSv1.1 because of RC4 cipher usage
			MinVersion: tls.VersionTLS12,
			// Populate PeerCertificates in requests, but don't yet reject connections without certificates.
			ClientAuth: tls.RequestClientCert,
		},
		CertFile: s.TLSCertFile,
		KeyFile:  s.TLSPrivateKeyFile,
	}
	return tlsOptions, nil
}

func authPathClientConfig(s *options.KubeletServer, useDefaults bool) (*restclient.Config, error) {
	authInfo, err := clientauth.LoadFromFile(s.AuthPath.Value())
	// If loading the default auth path, for backwards compatibility keep going
	// with the default auth.
	if err != nil {
		if !useDefaults {
			return nil, err
		}
		glog.Warningf("Could not load kubernetes auth path %s: %v. Continuing with defaults.", s.AuthPath, err)
	}
	if authInfo == nil {
		// authInfo didn't load correctly - continue with defaults.
		authInfo = &clientauth.Info{}
	}
	authConfig, err := authInfo.MergeWithConfig(restclient.Config{})
	if err != nil {
		return nil, err
	}
	authConfig.Host = s.APIServerList[0]
	return &authConfig, nil
}

func kubeconfigClientConfig(s *options.KubeletServer) (*restclient.Config, error) {
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
func createClientConfig(s *options.KubeletServer) (*restclient.Config, error) {
	if s.KubeConfig.Provided() && s.AuthPath.Provided() {
		return nil, fmt.Errorf("cannot specify both --kubeconfig and --auth-path")
	}
	if s.KubeConfig.Provided() {
		return kubeconfigClientConfig(s)
	}
	if s.AuthPath.Provided() {
		return authPathClientConfig(s, false)
	}
	// Try the kubeconfig default first, falling back to the auth path default.
	clientConfig, err := kubeconfigClientConfig(s)
	if err != nil {
		glog.Warningf("Could not load kubeconfig file %s: %v. Trying auth path instead.", s.KubeConfig, err)
		return authPathClientConfig(s, true)
	}
	return clientConfig, nil
}

// CreateAPIServerClientConfig generates a client.Config from command line flags,
// including api-server-list, via createClientConfig and then injects chaos into
// the configuration via addChaosToClientConfig. This func is exported to support
// integration with third party kubelet extensions (e.g. kubernetes-mesos).
func CreateAPIServerClientConfig(s *options.KubeletServer) (*restclient.Config, error) {
	if len(s.APIServerList) < 1 {
		return nil, fmt.Errorf("no api servers specified")
	}
	// TODO: adapt Kube client to support LB over several servers
	if len(s.APIServerList) > 1 {
		glog.Infof("Multiple api servers specified.  Picking first one")
	}

	clientConfig, err := createClientConfig(s)
	if err != nil {
		return nil, err
	}

	clientConfig.ContentType = s.ContentType
	// Override kubeconfig qps/burst settings from flags
	clientConfig.QPS = float32(s.KubeAPIQPS)
	clientConfig.Burst = int(s.KubeAPIBurst)

	addChaosToClientConfig(s, clientConfig)
	return clientConfig, nil
}

// addChaosToClientConfig injects random errors into client connections if configured.
func addChaosToClientConfig(s *options.KubeletServer, config *restclient.Config) {
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
func SimpleKubelet(client *clientset.Clientset,
	dockerClient dockertools.DockerInterface,
	hostname, rootDir, manifestURL, address string,
	port uint,
	readOnlyPort uint,
	masterServiceNamespace string,
	volumePlugins []volume.VolumePlugin,
	tlsOptions *server.TLSOptions,
	cadvisorInterface cadvisor.Interface,
	configFilePath string,
	cloud cloudprovider.Interface,
	osInterface kubecontainer.OSInterface,
	fileCheckFrequency, httpCheckFrequency, minimumGCAge, nodeStatusUpdateFrequency, syncFrequency, outOfDiskTransitionFrequency, evictionPressureTransitionPeriod time.Duration,
	maxPods int, podsPerCore int,
	containerManager cm.ContainerManager, clusterDNS net.IP) *KubeletConfig {
	imageGCPolicy := images.ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	diskSpacePolicy := kubelet.DiskSpacePolicy{
		DockerFreeDiskMB: 256,
		RootFreeDiskMB:   256,
	}
	evictionConfig := eviction.Config{
		PressureTransitionPeriod: evictionPressureTransitionPeriod,
	}

	c := componentconfig.KubeletConfiguration{}
	kcfg := KubeletConfig{
		Address:                      net.ParseIP(address),
		CAdvisorInterface:            cadvisorInterface,
		VolumeStatsAggPeriod:         time.Minute,
		CgroupRoot:                   "",
		Cloud:                        cloud,
		ClusterDNS:                   clusterDNS,
		ConfigFile:                   configFilePath,
		ContainerManager:             containerManager,
		ContainerRuntime:             "docker",
		CPUCFSQuota:                  true,
		DiskSpacePolicy:              diskSpacePolicy,
		DockerClient:                 dockerClient,
		RuntimeCgroups:               "",
		DockerExecHandler:            &dockertools.NativeExecHandler{},
		EnableControllerAttachDetach: false,
		EnableCustomMetrics:          false,
		EnableDebuggingHandlers:      true,
		EnableServer:                 true,
		CgroupsPerQOS:                false,
		FileCheckFrequency:           fileCheckFrequency,
		// Since this kubelet runs with --configure-cbr0=false, it needs to use
		// hairpin-veth to allow hairpin packets. Note that this deviates from
		// what the "real" kubelet currently does, because there's no way to
		// set promiscuous mode on docker0.
		HairpinMode:               componentconfig.HairpinVeth,
		HostnameOverride:          hostname,
		HTTPCheckFrequency:        httpCheckFrequency,
		ImageGCPolicy:             imageGCPolicy,
		KubeClient:                client,
		ManifestURL:               manifestURL,
		MasterServiceNamespace:    masterServiceNamespace,
		MaxContainerCount:         100,
		MaxOpenFiles:              1024,
		MaxPerPodContainerCount:   2,
		MaxPods:                   maxPods,
		NvidiaGPUs:                0,
		MinimumGCAge:              minimumGCAge,
		Mounter:                   mount.New(),
		NodeStatusUpdateFrequency: nodeStatusUpdateFrequency,
		OOMAdjuster:               oom.NewFakeOOMAdjuster(),
		OSInterface:               osInterface,
		PodInfraContainerImage:    c.PodInfraContainerImage,
		Port:                port,
		ReadOnlyPort:        readOnlyPort,
		RegisterNode:        true,
		RegisterSchedulable: true,
		RegistryBurst:       10,
		RegistryPullQPS:     5.0,
		ResolverConfig:      kubetypes.ResolvConfDefault,
		KubeletCgroups:      "/kubelet",
		RootDirectory:       rootDir,
		SerializeImagePulls: true,
		SyncFrequency:       syncFrequency,
		SystemCgroups:       "",
		TLSOptions:          tlsOptions,
		VolumePlugins:       volumePlugins,
		Writer:              &io.StdWriter{},
		OutOfDiskTransitionFrequency: outOfDiskTransitionFrequency,
		EvictionConfig:               evictionConfig,
		PodsPerCore:                  podsPerCore,
		ProtectKernelDefaults:        false,
	}
	return &kcfg
}

// RunKubelet is responsible for setting up and running a kubelet.  It is used in three different applications:
//   1 Integration tests
//   2 Kubelet binary
//   3 Standalone 'kubernetes' binary
// Eventually, #2 will be replaced with instances of #3
func RunKubelet(kcfg *KubeletConfig) error {
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
	if kcfg.EventClient != nil {
		glog.V(4).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: kcfg.EventClient.Events("")})
	} else {
		glog.Warning("No api server defined - no events will be sent to API server.")
	}

	privilegedSources := capabilities.PrivilegedSources{
		HostNetworkSources: kcfg.HostNetworkSources,
		HostPIDSources:     kcfg.HostPIDSources,
		HostIPCSources:     kcfg.HostIPCSources,
	}
	capabilities.Setup(kcfg.AllowPrivileged, privilegedSources, 0)

	credentialprovider.SetPreferredDockercfgPath(kcfg.RootDirectory)
	glog.V(2).Infof("Using root directory: %v", kcfg.RootDirectory)

	builder := kcfg.Builder
	if builder == nil {
		builder = CreateAndInitKubelet
	}
	if kcfg.OSInterface == nil {
		kcfg.OSInterface = kubecontainer.RealOS{}
	}
	k, podCfg, err := builder(kcfg)
	if err != nil {
		return fmt.Errorf("failed to create kubelet: %v", err)
	}

	rlimit.RlimitNumFiles(kcfg.MaxOpenFiles)

	// TODO(dawnchen): remove this once we deprecated old debian containervm images.
	// This is a workaround for issue: https://github.com/opencontainers/runc/issues/726
	// The current chosen number is consistent with most of other os dist.
	const maxkeysPath = "/proc/sys/kernel/keys/root_maxkeys"
	const minKeys uint64 = 1000000
	key, err := ioutil.ReadFile(maxkeysPath)
	if err != nil {
		glog.Errorf("Cannot read keys quota in %s", maxkeysPath)
	} else {
		fields := strings.Fields(string(key))
		nkey, _ := strconv.ParseUint(fields[0], 10, 64)
		if nkey < minKeys {
			glog.Infof("Setting keys quota in %s to %d", maxkeysPath, minKeys)
			err = ioutil.WriteFile(maxkeysPath, []byte(fmt.Sprintf("%d", uint64(minKeys))), 0644)
			if err != nil {
				glog.Warningf("Failed to update %s: %v", maxkeysPath, err)
			}
		}
	}
	const maxbytesPath = "/proc/sys/kernel/keys/root_maxbytes"
	const minBytes uint64 = 25000000
	bytes, err := ioutil.ReadFile(maxbytesPath)
	if err != nil {
		glog.Errorf("Cannot read keys bytes in %s", maxbytesPath)
	} else {
		fields := strings.Fields(string(bytes))
		nbyte, _ := strconv.ParseUint(fields[0], 10, 64)
		if nbyte < minBytes {
			glog.Infof("Setting keys bytes in %s to %d", maxbytesPath, minBytes)
			err = ioutil.WriteFile(maxbytesPath, []byte(fmt.Sprintf("%d", uint64(minBytes))), 0644)
			if err != nil {
				glog.Warningf("Failed to update %s: %v", maxbytesPath, err)
			}
		}
	}

	// process pods and exit.
	if kcfg.Runonce {
		if _, err := k.RunOnce(podCfg.Updates()); err != nil {
			return fmt.Errorf("runonce failed: %v", err)
		}
		glog.Infof("Started kubelet %s as runonce", version.Get().String())
	} else {
		startKubelet(k, podCfg, kcfg)
		glog.Infof("Started kubelet %s", version.Get().String())
	}
	return nil
}

func startKubelet(k KubeletBootstrap, podCfg *config.PodConfig, kc *KubeletConfig) {
	// start the kubelet
	go wait.Until(func() { k.Run(podCfg.Updates()) }, 0, wait.NeverStop)

	// start the kubelet server
	if kc.EnableServer {
		go wait.Until(func() {
			k.ListenAndServe(kc.Address, kc.Port, kc.TLSOptions, kc.Auth, kc.EnableDebuggingHandlers)
		}, 0, wait.NeverStop)
	}
	if kc.ReadOnlyPort > 0 {
		go wait.Until(func() {
			k.ListenAndServeReadOnly(kc.Address, kc.ReadOnlyPort)
		}, 0, wait.NeverStop)
	}
}

func makePodSourceConfig(kc *KubeletConfig) *config.PodConfig {
	// source of all configuration
	cfg := config.NewPodConfig(config.PodConfigNotificationIncremental, kc.Recorder)

	// define file config source
	if kc.ConfigFile != "" {
		glog.Infof("Adding manifest file: %v", kc.ConfigFile)
		config.NewSourceFile(kc.ConfigFile, kc.NodeName, kc.FileCheckFrequency, cfg.Channel(kubetypes.FileSource))
	}

	// define url config source
	if kc.ManifestURL != "" {
		glog.Infof("Adding manifest url %q with HTTP header %v", kc.ManifestURL, kc.ManifestURLHeader)
		config.NewSourceURL(kc.ManifestURL, kc.ManifestURLHeader, kc.NodeName, kc.HTTPCheckFrequency, cfg.Channel(kubetypes.HTTPSource))
	}
	if kc.KubeClient != nil {
		glog.Infof("Watching apiserver")
		config.NewSourceApiserver(kc.KubeClient, kc.NodeName, cfg.Channel(kubetypes.ApiserverSource))
	}
	return cfg
}

// KubeletConfig is all of the parameters necessary for running a kubelet.
// TODO: This should probably be merged with KubeletServer.  The extra object is a consequence of refactoring.
type KubeletConfig struct {
	Address                        net.IP
	AllowPrivileged                bool
	Auth                           server.AuthInterface
	AutoDetectCloudProvider        bool
	Builder                        KubeletBuilder
	CAdvisorInterface              cadvisor.Interface
	VolumeStatsAggPeriod           time.Duration
	CgroupRoot                     string
	Cloud                          cloudprovider.Interface
	ClusterDNS                     net.IP
	ClusterDomain                  string
	ConfigFile                     string
	ConfigureCBR0                  bool
	ContainerManager               cm.ContainerManager
	ContainerRuntime               string
	RuntimeRequestTimeout          time.Duration
	CPUCFSQuota                    bool
	DiskSpacePolicy                kubelet.DiskSpacePolicy
	DockerClient                   dockertools.DockerInterface
	RuntimeCgroups                 string
	DockerExecHandler              dockertools.ExecHandler
	EnableControllerAttachDetach   bool
	EnableCustomMetrics            bool
	EnableDebuggingHandlers        bool
	CgroupsPerQOS                  bool
	EnableServer                   bool
	EventClient                    *clientset.Clientset
	EventBurst                     int
	EventRecordQPS                 float32
	FileCheckFrequency             time.Duration
	Hostname                       string
	HostnameOverride               string
	HostNetworkSources             []string
	HostPIDSources                 []string
	HostIPCSources                 []string
	HTTPCheckFrequency             time.Duration
	ImageGCPolicy                  images.ImageGCPolicy
	KubeClient                     *clientset.Clientset
	ManifestURL                    string
	ManifestURLHeader              http.Header
	MasterServiceNamespace         string
	MaxContainerCount              int
	MaxOpenFiles                   uint64
	MaxPerPodContainerCount        int
	MaxPods                        int
	MinimumGCAge                   time.Duration
	Mounter                        mount.Interface
	NetworkPluginName              string
	NetworkPlugins                 []network.NetworkPlugin
	NodeName                       string
	NodeLabels                     map[string]string
	NodeStatusUpdateFrequency      time.Duration
	NonMasqueradeCIDR              string
	NvidiaGPUs                     int
	OOMAdjuster                    *oom.OOMAdjuster
	OSInterface                    kubecontainer.OSInterface
	PodCIDR                        string
	PodsPerCore                    int
	ReconcileCIDR                  bool
	PodConfig                      *config.PodConfig
	PodInfraContainerImage         string
	Port                           uint
	ReadOnlyPort                   uint
	Recorder                       record.EventRecorder
	RegisterNode                   bool
	RegisterSchedulable            bool
	RegistryBurst                  int
	RegistryPullQPS                float64
	Reservation                    kubetypes.Reservation
	ResolverConfig                 string
	KubeletCgroups                 string
	RktPath                        string
	RktAPIEndpoint                 string
	RktStage1Image                 string
	RootDirectory                  string
	Runonce                        bool
	SeccompProfileRoot             string
	SerializeImagePulls            bool
	StandaloneMode                 bool
	StreamingConnectionIdleTimeout time.Duration
	SyncFrequency                  time.Duration
	SystemCgroups                  string
	TLSOptions                     *server.TLSOptions
	Writer                         io.Writer
	VolumePlugins                  []volume.VolumePlugin
	OutOfDiskTransitionFrequency   time.Duration
	EvictionConfig                 eviction.Config

	ExperimentalFlannelOverlay bool
	NodeIP                     net.IP
	ContainerRuntimeOptions    []kubecontainer.Option
	HairpinMode                string
	BabysitDaemons             bool
	Options                    []kubelet.Option

	ProtectKernelDefaults bool
}

func CreateAndInitKubelet(kc *KubeletConfig) (k KubeletBootstrap, pc *config.PodConfig, err error) {
	// TODO: block until all sources have delivered at least one update to the channel, or break the sync loop
	// up into "per source" synchronizations
	// TODO: KubeletConfig.KubeClient should be a client interface, but client interface misses certain methods
	// used by kubelet. Since NewMainKubelet expects a client interface, we need to make sure we are not passing
	// a nil pointer to it when what we really want is a nil interface.
	var kubeClient clientset.Interface
	if kc.KubeClient != nil {
		kubeClient = kc.KubeClient
		// TODO: remove this when we've refactored kubelet to only use clientset.
	}

	gcPolicy := kubecontainer.ContainerGCPolicy{
		MinAge:             kc.MinimumGCAge,
		MaxPerPodContainer: kc.MaxPerPodContainerCount,
		MaxContainers:      kc.MaxContainerCount,
	}

	daemonEndpoints := &api.NodeDaemonEndpoints{
		KubeletEndpoint: api.DaemonEndpoint{Port: int32(kc.Port)},
	}

	pc = kc.PodConfig
	if pc == nil {
		pc = makePodSourceConfig(kc)
	}
	k, err = kubelet.NewMainKubelet(
		kc.Hostname,
		kc.NodeName,
		kc.DockerClient,
		kubeClient,
		kc.RootDirectory,
		kc.SeccompProfileRoot,
		kc.PodInfraContainerImage,
		kc.SyncFrequency,
		float32(kc.RegistryPullQPS),
		kc.RegistryBurst,
		kc.EventRecordQPS,
		kc.EventBurst,
		gcPolicy,
		pc.SeenAllSources,
		kc.RegisterNode,
		kc.RegisterSchedulable,
		kc.StandaloneMode,
		kc.ClusterDomain,
		kc.ClusterDNS,
		kc.MasterServiceNamespace,
		kc.VolumePlugins,
		kc.NetworkPlugins,
		kc.NetworkPluginName,
		kc.StreamingConnectionIdleTimeout,
		kc.Recorder,
		kc.CAdvisorInterface,
		kc.ImageGCPolicy,
		kc.DiskSpacePolicy,
		kc.Cloud,
		kc.AutoDetectCloudProvider,
		kc.NodeLabels,
		kc.NodeStatusUpdateFrequency,
		kc.OSInterface,
		kc.CgroupsPerQOS,
		kc.CgroupRoot,
		kc.ContainerRuntime,
		kc.RuntimeRequestTimeout,
		kc.RktPath,
		kc.RktAPIEndpoint,
		kc.RktStage1Image,
		kc.Mounter,
		kc.Writer,
		kc.ConfigureCBR0,
		kc.NonMasqueradeCIDR,
		kc.PodCIDR,
		kc.ReconcileCIDR,
		kc.MaxPods,
		kc.PodsPerCore,
		kc.NvidiaGPUs,
		kc.DockerExecHandler,
		kc.ResolverConfig,
		kc.CPUCFSQuota,
		daemonEndpoints,
		kc.OOMAdjuster,
		kc.SerializeImagePulls,
		kc.ContainerManager,
		kc.OutOfDiskTransitionFrequency,
		kc.ExperimentalFlannelOverlay,
		kc.NodeIP,
		kc.Reservation,
		kc.EnableCustomMetrics,
		kc.VolumeStatsAggPeriod,
		kc.ContainerRuntimeOptions,
		kc.HairpinMode,
		kc.BabysitDaemons,
		kc.EvictionConfig,
		kc.Options,
		kc.EnableControllerAttachDetach,
	)

	if err != nil {
		return nil, nil, err
	}

	k.BirthCry()

	k.StartGarbageCollection()

	return k, pc, nil
}

func parseReservation(kubeReserved, systemReserved utilconfig.ConfigurationMap) (*kubetypes.Reservation, error) {
	reservation := new(kubetypes.Reservation)
	rl, err := parseResourceList(kubeReserved)
	if err != nil {
		return nil, err
	}
	reservation.Kubernetes = rl

	rl, err = parseResourceList(systemReserved)
	if err != nil {
		return nil, err
	}
	reservation.System = rl

	return reservation, nil
}

func parseResourceList(m utilconfig.ConfigurationMap) (api.ResourceList, error) {
	rl := make(api.ResourceList)
	for k, v := range m {
		switch api.ResourceName(k) {
		// Only CPU and memory resources are supported.
		case api.ResourceCPU, api.ResourceMemory:
			q, err := resource.ParseQuantity(v)
			if err != nil {
				return nil, err
			}
			if q.Sign() == -1 {
				return nil, fmt.Errorf("resource quantity for %q cannot be negative: %v", k, v)
			}
			rl[api.ResourceName(k)] = q
		default:
			return nil, fmt.Errorf("cannot reserve %q resource", k)
		}
	}
	return rl, nil
}
