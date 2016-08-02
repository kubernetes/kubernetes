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
	"github.com/syndtr/gocapability/capability"

	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/api"
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
	"k8s.io/kubernetes/pkg/kubelet/server"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/util/crypto"
	"k8s.io/kubernetes/pkg/util/flock"
	kubeio "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/rlimit"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/volume"
)

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
described in those PodSpecs are running and healthy.

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

// UnsecuredKubeletDeps returns a KubeletDeps suitable for being run, or an error if the server setup
// is not valid.  It will not start any background processes, and does not include authentication/authorization
func UnsecuredKubeletDeps(s *options.KubeletServer) (*kubelet.KubeletDeps, error) {

	mounter := mount.New()
	var writer kubeio.Writer = &kubeio.StdWriter{}
	if s.Containerized {
		glog.V(2).Info("Running kubelet in containerized mode (experimental)")
		mounter = mount.NewNsenterMounter()
		writer = &kubeio.NsenterWriter{}
	}

	return &kubelet.KubeletDeps{
		Auth:              nil, // default does not enforce auth[nz]
		CAdvisorInterface: nil, // launches background processes, not set here
		Cloud:             nil, // cloud provider might start background processes
		ContainerManager:  nil,
		DockerClient:      dockertools.ConnectToDockerOrDie(s.DockerEndpoint, s.RuntimeRequestTimeout.Duration), // TODO(random-liu): Set RuntimeRequestTimeout for rkt.
		KubeClient:        nil,
		Mounter:           mounter,
		NetworkPlugins:    ProbeNetworkPlugins(s.NetworkPluginDir),
		OOMAdjuster:       oom.NewOOMAdjuster(),
		OSInterface:       kubecontainer.RealOS{},
		Writer:            writer,
		VolumePlugins:     ProbeVolumePlugins(s.VolumePluginDir),
	}, nil
}

// Run runs the specified KubeletServer with the given KubeletDeps.  This should never exit.
// The kubeDeps argument may be nil - if so, it is initialized from the settings on KubeletServer.
// Otherwise, the caller is assumed to have set up the KubeletDeps object and a default one will
// not be generated.
func Run(s *options.KubeletServer, kubeDeps *kubelet.KubeletDeps) error {
	err := run(s, kubeDeps)
	if err != nil {
		glog.Errorf("Failed running kubelet: %v", err)
	}
	return err
}

func run(s *options.KubeletServer, kubeDeps *kubelet.KubeletDeps) (err error) {
	if s.ExitOnLockContention && s.LockFilePath == "" {
		return errors.New("cannot exit on lock file contention: no lock file specified")
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

	// check if we have CAP_SYS_ADMIN to setgroup properly
	pid, err := capability.NewPid(os.Getpid())
	if err != nil {
		return err
	}
	if !pid.Get(capability.EFFECTIVE, capability.CAP_SYS_ADMIN) {
		return fmt.Errorf("Kubelet needs the CAP_SYS_ADMIN capability. Please run kubelet as root or in a privileged container")
	}

	if kubeDeps == nil {
		deps, err := UnsecuredKubeletDeps(s)
		if err != nil {
			return err
		}
		kubeDeps = deps

		clientConfig, err := CreateAPIServerClientConfig(s)
		if err == nil {
			kubeDeps.KubeClient, err = clientset.NewForConfig(clientConfig)

			// make a separate client for events
			eventClientConfig := *clientConfig
			eventClientConfig.QPS = float32(s.EventRecordQPS)
			eventClientConfig.Burst = int(s.EventBurst)
			kubeDeps.EventClient, err = clientset.NewForConfig(&eventClientConfig)
		}
		if err != nil && len(s.APIServerList) > 0 {
			glog.Warningf("No API client: %v", err)
		}

		if s.CloudProvider != kubeExternal.AutoDetectCloudProvider {
			cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
			if err != nil {
				return err
			}
			if cloud == nil {
				glog.V(2).Infof("No cloud provider specified: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)
			} else {
				glog.V(2).Infof("Successfully initialized cloud provider: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)
				kubeDeps.Cloud = cloud
			}
		}
	}

	if kubeDeps.CAdvisorInterface == nil {
		kubeDeps.CAdvisorInterface, err = cadvisor.New(uint(s.CAdvisorPort), s.ContainerRuntime)
		if err != nil {
			return err
		}
	}

	if kubeDeps.ContainerManager == nil {
		if s.SystemCgroups != "" && s.CgroupRoot == "" {
			return fmt.Errorf("invalid configuration: system container was specified and cgroup root was not specified")
		}
		kubeDeps.ContainerManager, err = cm.NewContainerManager(kubeDeps.Mounter, kubeDeps.CAdvisorInterface, cm.NodeConfig{
			RuntimeCgroupsName: s.RuntimeCgroups,
			SystemCgroupsName:  s.SystemCgroups,
			KubeletCgroupsName: s.KubeletCgroups,
			ContainerRuntime:   s.ContainerRuntime,
			CgroupsPerQOS:      s.CgroupsPerQOS,
			CgroupRoot:         s.CgroupRoot,
		})
		if err != nil {
			return err
		}
	}

	runtime.ReallyCrash = s.ReallyCrashForTesting
	rand.Seed(time.Now().UTC().UnixNano())

	// TODO(vmarmol): Do this through container config.
	oomAdjuster := kubeDeps.OOMAdjuster
	if err := oomAdjuster.ApplyOOMScoreAdj(0, int(s.OOMScoreAdj)); err != nil {
		glog.Warning(err)
	}

	if err := RunKubelet(&s.KubeletConfiguration, kubeDeps); err != nil {
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
func InitializeTLS(kc *componentconfig.KubeletConfiguration) (*server.TLSOptions, error) {
	if kc.TLSCertFile == "" && kc.TLSPrivateKeyFile == "" {
		kc.TLSCertFile = path.Join(kc.CertDirectory, "kubelet.crt")
		kc.TLSPrivateKeyFile = path.Join(kc.CertDirectory, "kubelet.key")
		if crypto.ShouldGenSelfSignedCerts(kc.TLSCertFile, kc.TLSPrivateKeyFile) {
			if err := crypto.GenerateSelfSignedCert(nodeutil.GetHostname(kc.HostnameOverride), kc.TLSCertFile, kc.TLSPrivateKeyFile, nil, nil); err != nil {
				return nil, fmt.Errorf("unable to generate self signed cert: %v", err)
			}
			glog.V(4).Infof("Using self-signed cert (%s, %s)", kc.TLSCertFile, kc.TLSPrivateKeyFile)
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
		CertFile: kc.TLSCertFile,
		KeyFile:  kc.TLSPrivateKeyFile,
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
	containerManager cm.ContainerManager, clusterDNS net.IP) (*componentconfig.KubeletConfiguration, *kubelet.KubeletDeps) {
	// imageGCPolicy := images.ImageGCPolicy{
	// 	HighThresholdPercent: 90,
	// 	LowThresholdPercent:  80,
	// }
	// diskSpacePolicy := kubelet.DiskSpacePolicy{
	// 	DockerFreeDiskMB: 256,
	// 	RootFreeDiskMB:   256,
	// }
	// evictionConfig := eviction.Config{
	// 	PressureTransitionPeriod: evictionPressureTransitionPeriod,
	// }

	// TODO(mtaufen): Transfer the default values from kcfg into
	//                this KubeletConfiguration e.g. CgroupRoot = ""
	kubeCfg := componentconfig.KubeletConfiguration{}
	kubeDeps := kubelet.KubeletDeps{
		//		Address:                      net.ParseIP(address),
		CAdvisorInterface: cadvisorInterface,
		// VolumeStatsAggPeriod:         time.Minute,
		// CgroupRoot:                   "",
		Cloud: cloud,
		// ClusterDNS:                   clusterDNS,
		// ConfigFile:                   configFilePath,
		ContainerManager: containerManager,
		// ContainerRuntime:             "docker",
		// CPUCFSQuota:                  true,
		// DiskSpacePolicy: diskSpacePolicy,
		DockerClient: dockerClient,
		// RuntimeCgroups:               "",
		// DockerExecHandler: &dockertools.NativeExecHandler{},
		// EnableControllerAttachDetach: false,
		// EnableCustomMetrics:          false,
		// EnableDebuggingHandlers:      true,
		// EnableServer:                 true,
		// CgroupsPerQOS:                false,
		// FileCheckFrequency:           fileCheckFrequency,
		// Since this kubelet runs with --configure-cbr0=false, it needs to use
		// hairpin-veth to allow hairpin packets. Note that this deviates from
		// what the "real" kubelet currently does, because there's no way to
		// set promiscuous mode on docker0.
		// HairpinMode:               componentconfig.HairpinVeth,
		// HostnameOverride:          hostname,
		// HTTPCheckFrequency:        httpCheckFrequency,
		// ImageGCPolicy: imageGCPolicy,
		KubeClient: client,
		// ManifestURL:               manifestURL,
		// MasterServiceNamespace:    masterServiceNamespace,
		// MaxContainerCount:       100,
		// MaxOpenFiles: 1024,
		// MaxPerPodContainerCount: 2,
		// MaxPods:                 maxPods,
		// NvidiaGPUs:              0,
		// MinimumGCAge:              minimumGCAge,
		// Mounter:                   mount.New(),
		// NodeStatusUpdateFrequency: nodeStatusUpdateFrequency,
		OOMAdjuster: oom.NewFakeOOMAdjuster(),
		OSInterface: osInterface,
		// PodInfraContainerImage:    c.PodInfraContainerImage,
		// Port:         port,
		// ReadOnlyPort: readOnlyPort,
		// RegisterNode:        true,
		// RegisterSchedulable: true,
		// RegistryBurst:   10,
		// RegistryPullQPS: 5.0,
		// ResolverConfig:      kubetypes.ResolvConfDefault,
		// KubeletCgroups:      "/kubelet",
		// RootDirectory:       rootDir,
		// SerializeImagePulls: true,
		// SyncFrequency:       syncFrequency,
		// SystemCgroups:       "",
		// TLSOptions:    tlsOptions,
		VolumePlugins: volumePlugins,
		Writer:        &kubeio.StdWriter{},
		// OutOfDiskTransitionFrequency: outOfDiskTransitionFrequency,
		// EvictionConfig: evictionConfig,
		// PodsPerCore:    podsPerCore,
	}
	return &kubeCfg, &kubeDeps
}

// RunKubelet is responsible for setting up and running a kubelet.  It is used in three different applications:
//   1 Integration tests
//   2 Kubelet binary
//   3 Standalone 'kubernetes' binary
// Eventually, #2 will be replaced with instances of #3
func RunKubelet(kubeCfg *componentconfig.KubeletConfiguration, kubeDeps *kubelet.KubeletDeps) error {
	hostname := nodeutil.GetHostname(kubeCfg.HostnameOverride)
	// Query the cloud provider for our node name, default to hostname if kcfg.Cloud == nil
	// TODO(mtaufen): Was there actually any reason to check len(kcfg.NodeName) == 0?
	//            --> Was anyone passing a kcfg.NodeName simultaneously with a non-nil kcfg.Cloud?
	nodeName := hostname
	if kubeDeps.Cloud != nil {
		var err error
		instances, ok := kubeDeps.Cloud.Instances()
		if !ok {
			return fmt.Errorf("failed to get instances from cloud provider")
		}

		nodeName, err = instances.CurrentNodeName(hostname)
		if err != nil {
			return fmt.Errorf("error fetching current instance name from cloud provider: %v", err)
		}

		glog.V(2).Infof("cloud provider determined current node name to be %s", nodeName)
	}

	eventBroadcaster := record.NewBroadcaster()
	kubeDeps.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: "kubelet", Host: nodeName})
	eventBroadcaster.StartLogging(glog.V(3).Infof)
	if kubeDeps.EventClient != nil {
		glog.V(4).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: kubeDeps.EventClient.Events("")})
	} else {
		glog.Warning("No api server defined - no events will be sent to API server.")
	}

	// TODO(mtaufen): I moved the validation of these fields here, from UnsecuredKubeletConfig,
	//                so that I could remove the associated fields from KubeletConfig. I would
	//                prefer this to be done as part of an independent validation step on the
	//                KubeletConfiguration. But as far as I can tell, we don't have an explicit
	//                place for validation of the KubeletConfiguration yet.
	hostNetworkSources, err := kubetypes.GetValidatedSources(kubeCfg.HostNetworkSources)
	if err != nil {
		return err
	}

	hostPIDSources, err := kubetypes.GetValidatedSources(kubeCfg.HostPIDSources)
	if err != nil {
		return err
	}

	hostIPCSources, err := kubetypes.GetValidatedSources(kubeCfg.HostIPCSources)
	if err != nil {
		return err
	}

	privilegedSources := capabilities.PrivilegedSources{
		HostNetworkSources: hostNetworkSources,
		HostPIDSources:     hostPIDSources,
		HostIPCSources:     hostIPCSources,
	}
	capabilities.Setup(kubeCfg.AllowPrivileged, privilegedSources, 0)

	credentialprovider.SetPreferredDockercfgPath(kubeCfg.RootDirectory)
	glog.V(2).Infof("Using root directory: %v", kubeCfg.RootDirectory)

	builder := kubeDeps.Builder
	if builder == nil {
		builder = CreateAndInitKubelet
	}
	if kubeDeps.OSInterface == nil {
		kubeDeps.OSInterface = kubecontainer.RealOS{}
	}
	k, err := builder(kubeCfg, kubeDeps)
	if err != nil {
		return fmt.Errorf("failed to create kubelet: %v", err)
	}
	podCfg := k.GetDeps().PodConfig

	rlimit.RlimitNumFiles(uint64(kubeCfg.MaxOpenFiles))

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
	if kubeCfg.RunOnce {
		if _, err := k.RunOnce(podCfg.Updates()); err != nil {
			return fmt.Errorf("runonce failed: %v", err)
		}
		glog.Infof("Started kubelet %s as runonce", version.Get().String())
	} else {
		err := startKubelet(k, podCfg, kubeCfg, kubeDeps)
		if err != nil {
			return err
		}
		glog.Infof("Started kubelet %s", version.Get().String())
	}
	return nil
}

// TODO(mtaufen): I think we can actually get podCfg out of kubeDeps now. Remember to look into this.
func startKubelet(k kubelet.KubeletBootstrap, podCfg *config.PodConfig, kubeCfg *componentconfig.KubeletConfiguration, kubeDeps *kubelet.KubeletDeps) error {

	// Initialize the TLS Options
	tlsOptions, err := InitializeTLS(kubeCfg)
	if err != nil {
		return err
	}

	// start the kubelet
	go wait.Until(func() { k.Run(podCfg.Updates()) }, 0, wait.NeverStop)

	// start the kubelet server
	if kubeCfg.EnableServer {
		go wait.Until(func() {
			k.ListenAndServe(net.ParseIP(kubeCfg.Address), uint(kubeCfg.Port), tlsOptions, kubeDeps.Auth, kubeCfg.EnableDebuggingHandlers)
		}, 0, wait.NeverStop)
	}
	if kubeCfg.ReadOnlyPort > 0 {
		go wait.Until(func() {
			k.ListenAndServeReadOnly(net.ParseIP(kubeCfg.Address), uint(kubeCfg.ReadOnlyPort))
		}, 0, wait.NeverStop)
	}

	return nil
}

func CreateAndInitKubelet(kubeCfg *componentconfig.KubeletConfiguration, kubeDeps *kubelet.KubeletDeps) (k kubelet.KubeletBootstrap, err error) {
	// TODO: block until all sources have delivered at least one update to the channel, or break the sync loop
	// up into "per source" synchronizations

	k, err = kubelet.NewMainKubelet(kubeCfg, kubeDeps)
	if err != nil {
		return nil, err
	}

	k.BirthCry()

	k.StartGarbageCollection()

	return k, nil
}
