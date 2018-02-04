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
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	_ "net/http/pprof"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/flag"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/certificate"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/client/chaosclient"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	kubeletconfigv1beta1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kubeletcertificate "k8s.io/kubernetes/pkg/kubelet/certificate"
	"k8s.io/kubernetes/pkg/kubelet/certificate/bootstrap"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	dockerremote "k8s.io/kubernetes/pkg/kubelet/dockershim/remote"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	dynamickubeletconfig "k8s.io/kubernetes/pkg/kubelet/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/configfiles"
	"k8s.io/kubernetes/pkg/kubelet/server"
	"k8s.io/kubernetes/pkg/kubelet/server/streaming"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/configz"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
	"k8s.io/kubernetes/pkg/util/flock"
	kubeio "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/rlimit"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/version/verflag"
)

const (
	// Kubelet component name
	componentKubelet = "kubelet"
)

// NewKubeletCommand creates a *cobra.Command object with default parameters
func NewKubeletCommand() *cobra.Command {
	cleanFlagSet := pflag.NewFlagSet(componentKubelet, pflag.ContinueOnError)
	kubeletFlags := options.NewKubeletFlags()
	kubeletConfig, err := options.NewKubeletConfiguration()
	// programmer error
	if err != nil {
		glog.Fatal(err)
	}

	cmd := &cobra.Command{
		Use: componentKubelet,
		Long: `The kubelet is the primary "node agent" that runs on each
node. The kubelet works in terms of a PodSpec. A PodSpec is a YAML or JSON object
that describes a pod. The kubelet takes a set of PodSpecs that are provided through
various mechanisms (primarily through the apiserver) and ensures that the containers
described in those PodSpecs are running and healthy. The kubelet doesn't manage
containers which were not created by Kubernetes.

Other than from an PodSpec from the apiserver, there are three ways that a container
manifest can be provided to the Kubelet.

File: Path passed as a flag on the command line. Files under this path will be monitored
periodically for updates. The monitoring period is 20s by default and is configurable
via a flag.

HTTP endpoint: HTTP endpoint passed as a parameter on the command line. This endpoint
is checked every 20 seconds (also configurable with a flag).

HTTP server: The kubelet can also listen for HTTP and respond to a simple API
(underspec'd currently) to submit a new manifest.`,
		// The Kubelet has special flag parsing requirements to enforce flag precedence rules,
		// so we do all our parsing manually in Run, below.
		// DisableFlagParsing=true provides the full set of flags passed to the kubelet in the
		// `args` arg to Run, without Cobra's interference.
		DisableFlagParsing: true,
		Run: func(cmd *cobra.Command, args []string) {
			// initial flag parse, since we disable cobra's flag parsing
			if err := cleanFlagSet.Parse(args); err != nil {
				cmd.Usage()
				glog.Fatal(err)
			}

			// short-circuit on help
			help, err := cleanFlagSet.GetBool("help")
			if err != nil {
				glog.Fatal(`"help" flag is non-bool, programmer error, please correct`)
			}
			if help {
				cmd.Help()
				return
			}

			// short-circuit on verflag
			verflag.PrintAndExitIfRequested()
			utilflag.PrintFlags(cleanFlagSet)

			// set feature gates from initial flags-based config
			if err := utilfeature.DefaultFeatureGate.SetFromMap(kubeletConfig.FeatureGates); err != nil {
				glog.Fatal(err)
			}

			// validate the initial KubeletFlags
			if err := options.ValidateKubeletFlags(kubeletFlags); err != nil {
				glog.Fatal(err)
			}

			// load kubelet config file, if provided
			if configFile := kubeletFlags.KubeletConfigFile; len(configFile) > 0 {
				kubeletConfig, err = loadConfigFile(configFile)
				if err != nil {
					glog.Fatal(err)
				}
				// We must enforce flag precedence by re-parsing the command line into the new object.
				// This is necessary to preserve backwards-compatibility across binary upgrades.
				// See issue #56171 for more details.
				if err := kubeletConfigFlagPrecedence(kubeletConfig, args); err != nil {
					glog.Fatal(err)
				}
				// update feature gates based on new config
				if err := utilfeature.DefaultFeatureGate.SetFromMap(kubeletConfig.FeatureGates); err != nil {
					glog.Fatal(err)
				}
			}

			// use dynamic kubelet config, if enabled
			var kubeletConfigController *dynamickubeletconfig.Controller
			if dynamicConfigDir := kubeletFlags.DynamicConfigDir.Value(); len(dynamicConfigDir) > 0 {
				kubeletConfig, kubeletConfigController, err = BootstrapKubeletConfigController(kubeletConfig, dynamicConfigDir)
				if err != nil {
					glog.Fatal(err)
				}
				// We must enforce flag precedence by re-parsing the command line into the new object.
				// This is necessary to preserve backwards-compatibility across binary upgrades.
				// See issue #56171 for more details.
				if err := kubeletConfigFlagPrecedence(kubeletConfig, args); err != nil {
					glog.Fatal(err)
				}
				// update feature gates based on new config
				if err := utilfeature.DefaultFeatureGate.SetFromMap(kubeletConfig.FeatureGates); err != nil {
					glog.Fatal(err)
				}
			}

			// construct a KubeletServer from kubeletFlags and kubeletConfig
			kubeletServer := &options.KubeletServer{
				KubeletFlags:         *kubeletFlags,
				KubeletConfiguration: *kubeletConfig,
			}

			// use kubeletServer to construct the default KubeletDeps
			kubeletDeps, err := UnsecuredDependencies(kubeletServer)
			if err != nil {
				glog.Fatal(err)
			}

			// add the kubelet config controller to kubeletDeps
			kubeletDeps.KubeletConfigController = kubeletConfigController

			// start the experimental docker shim, if enabled
			if kubeletServer.KubeletFlags.ExperimentalDockershim {
				if err := RunDockershim(&kubeletServer.KubeletFlags, kubeletConfig); err != nil {
					glog.Fatal(err)
				}
			}

			// run the kubelet
			if err := Run(kubeletServer, kubeletDeps); err != nil {
				glog.Fatal(err)
			}
		},
	}

	// keep cleanFlagSet separate, so Cobra doesn't pollute it with the global flags
	kubeletFlags.AddFlags(cleanFlagSet)
	options.AddKubeletConfigFlags(cleanFlagSet, kubeletConfig)
	options.AddGlobalFlags(cleanFlagSet)
	cleanFlagSet.BoolP("help", "h", false, fmt.Sprintf("help for %s", cmd.Name()))

	// ugly, but necessary, because Cobra's default UsageFunc and HelpFunc pollute the flagset with global flags
	const usageFmt = "Usage:\n  %s\n\nFlags:\n%s"
	cmd.SetUsageFunc(func(cmd *cobra.Command) error {
		fmt.Fprintf(cmd.OutOrStderr(), usageFmt, cmd.UseLine(), cleanFlagSet.FlagUsagesWrapped(2))
		return nil
	})
	cmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
		fmt.Fprintf(cmd.OutOrStdout(), "%s\n\n"+usageFmt, cmd.Long, cmd.UseLine(), cleanFlagSet.FlagUsagesWrapped(2))
	})

	return cmd
}

// newFlagSetWithGlobals constructs a new pflag.FlagSet with global flags registered
// on it.
func newFlagSetWithGlobals() *pflag.FlagSet {
	fs := pflag.NewFlagSet("", pflag.ExitOnError)
	// set the normalize func, similar to k8s.io/apiserver/pkg/util/flag/flags.go:InitFlags
	fs.SetNormalizeFunc(flag.WordSepNormalizeFunc)
	// explicitly add flags from libs that register global flags
	options.AddGlobalFlags(fs)
	return fs
}

// newFakeFlagSet constructs a pflag.FlagSet with the same flags as fs, but where
// all values have noop Set implementations
func newFakeFlagSet(fs *pflag.FlagSet) *pflag.FlagSet {
	ret := pflag.NewFlagSet("", pflag.ExitOnError)
	ret.SetNormalizeFunc(fs.GetNormalizeFunc())
	fs.VisitAll(func(f *pflag.Flag) {
		ret.VarP(flag.NoOp{}, f.Name, f.Shorthand, f.Usage)
	})
	return ret
}

// kubeletConfigFlagPrecedence re-parses flags over the KubeletConfiguration object.
// We must enforce flag precedence by re-parsing the command line into the new object.
// This is necessary to preserve backwards-compatibility across binary upgrades.
// See issue #56171 for more details.
func kubeletConfigFlagPrecedence(kc *kubeletconfiginternal.KubeletConfiguration, args []string) error {
	// We use a throwaway kubeletFlags and a fake global flagset to avoid double-parses,
	// as some Set implementations accumulate values from multiple flag invocations.
	fs := newFakeFlagSet(newFlagSetWithGlobals())
	// register throwaway KubeletFlags
	options.NewKubeletFlags().AddFlags(fs)
	// register new KubeletConfiguration
	options.AddKubeletConfigFlags(fs, kc)
	// Remember original feature gates, so we can merge with flag gates later
	original := kc.FeatureGates
	// re-parse flags
	if err := fs.Parse(args); err != nil {
		return err
	}
	// Add back feature gates that were set in the original kc, but not in flags
	for k, v := range original {
		if _, ok := kc.FeatureGates[k]; !ok {
			kc.FeatureGates[k] = v
		}
	}
	return nil
}

func loadConfigFile(name string) (*kubeletconfiginternal.KubeletConfiguration, error) {
	const errFmt = "failed to load Kubelet config file %s, error %v"
	// compute absolute path based on current working dir
	kubeletConfigFile, err := filepath.Abs(name)
	if err != nil {
		return nil, fmt.Errorf(errFmt, name, err)
	}
	loader, err := configfiles.NewFsLoader(utilfs.DefaultFs{}, kubeletConfigFile)
	if err != nil {
		return nil, fmt.Errorf(errFmt, name, err)
	}
	kc, err := loader.Load()
	if err != nil {
		return nil, fmt.Errorf(errFmt, name, err)
	}
	return kc, err
}

// UnsecuredDependencies returns a Dependencies suitable for being run, or an error if the server setup
// is not valid.  It will not start any background processes, and does not include authentication/authorization
func UnsecuredDependencies(s *options.KubeletServer) (*kubelet.Dependencies, error) {
	// Initialize the TLS Options
	tlsOptions, err := InitializeTLS(&s.KubeletFlags, &s.KubeletConfiguration)
	if err != nil {
		return nil, err
	}

	mounter := mount.New(s.ExperimentalMounterPath)
	var writer kubeio.Writer = &kubeio.StdWriter{}
	if s.Containerized {
		glog.V(2).Info("Running kubelet in containerized mode")
		mounter = mount.NewNsenterMounter()
		writer = &kubeio.NsenterWriter{}
	}

	var dockerClientConfig *dockershim.ClientConfig
	if s.ContainerRuntime == kubetypes.DockerContainerRuntime {
		dockerClientConfig = &dockershim.ClientConfig{
			DockerEndpoint:            s.DockerEndpoint,
			RuntimeRequestTimeout:     s.RuntimeRequestTimeout.Duration,
			ImagePullProgressDeadline: s.ImagePullProgressDeadline.Duration,
		}
	}

	return &kubelet.Dependencies{
		Auth:                nil, // default does not enforce auth[nz]
		CAdvisorInterface:   nil, // cadvisor.New launches background processes (bg http.ListenAndServe, and some bg cleaners), not set here
		Cloud:               nil, // cloud provider might start background processes
		ContainerManager:    nil,
		DockerClientConfig:  dockerClientConfig,
		KubeClient:          nil,
		HeartbeatClient:     nil,
		ExternalKubeClient:  nil,
		EventClient:         nil,
		Mounter:             mounter,
		NetworkPlugins:      ProbeNetworkPlugins(s.CNIConfDir, s.CNIBinDir),
		OOMAdjuster:         oom.NewOOMAdjuster(),
		OSInterface:         kubecontainer.RealOS{},
		Writer:              writer,
		VolumePlugins:       ProbeVolumePlugins(),
		DynamicPluginProber: GetDynamicPluginProber(s.VolumePluginDir),
		TLSOptions:          tlsOptions}, nil
}

// Run runs the specified KubeletServer with the given Dependencies. This should never exit.
// The kubeDeps argument may be nil - if so, it is initialized from the settings on KubeletServer.
// Otherwise, the caller is assumed to have set up the Dependencies object and a default one will
// not be generated.
func Run(s *options.KubeletServer, kubeDeps *kubelet.Dependencies) error {
	// To help debugging, immediately log version
	glog.Infof("Version: %+v", version.Get())
	if err := run(s, kubeDeps); err != nil {
		return fmt.Errorf("failed to run Kubelet: %v", err)
	}
	return nil
}

func checkPermissions() error {
	if uid := os.Getuid(); uid != 0 {
		return fmt.Errorf("Kubelet needs to run as uid `0`. It is being run as %d", uid)
	}
	// TODO: Check if kubelet is running in the `initial` user namespace.
	// http://man7.org/linux/man-pages/man7/user_namespaces.7.html
	return nil
}

func setConfigz(cz *configz.Config, kc *kubeletconfiginternal.KubeletConfiguration) error {
	scheme, _, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		return err
	}
	versioned := kubeletconfigv1beta1.KubeletConfiguration{}
	if err := scheme.Convert(kc, &versioned, nil); err != nil {
		return err
	}
	cz.Set(versioned)
	return nil
}

func initConfigz(kc *kubeletconfiginternal.KubeletConfiguration) error {
	cz, err := configz.New("kubeletconfig")
	if err != nil {
		glog.Errorf("unable to register configz: %s", err)
		return err
	}
	if err := setConfigz(cz, kc); err != nil {
		glog.Errorf("unable to register config: %s", err)
		return err
	}
	return nil
}

// makeEventRecorder sets up kubeDeps.Recorder if it's nil. It's a no-op otherwise.
func makeEventRecorder(kubeDeps *kubelet.Dependencies, nodeName types.NodeName) {
	if kubeDeps.Recorder != nil {
		return
	}
	eventBroadcaster := record.NewBroadcaster()
	kubeDeps.Recorder = eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: componentKubelet, Host: string(nodeName)})
	eventBroadcaster.StartLogging(glog.V(3).Infof)
	if kubeDeps.EventClient != nil {
		glog.V(4).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeDeps.EventClient.Events("")})
	} else {
		glog.Warning("No api server defined - no events will be sent to API server.")
	}
}

func run(s *options.KubeletServer, kubeDeps *kubelet.Dependencies) (err error) {
	// Set global feature gates based on the value on the initial KubeletServer
	err = utilfeature.DefaultFeatureGate.SetFromMap(s.KubeletConfiguration.FeatureGates)
	if err != nil {
		return err
	}
	// validate the initial KubeletServer (we set feature gates first, because this validation depends on feature gates)
	if err := options.ValidateKubeletServer(s); err != nil {
		return err
	}

	// Obtain Kubelet Lock File
	if s.ExitOnLockContention && s.LockFilePath == "" {
		return errors.New("cannot exit on lock file contention: no lock file specified")
	}
	done := make(chan struct{})
	if s.LockFilePath != "" {
		glog.Infof("acquiring file lock on %q", s.LockFilePath)
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

	// Register current configuration with /configz endpoint
	err = initConfigz(&s.KubeletConfiguration)
	if err != nil {
		glog.Errorf("unable to register KubeletConfiguration with configz, error: %v", err)
	}

	// About to get clients and such, detect standaloneMode
	standaloneMode := true
	if len(s.KubeConfig) > 0 {
		standaloneMode = false
	}

	if kubeDeps == nil {
		kubeDeps, err = UnsecuredDependencies(s)
		if err != nil {
			return err
		}
	}

	if kubeDeps.Cloud == nil {
		if !cloudprovider.IsExternal(s.CloudProvider) {
			cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
			if err != nil {
				return err
			}
			if cloud == nil {
				glog.V(2).Infof("No cloud provider specified: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)
			} else {
				glog.V(2).Infof("Successfully initialized cloud provider: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)
			}
			kubeDeps.Cloud = cloud
		}
	}

	nodeName, err := getNodeName(kubeDeps.Cloud, nodeutil.GetHostname(s.HostnameOverride))
	if err != nil {
		return err
	}

	if s.BootstrapKubeconfig != "" {
		if err := bootstrap.LoadClientCert(s.KubeConfig, s.BootstrapKubeconfig, s.CertDirectory, nodeName); err != nil {
			return err
		}
	}

	// if in standalone mode, indicate as much by setting all clients to nil
	if standaloneMode {
		kubeDeps.KubeClient = nil
		kubeDeps.ExternalKubeClient = nil
		kubeDeps.EventClient = nil
		kubeDeps.HeartbeatClient = nil
		glog.Warningf("standalone mode, no API client")
	} else if kubeDeps.KubeClient == nil || kubeDeps.ExternalKubeClient == nil || kubeDeps.EventClient == nil || kubeDeps.HeartbeatClient == nil {
		// initialize clients if not standalone mode and any of the clients are not provided
		var kubeClient clientset.Interface
		var eventClient v1core.EventsGetter
		var heartbeatClient v1core.CoreV1Interface
		var externalKubeClient clientset.Interface

		clientConfig, err := createAPIServerClientConfig(s)
		if err != nil {
			return fmt.Errorf("invalid kubeconfig: %v", err)
		}

		var clientCertificateManager certificate.Manager
		if s.RotateCertificates && utilfeature.DefaultFeatureGate.Enabled(features.RotateKubeletClientCertificate) {
			clientCertificateManager, err = kubeletcertificate.NewKubeletClientCertificateManager(s.CertDirectory, nodeName, clientConfig.CertData, clientConfig.KeyData, clientConfig.CertFile, clientConfig.KeyFile)
			if err != nil {
				return err
			}

			// we set exitAfter to five minutes because we use this client configuration to request new certs - if we are unable
			// to request new certs, we will be unable to continue normal operation. Exiting the process allows a wrapper
			// or the bootstrapping credentials to potentially lay down new initial config.
			if err := kubeletcertificate.UpdateTransport(wait.NeverStop, clientConfig, clientCertificateManager, 5*time.Minute); err != nil {
				return err
			}
		}

		kubeClient, err = clientset.NewForConfig(clientConfig)
		if err != nil {
			glog.Warningf("New kubeClient from clientConfig error: %v", err)
		} else if kubeClient.CertificatesV1beta1() != nil && clientCertificateManager != nil {
			glog.V(2).Info("Starting client certificate rotation.")
			clientCertificateManager.SetCertificateSigningRequestClient(kubeClient.CertificatesV1beta1().CertificateSigningRequests())
			clientCertificateManager.Start()
		}
		externalKubeClient, err = clientset.NewForConfig(clientConfig)
		if err != nil {
			glog.Warningf("New kubeClient from clientConfig error: %v", err)
		}

		// make a separate client for events
		eventClientConfig := *clientConfig
		eventClientConfig.QPS = float32(s.EventRecordQPS)
		eventClientConfig.Burst = int(s.EventBurst)
		eventClient, err = v1core.NewForConfig(&eventClientConfig)
		if err != nil {
			glog.Warningf("Failed to create API Server client for Events: %v", err)
		}

		// make a separate client for heartbeat with throttling disabled and a timeout attached
		heartbeatClientConfig := *clientConfig
		heartbeatClientConfig.Timeout = s.KubeletConfiguration.NodeStatusUpdateFrequency.Duration
		heartbeatClientConfig.QPS = float32(-1)
		heartbeatClient, err = v1core.NewForConfig(&heartbeatClientConfig)
		if err != nil {
			glog.Warningf("Failed to create API Server client for heartbeat: %v", err)
		}

		kubeDeps.KubeClient = kubeClient
		kubeDeps.ExternalKubeClient = externalKubeClient
		if heartbeatClient != nil {
			kubeDeps.HeartbeatClient = heartbeatClient
		}
		if eventClient != nil {
			kubeDeps.EventClient = eventClient
		}
	}

	// If the kubelet config controller is available, and dynamic config is enabled, start the config and status sync loops
	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) && len(s.DynamicConfigDir.Value()) > 0 &&
		kubeDeps.KubeletConfigController != nil && !standaloneMode && !s.RunOnce {
		kubeDeps.KubeletConfigController.StartSync(kubeDeps.KubeClient, kubeDeps.EventClient, string(nodeName))
	}

	if kubeDeps.Auth == nil {
		auth, err := BuildAuth(nodeName, kubeDeps.ExternalKubeClient, s.KubeletConfiguration)
		if err != nil {
			return err
		}
		kubeDeps.Auth = auth
	}

	if kubeDeps.CAdvisorInterface == nil {
		imageFsInfoProvider := cadvisor.NewImageFsInfoProvider(s.ContainerRuntime, s.RemoteRuntimeEndpoint)
		kubeDeps.CAdvisorInterface, err = cadvisor.New(s.Address, uint(s.CAdvisorPort), imageFsInfoProvider, s.RootDirectory, cadvisor.UsingLegacyCadvisorStats(s.ContainerRuntime, s.RemoteRuntimeEndpoint))
		if err != nil {
			return err
		}
	}

	// Setup event recorder if required.
	makeEventRecorder(kubeDeps, nodeName)

	if kubeDeps.ContainerManager == nil {
		if s.CgroupsPerQOS && s.CgroupRoot == "" {
			glog.Infof("--cgroups-per-qos enabled, but --cgroup-root was not specified.  defaulting to /")
			s.CgroupRoot = "/"
		}
		kubeReserved, err := parseResourceList(s.KubeReserved)
		if err != nil {
			return err
		}
		systemReserved, err := parseResourceList(s.SystemReserved)
		if err != nil {
			return err
		}
		var hardEvictionThresholds []evictionapi.Threshold
		// If the user requested to ignore eviction thresholds, then do not set valid values for hardEvictionThresholds here.
		if !s.ExperimentalNodeAllocatableIgnoreEvictionThreshold {
			hardEvictionThresholds, err = eviction.ParseThresholdConfig([]string{}, s.EvictionHard, nil, nil, nil)
			if err != nil {
				return err
			}
		}
		experimentalQOSReserved, err := cm.ParseQOSReserved(s.ExperimentalQOSReserved)
		if err != nil {
			return err
		}

		devicePluginEnabled := utilfeature.DefaultFeatureGate.Enabled(features.DevicePlugins)

		kubeDeps.ContainerManager, err = cm.NewContainerManager(
			kubeDeps.Mounter,
			kubeDeps.CAdvisorInterface,
			cm.NodeConfig{
				RuntimeCgroupsName:    s.RuntimeCgroups,
				SystemCgroupsName:     s.SystemCgroups,
				KubeletCgroupsName:    s.KubeletCgroups,
				ContainerRuntime:      s.ContainerRuntime,
				CgroupsPerQOS:         s.CgroupsPerQOS,
				CgroupRoot:            s.CgroupRoot,
				CgroupDriver:          s.CgroupDriver,
				KubeletRootDir:        s.RootDirectory,
				ProtectKernelDefaults: s.ProtectKernelDefaults,
				NodeAllocatableConfig: cm.NodeAllocatableConfig{
					KubeReservedCgroupName:   s.KubeReservedCgroup,
					SystemReservedCgroupName: s.SystemReservedCgroup,
					EnforceNodeAllocatable:   sets.NewString(s.EnforceNodeAllocatable...),
					KubeReserved:             kubeReserved,
					SystemReserved:           systemReserved,
					HardEvictionThresholds:   hardEvictionThresholds,
				},
				ExperimentalQOSReserved:               *experimentalQOSReserved,
				ExperimentalCPUManagerPolicy:          s.CPUManagerPolicy,
				ExperimentalCPUManagerReconcilePeriod: s.CPUManagerReconcilePeriod.Duration,
				ExperimentalPodPidsLimit:              s.PodPidsLimit,
			},
			s.FailSwapOn,
			devicePluginEnabled,
			kubeDeps.Recorder)

		if err != nil {
			return err
		}
	}

	if err := checkPermissions(); err != nil {
		glog.Error(err)
	}

	utilruntime.ReallyCrash = s.ReallyCrashForTesting

	rand.Seed(time.Now().UTC().UnixNano())

	// TODO(vmarmol): Do this through container config.
	oomAdjuster := kubeDeps.OOMAdjuster
	if err := oomAdjuster.ApplyOOMScoreAdj(0, int(s.OOMScoreAdj)); err != nil {
		glog.Warning(err)
	}

	if err := RunKubelet(&s.KubeletFlags, &s.KubeletConfiguration, kubeDeps, s.RunOnce); err != nil {
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

// getNodeName returns the node name according to the cloud provider
// if cloud provider is specified. Otherwise, returns the hostname of the node.
func getNodeName(cloud cloudprovider.Interface, hostname string) (types.NodeName, error) {
	if cloud == nil {
		return types.NodeName(hostname), nil
	}

	instances, ok := cloud.Instances()
	if !ok {
		return "", fmt.Errorf("failed to get instances from cloud provider")
	}

	nodeName, err := instances.CurrentNodeName(context.TODO(), hostname)
	if err != nil {
		return "", fmt.Errorf("error fetching current node name from cloud provider: %v", err)
	}

	glog.V(2).Infof("cloud provider determined current node name to be %s", nodeName)

	return nodeName, nil
}

// InitializeTLS checks for a configured TLSCertFile and TLSPrivateKeyFile: if unspecified a new self-signed
// certificate and key file are generated. Returns a configured server.TLSOptions object.
func InitializeTLS(kf *options.KubeletFlags, kc *kubeletconfiginternal.KubeletConfiguration) (*server.TLSOptions, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.RotateKubeletServerCertificate) && kc.TLSCertFile == "" && kc.TLSPrivateKeyFile == "" {
		kc.TLSCertFile = path.Join(kf.CertDirectory, "kubelet.crt")
		kc.TLSPrivateKeyFile = path.Join(kf.CertDirectory, "kubelet.key")

		canReadCertAndKey, err := certutil.CanReadCertAndKey(kc.TLSCertFile, kc.TLSPrivateKeyFile)
		if err != nil {
			return nil, err
		}
		if !canReadCertAndKey {
			cert, key, err := certutil.GenerateSelfSignedCertKey(nodeutil.GetHostname(kf.HostnameOverride), nil, nil)
			if err != nil {
				return nil, fmt.Errorf("unable to generate self signed cert: %v", err)
			}

			if err := certutil.WriteCert(kc.TLSCertFile, cert); err != nil {
				return nil, err
			}

			if err := certutil.WriteKey(kc.TLSPrivateKeyFile, key); err != nil {
				return nil, err
			}

			glog.V(4).Infof("Using self-signed cert (%s, %s)", kc.TLSCertFile, kc.TLSPrivateKeyFile)
		}
	}

	tlsCipherSuites, err := flag.TLSCipherSuites(kc.TLSCipherSuites)
	if err != nil {
		return nil, err
	}

	minTLSVersion, err := flag.TLSVersion(kc.TLSMinVersion)
	if err != nil {
		return nil, err
	}

	tlsOptions := &server.TLSOptions{
		Config: &tls.Config{
			MinVersion:   minTLSVersion,
			CipherSuites: tlsCipherSuites,
		},
		CertFile: kc.TLSCertFile,
		KeyFile:  kc.TLSPrivateKeyFile,
	}

	if len(kc.Authentication.X509.ClientCAFile) > 0 {
		clientCAs, err := certutil.NewPool(kc.Authentication.X509.ClientCAFile)
		if err != nil {
			return nil, fmt.Errorf("unable to load client CA file %s: %v", kc.Authentication.X509.ClientCAFile, err)
		}
		// Specify allowed CAs for client certificates
		tlsOptions.Config.ClientCAs = clientCAs
		// Populate PeerCertificates in requests, but don't reject connections without verified certificates
		tlsOptions.Config.ClientAuth = tls.RequestClientCert
	}

	return tlsOptions, nil
}

func kubeconfigClientConfig(s *options.KubeletServer) (*restclient.Config, error) {
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: s.KubeConfig},
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
}

// createClientConfig creates a client configuration from the command line arguments.
// If --kubeconfig is explicitly set, it will be used.
func createClientConfig(s *options.KubeletServer) (*restclient.Config, error) {
	if s.BootstrapKubeconfig != "" || len(s.KubeConfig) > 0 {
		return kubeconfigClientConfig(s)
	} else {
		return nil, fmt.Errorf("createClientConfig called in standalone mode")
	}
}

// createAPIServerClientConfig generates a client.Config from command line flags
// via createClientConfig and then injects chaos into the configuration via addChaosToClientConfig.
func createAPIServerClientConfig(s *options.KubeletServer) (*restclient.Config, error) {
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

// RunKubelet is responsible for setting up and running a kubelet.  It is used in three different applications:
//   1 Integration tests
//   2 Kubelet binary
//   3 Standalone 'kubernetes' binary
// Eventually, #2 will be replaced with instances of #3
func RunKubelet(kubeFlags *options.KubeletFlags, kubeCfg *kubeletconfiginternal.KubeletConfiguration, kubeDeps *kubelet.Dependencies, runOnce bool) error {
	hostname := nodeutil.GetHostname(kubeFlags.HostnameOverride)
	// Query the cloud provider for our node name, default to hostname if kubeDeps.Cloud == nil
	nodeName, err := getNodeName(kubeDeps.Cloud, hostname)
	if err != nil {
		return err
	}
	// Setup event recorder if required.
	makeEventRecorder(kubeDeps, nodeName)

	// TODO(mtaufen): I moved the validation of these fields here, from UnsecuredKubeletConfig,
	//                so that I could remove the associated fields from KubeletConfiginternal. I would
	//                prefer this to be done as part of an independent validation step on the
	//                KubeletConfiguration. But as far as I can tell, we don't have an explicit
	//                place for validation of the KubeletConfiguration yet.
	hostNetworkSources, err := kubetypes.GetValidatedSources(kubeFlags.HostNetworkSources)
	if err != nil {
		return err
	}

	hostPIDSources, err := kubetypes.GetValidatedSources(kubeFlags.HostPIDSources)
	if err != nil {
		return err
	}

	hostIPCSources, err := kubetypes.GetValidatedSources(kubeFlags.HostIPCSources)
	if err != nil {
		return err
	}

	privilegedSources := capabilities.PrivilegedSources{
		HostNetworkSources: hostNetworkSources,
		HostPIDSources:     hostPIDSources,
		HostIPCSources:     hostIPCSources,
	}
	capabilities.Setup(kubeFlags.AllowPrivileged, privilegedSources, 0)

	credentialprovider.SetPreferredDockercfgPath(kubeFlags.RootDirectory)
	glog.V(2).Infof("Using root directory: %v", kubeFlags.RootDirectory)

	if kubeDeps.OSInterface == nil {
		kubeDeps.OSInterface = kubecontainer.RealOS{}
	}

	k, err := CreateAndInitKubelet(kubeCfg,
		kubeDeps,
		&kubeFlags.ContainerRuntimeOptions,
		kubeFlags.ContainerRuntime,
		kubeFlags.RuntimeCgroups,
		kubeFlags.HostnameOverride,
		kubeFlags.NodeIP,
		kubeFlags.ProviderID,
		kubeFlags.CloudProvider,
		kubeFlags.CertDirectory,
		kubeFlags.RootDirectory,
		kubeFlags.RegisterNode,
		kubeFlags.RegisterWithTaints,
		kubeFlags.AllowedUnsafeSysctls,
		kubeFlags.RemoteRuntimeEndpoint,
		kubeFlags.RemoteImageEndpoint,
		kubeFlags.ExperimentalMounterPath,
		kubeFlags.ExperimentalKernelMemcgNotification,
		kubeFlags.ExperimentalCheckNodeCapabilitiesBeforeMount,
		kubeFlags.ExperimentalNodeAllocatableIgnoreEvictionThreshold,
		kubeFlags.MinimumGCAge,
		kubeFlags.MaxPerPodContainerCount,
		kubeFlags.MaxContainerCount,
		kubeFlags.MasterServiceNamespace,
		kubeFlags.RegisterSchedulable,
		kubeFlags.NonMasqueradeCIDR,
		kubeFlags.KeepTerminatedPodVolumes,
		kubeFlags.NodeLabels,
		kubeFlags.SeccompProfileRoot,
		kubeFlags.BootstrapCheckpointPath)
	if err != nil {
		return fmt.Errorf("failed to create kubelet: %v", err)
	}

	// NewMainKubelet should have set up a pod source config if one didn't exist
	// when the builder was run. This is just a precaution.
	if kubeDeps.PodConfig == nil {
		return fmt.Errorf("failed to create kubelet, pod source config was nil")
	}
	podCfg := kubeDeps.PodConfig

	rlimit.RlimitNumFiles(uint64(kubeCfg.MaxOpenFiles))

	// process pods and exit.
	if runOnce {
		if _, err := k.RunOnce(podCfg.Updates()); err != nil {
			return fmt.Errorf("runonce failed: %v", err)
		}
		glog.Infof("Started kubelet as runonce")
	} else {
		startKubelet(k, podCfg, kubeCfg, kubeDeps, kubeFlags.EnableServer)
		glog.Infof("Started kubelet")
	}
	return nil
}

func startKubelet(k kubelet.Bootstrap, podCfg *config.PodConfig, kubeCfg *kubeletconfiginternal.KubeletConfiguration, kubeDeps *kubelet.Dependencies, enableServer bool) {
	// start the kubelet
	go wait.Until(func() { k.Run(podCfg.Updates()) }, 0, wait.NeverStop)

	// start the kubelet server
	if enableServer {
		go wait.Until(func() {
			k.ListenAndServe(net.ParseIP(kubeCfg.Address), uint(kubeCfg.Port), kubeDeps.TLSOptions, kubeDeps.Auth, kubeCfg.EnableDebuggingHandlers, kubeCfg.EnableContentionProfiling)
		}, 0, wait.NeverStop)
	}
	if kubeCfg.ReadOnlyPort > 0 {
		go wait.Until(func() {
			k.ListenAndServeReadOnly(net.ParseIP(kubeCfg.Address), uint(kubeCfg.ReadOnlyPort))
		}, 0, wait.NeverStop)
	}
}

func CreateAndInitKubelet(kubeCfg *kubeletconfiginternal.KubeletConfiguration,
	kubeDeps *kubelet.Dependencies,
	crOptions *config.ContainerRuntimeOptions,
	containerRuntime string,
	runtimeCgroups string,
	hostnameOverride string,
	nodeIP string,
	providerID string,
	cloudProvider string,
	certDirectory string,
	rootDirectory string,
	registerNode bool,
	registerWithTaints []api.Taint,
	allowedUnsafeSysctls []string,
	remoteRuntimeEndpoint string,
	remoteImageEndpoint string,
	experimentalMounterPath string,
	experimentalKernelMemcgNotification bool,
	experimentalCheckNodeCapabilitiesBeforeMount bool,
	experimentalNodeAllocatableIgnoreEvictionThreshold bool,
	minimumGCAge metav1.Duration,
	maxPerPodContainerCount int32,
	maxContainerCount int32,
	masterServiceNamespace string,
	registerSchedulable bool,
	nonMasqueradeCIDR string,
	keepTerminatedPodVolumes bool,
	nodeLabels map[string]string,
	seccompProfileRoot string,
	bootstrapCheckpointPath string) (k kubelet.Bootstrap, err error) {
	// TODO: block until all sources have delivered at least one update to the channel, or break the sync loop
	// up into "per source" synchronizations

	k, err = kubelet.NewMainKubelet(kubeCfg,
		kubeDeps,
		crOptions,
		containerRuntime,
		runtimeCgroups,
		hostnameOverride,
		nodeIP,
		providerID,
		cloudProvider,
		certDirectory,
		rootDirectory,
		registerNode,
		registerWithTaints,
		allowedUnsafeSysctls,
		remoteRuntimeEndpoint,
		remoteImageEndpoint,
		experimentalMounterPath,
		experimentalKernelMemcgNotification,
		experimentalCheckNodeCapabilitiesBeforeMount,
		experimentalNodeAllocatableIgnoreEvictionThreshold,
		minimumGCAge,
		maxPerPodContainerCount,
		maxContainerCount,
		masterServiceNamespace,
		registerSchedulable,
		nonMasqueradeCIDR,
		keepTerminatedPodVolumes,
		nodeLabels,
		seccompProfileRoot,
		bootstrapCheckpointPath)
	if err != nil {
		return nil, err
	}

	k.BirthCry()

	k.StartGarbageCollection()

	return k, nil
}

// parseResourceList parses the given configuration map into an API
// ResourceList or returns an error.
func parseResourceList(m map[string]string) (v1.ResourceList, error) {
	if len(m) == 0 {
		return nil, nil
	}
	rl := make(v1.ResourceList)
	for k, v := range m {
		switch v1.ResourceName(k) {
		// CPU, memory and local storage resources are supported.
		case v1.ResourceCPU, v1.ResourceMemory, v1.ResourceEphemeralStorage:
			q, err := resource.ParseQuantity(v)
			if err != nil {
				return nil, err
			}
			if q.Sign() == -1 {
				return nil, fmt.Errorf("resource quantity for %q cannot be negative: %v", k, v)
			}
			rl[v1.ResourceName(k)] = q
		default:
			return nil, fmt.Errorf("cannot reserve %q resource", k)
		}
	}
	return rl, nil
}

// BootstrapKubeletConfigController constructs and bootstrap a configuration controller
func BootstrapKubeletConfigController(defaultConfig *kubeletconfiginternal.KubeletConfiguration,
	dynamicConfigDir string) (*kubeletconfiginternal.KubeletConfiguration, *dynamickubeletconfig.Controller, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
		return nil, nil, fmt.Errorf("failed to bootstrap Kubelet config controller, you must enable the DynamicKubeletConfig feature gate")
	}
	if len(dynamicConfigDir) == 0 {
		return nil, nil, fmt.Errorf("cannot bootstrap Kubelet config controller, --dynamic-config-dir was not provided")
	}

	// compute absolute path and bootstrap controller
	dir, err := filepath.Abs(dynamicConfigDir)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get absolute path for --dynamic-config-dir=%s", dynamicConfigDir)
	}
	// get the latest KubeletConfiguration checkpoint from disk, or return the default config if no valid checkpoints exist
	c := dynamickubeletconfig.NewController(defaultConfig, dir)
	kc, err := c.Bootstrap()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to determine a valid configuration, error: %v", err)
	}
	return kc, c, nil
}

// RunDockershim only starts the dockershim in current process. This is only used for cri validate testing purpose
// TODO(random-liu): Move this to a separate binary.
func RunDockershim(f *options.KubeletFlags, c *kubeletconfiginternal.KubeletConfiguration) error {
	r := &f.ContainerRuntimeOptions

	// Initialize docker client configuration.
	dockerClientConfig := &dockershim.ClientConfig{
		DockerEndpoint:            r.DockerEndpoint,
		RuntimeRequestTimeout:     c.RuntimeRequestTimeout.Duration,
		ImagePullProgressDeadline: r.ImagePullProgressDeadline.Duration,
	}

	// Initialize network plugin settings.
	nh := &kubelet.NoOpLegacyHost{}
	pluginSettings := dockershim.NetworkPluginSettings{
		HairpinMode:       kubeletconfiginternal.HairpinMode(c.HairpinMode),
		NonMasqueradeCIDR: f.NonMasqueradeCIDR,
		PluginName:        r.NetworkPluginName,
		PluginConfDir:     r.CNIConfDir,
		PluginBinDir:      r.CNIBinDir,
		MTU:               int(r.NetworkPluginMTU),
		LegacyRuntimeHost: nh,
	}

	// Initialize streaming configuration. (Not using TLS now)
	streamingConfig := &streaming.Config{
		// Use a relative redirect (no scheme or host).
		BaseURL:                         &url.URL{Path: "/cri/"},
		StreamIdleTimeout:               c.StreamingConnectionIdleTimeout.Duration,
		StreamCreationTimeout:           streaming.DefaultConfig.StreamCreationTimeout,
		SupportedRemoteCommandProtocols: streaming.DefaultConfig.SupportedRemoteCommandProtocols,
		SupportedPortForwardProtocols:   streaming.DefaultConfig.SupportedPortForwardProtocols,
	}

	ds, err := dockershim.NewDockerService(dockerClientConfig, r.PodSandboxImage, streamingConfig, &pluginSettings,
		f.RuntimeCgroups, c.CgroupDriver, r.DockershimRootDirectory, r.DockerDisableSharedPID)
	if err != nil {
		return err
	}
	glog.V(2).Infof("Starting the GRPC server for the docker CRI shim.")
	server := dockerremote.NewDockerServer(f.RemoteRuntimeEndpoint, ds)
	if err := server.Start(); err != nil {
		return err
	}

	// Start the streaming server
	addr := net.JoinHostPort(c.Address, strconv.Itoa(int(c.Port)))
	return http.ListenAndServe(addr, ds)
}
