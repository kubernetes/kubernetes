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

	"k8s.io/apiserver/pkg/healthz"
	clientgoclientset "k8s.io/client-go/kubernetes"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	componentconfigv1alpha1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/client/chaosclient"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	clientauth "k8s.io/kubernetes/pkg/client/unversioned/auth"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/server"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/cert"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	utilconfig "k8s.io/kubernetes/pkg/util/config"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/util/flock"
	kubeio "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/rlimit"
	"k8s.io/kubernetes/pkg/version"
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

// UnsecuredKubeletDeps returns a KubeletDeps suitable for being run, or an error if the server setup
// is not valid.  It will not start any background processes, and does not include authentication/authorization
func UnsecuredKubeletDeps(s *options.KubeletServer) (*kubelet.KubeletDeps, error) {

	// Initialize the TLS Options
	tlsOptions, err := InitializeTLS(&s.KubeletConfiguration)
	if err != nil {
		return nil, err
	}

	mounter := mount.New(s.ExperimentalMounterPath)
	var writer kubeio.Writer = &kubeio.StdWriter{}
	if s.Containerized {
		glog.V(2).Info("Running kubelet in containerized mode (experimental)")
		mounter = mount.NewNsenterMounter()
		writer = &kubeio.NsenterWriter{}
	}

	var dockerClient dockertools.DockerInterface
	if s.ContainerRuntime == "docker" {
		dockerClient = dockertools.ConnectToDockerOrDie(s.DockerEndpoint, s.RuntimeRequestTimeout.Duration,
			s.ImagePullProgressDeadline.Duration)
	} else {
		dockerClient = nil
	}

	return &kubelet.KubeletDeps{
		Auth:               nil, // default does not enforce auth[nz]
		CAdvisorInterface:  nil, // cadvisor.New launches background processes (bg http.ListenAndServe, and some bg cleaners), not set here
		Cloud:              nil, // cloud provider might start background processes
		ContainerManager:   nil,
		DockerClient:       dockerClient,
		KubeClient:         nil,
		ExternalKubeClient: nil,
		Mounter:            mounter,
		NetworkPlugins:     ProbeNetworkPlugins(s.NetworkPluginDir, s.CNIConfDir, s.CNIBinDir),
		OOMAdjuster:        oom.NewOOMAdjuster(),
		OSInterface:        kubecontainer.RealOS{},
		Writer:             writer,
		VolumePlugins:      ProbeVolumePlugins(s.VolumePluginDir),
		TLSOptions:         tlsOptions,
	}, nil
}

func getKubeClient(s *options.KubeletServer) (*clientset.Clientset, error) {
	clientConfig, err := CreateAPIServerClientConfig(s)
	if err == nil {
		kubeClient, err := clientset.NewForConfig(clientConfig)
		if err != nil {
			return nil, err
		}
		return kubeClient, nil
	}
	return nil, err
}

// Tries to download the kubelet-<node-name> configmap from "kube-system" namespace via the API server and returns a JSON string or error
func getRemoteKubeletConfig(s *options.KubeletServer, kubeDeps *kubelet.KubeletDeps) (string, error) {
	// TODO(mtaufen): should probably cache clientset and pass into this function rather than regenerate on every request
	kubeClient, err := getKubeClient(s)
	if err != nil {
		return "", err
	}

	configmap, err := func() (*v1.ConfigMap, error) {
		var nodename types.NodeName
		hostname := nodeutil.GetHostname(s.HostnameOverride)

		if kubeDeps != nil && kubeDeps.Cloud != nil {
			instances, ok := kubeDeps.Cloud.Instances()
			if !ok {
				err = fmt.Errorf("failed to get instances from cloud provider, can't determine nodename.")
				return nil, err
			}
			nodename, err = instances.CurrentNodeName(hostname)
			if err != nil {
				err = fmt.Errorf("error fetching current instance name from cloud provider: %v", err)
				return nil, err
			}
			// look for kubelet-<node-name> configmap from "kube-system"
			configmap, err := kubeClient.CoreV1Client.ConfigMaps("kube-system").Get(fmt.Sprintf("kubelet-%s", nodename), metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
			return configmap, nil
		}
		// No cloud provider yet, so can't get the nodename via Cloud.Instances().CurrentNodeName(hostname), try just using the hostname
		configmap, err := kubeClient.CoreV1Client.ConfigMaps("kube-system").Get(fmt.Sprintf("kubelet-%s", hostname), metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("cloud provider was nil, and attempt to use hostname to find config resulted in: %v", err)
		}
		return configmap, nil
	}()
	if err != nil {
		return "", err
	}

	// When we create the KubeletConfiguration configmap, we put a json string
	// representation of the config in a `kubelet.config` key.
	jsonstr, ok := configmap.Data["kubelet.config"]
	if !ok {
		return "", fmt.Errorf("KubeletConfiguration configmap did not contain a value with key `kubelet.config`")
	}

	return jsonstr, nil
}

func startKubeletConfigSyncLoop(s *options.KubeletServer, currentKC string) {
	glog.Infof("Starting Kubelet configuration sync loop")
	go func() {
		wait.PollInfinite(30*time.Second, func() (bool, error) {
			glog.Infof("Checking API server for new Kubelet configuration.")
			remoteKC, err := getRemoteKubeletConfig(s, nil)
			if err == nil {
				// Detect new config by comparing with the last JSON string we extracted.
				if remoteKC != currentKC {
					glog.Info("Found new Kubelet configuration via API server, restarting!")
					os.Exit(0)
				}
			} else {
				glog.Infof("Did not find a configuration for this Kubelet via API server: %v", err)
			}
			return false, nil // Always return (false, nil) so we poll forever.
		})
	}()
}

// Try to check for config on the API server, return that config if we get it, and start
// a background thread that checks for updates to configs.
func initKubeletConfigSync(s *options.KubeletServer) (*componentconfig.KubeletConfiguration, error) {
	jsonstr, err := getRemoteKubeletConfig(s, nil)
	if err == nil {
		// We will compare future API server config against the config we just got (jsonstr):
		startKubeletConfigSyncLoop(s, jsonstr)

		// Convert json from API server to external type struct, and convert that to internal type struct
		extKC := componentconfigv1alpha1.KubeletConfiguration{}
		err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), []byte(jsonstr), &extKC)
		if err != nil {
			return nil, err
		}
		api.Scheme.Default(&extKC)
		kc := componentconfig.KubeletConfiguration{}
		err = api.Scheme.Convert(&extKC, &kc, nil)
		if err != nil {
			return nil, err
		}
		return &kc, nil
	} else {
		// Couldn't get a configuration from the API server yet.
		// Restart as soon as anything comes back from the API server.
		startKubeletConfigSyncLoop(s, "")
		return nil, err
	}
}

// Run runs the specified KubeletServer with the given KubeletDeps.  This should never exit.
// The kubeDeps argument may be nil - if so, it is initialized from the settings on KubeletServer.
// Otherwise, the caller is assumed to have set up the KubeletDeps object and a default one will
// not be generated.
func Run(s *options.KubeletServer, kubeDeps *kubelet.KubeletDeps) error {
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

func setConfigz(cz *configz.Config, kc *componentconfig.KubeletConfiguration) {
	tmp := componentconfigv1alpha1.KubeletConfiguration{}
	api.Scheme.Convert(kc, &tmp, nil)
	cz.Set(tmp)
}

func initConfigz(kc *componentconfig.KubeletConfiguration) (*configz.Config, error) {
	cz, err := configz.New("componentconfig")
	if err == nil {
		setConfigz(cz, kc)
	} else {
		glog.Errorf("unable to register configz: %s", err)
	}
	return cz, err
}

func run(s *options.KubeletServer, kubeDeps *kubelet.KubeletDeps) (err error) {
	// TODO: this should be replaced by a --standalone flag
	standaloneMode := (len(s.APIServerList) == 0 && !s.RequireKubeConfig)

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

	// Set feature gates based on the value in KubeletConfiguration
	err = utilconfig.DefaultFeatureGate.Set(s.KubeletConfiguration.FeatureGates)
	if err != nil {
		return err
	}

	// Register current configuration with /configz endpoint
	cfgz, cfgzErr := initConfigz(&s.KubeletConfiguration)
	if utilconfig.DefaultFeatureGate.DynamicKubeletConfig() {
		// Look for config on the API server. If it exists, replace s.KubeletConfiguration
		// with it and continue. initKubeletConfigSync also starts the background thread that checks for new config.

		// Don't do dynamic Kubelet configuration in runonce mode
		if s.RunOnce == false {
			remoteKC, err := initKubeletConfigSync(s)
			if err == nil {
				// Update s (KubeletServer) with new config from API server
				s.KubeletConfiguration = *remoteKC
				// Ensure that /configz is up to date with the new config
				if cfgzErr != nil {
					glog.Errorf("was unable to register configz before due to %s, will not be able to set now", cfgzErr)
				} else {
					setConfigz(cfgz, &s.KubeletConfiguration)
				}
				// Update feature gates from the new config
				err = utilconfig.DefaultFeatureGate.Set(s.KubeletConfiguration.FeatureGates)
				if err != nil {
					return err
				}
			}
		}
	}

	if kubeDeps == nil {
		var kubeClient, eventClient *clientset.Clientset
		var externalKubeClient clientgoclientset.Interface
		var cloud cloudprovider.Interface

		if s.CloudProvider != componentconfigv1alpha1.AutoDetectCloudProvider {
			cloud, err = cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
			if err != nil {
				return err
			}
			if cloud == nil {
				glog.V(2).Infof("No cloud provider specified: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)
			} else {
				glog.V(2).Infof("Successfully initialized cloud provider: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)
			}
		}

		if s.BootstrapKubeconfig != "" {
			nodeName, err := getNodeName(cloud, nodeutil.GetHostname(s.HostnameOverride))
			if err != nil {
				return err
			}
			if err := bootstrapClientCert(s.KubeConfig.Value(), s.BootstrapKubeconfig, s.CertDirectory, nodeName); err != nil {
				return err
			}
		}

		clientConfig, err := CreateAPIServerClientConfig(s)
		if err == nil {
			kubeClient, err = clientset.NewForConfig(clientConfig)
			if err != nil {
				glog.Warningf("New kubeClient from clientConfig error: %v", err)
			}
			// make a separate client for events
			eventClientConfig := *clientConfig
			eventClientConfig.QPS = float32(s.EventRecordQPS)
			eventClientConfig.Burst = int(s.EventBurst)
			eventClient, err = clientset.NewForConfig(&eventClientConfig)
			if err != nil {
				glog.Warningf("Failed to create API Server client: %v", err)
			}
		} else {
			if s.RequireKubeConfig {
				return fmt.Errorf("invalid kubeconfig: %v", err)
			}
			if standaloneMode {
				glog.Warningf("No API client: %v", err)
			}
		}

		// client-go and kuberenetes generated clients are incompatible because the runtime
		// and runtime/serializer types have been duplicated in client-go.  This means that
		// you can't reasonably convert from one to the other and its impossible for a single
		// type to fulfill both interfaces.  Because of that, we have to build the clients
		// up from scratch twice.
		// TODO eventually the kubelet should only use the client-go library
		clientGoConfig, err := createAPIServerClientGoConfig(s)
		if err == nil {
			externalKubeClient, err = clientgoclientset.NewForConfig(clientGoConfig)
			if err != nil {
				glog.Warningf("New kubeClient from clientConfig error: %v", err)
			}
		} else {
			if s.RequireKubeConfig {
				return fmt.Errorf("invalid kubeconfig: %v", err)
			}
			if standaloneMode {
				glog.Warningf("No API client: %v", err)
			}
		}

		kubeDeps, err = UnsecuredKubeletDeps(s)
		if err != nil {
			return err
		}

		kubeDeps.Cloud = cloud
		kubeDeps.KubeClient = kubeClient
		kubeDeps.ExternalKubeClient = externalKubeClient
		kubeDeps.EventClient = eventClient
	}

	if kubeDeps.Auth == nil {
		nodeName, err := getNodeName(kubeDeps.Cloud, nodeutil.GetHostname(s.HostnameOverride))
		if err != nil {
			return err
		}

		auth, err := buildAuth(nodeName, kubeDeps.ExternalKubeClient, s.KubeletConfiguration)
		if err != nil {
			return err
		}
		kubeDeps.Auth = auth
	}

	if kubeDeps.CAdvisorInterface == nil {
		kubeDeps.CAdvisorInterface, err = cadvisor.New(uint(s.CAdvisorPort), s.ContainerRuntime, s.RootDirectory)
		if err != nil {
			return err
		}
	}

	if kubeDeps.ContainerManager == nil {
		if s.SystemCgroups != "" && s.CgroupRoot == "" {
			return fmt.Errorf("invalid configuration: system container was specified and cgroup root was not specified")
		}
		kubeDeps.ContainerManager, err = cm.NewContainerManager(
			kubeDeps.Mounter,
			kubeDeps.CAdvisorInterface,
			cm.NodeConfig{
				RuntimeCgroupsName:    s.RuntimeCgroups,
				SystemCgroupsName:     s.SystemCgroups,
				KubeletCgroupsName:    s.KubeletCgroups,
				ContainerRuntime:      s.ContainerRuntime,
				CgroupsPerQOS:         s.ExperimentalCgroupsPerQOS,
				CgroupRoot:            s.CgroupRoot,
				CgroupDriver:          s.CgroupDriver,
				ProtectKernelDefaults: s.ProtectKernelDefaults,
				EnableCRI:             s.EnableCRI,
			},
			s.ExperimentalFailSwapOn)

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

	if err := RunKubelet(&s.KubeletConfiguration, kubeDeps, s.RunOnce, standaloneMode); err != nil {
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

	nodeName, err := instances.CurrentNodeName(hostname)
	if err != nil {
		return "", fmt.Errorf("error fetching current node name from cloud provider: %v", err)
	}

	glog.V(2).Infof("cloud provider determined current node name to be %s", nodeName)

	return nodeName, nil
}

// InitializeTLS checks for a configured TLSCertFile and TLSPrivateKeyFile: if unspecified a new self-signed
// certificate and key file are generated. Returns a configured server.TLSOptions object.
func InitializeTLS(kc *componentconfig.KubeletConfiguration) (*server.TLSOptions, error) {
	if kc.TLSCertFile == "" && kc.TLSPrivateKeyFile == "" {
		kc.TLSCertFile = path.Join(kc.CertDirectory, "kubelet.crt")
		kc.TLSPrivateKeyFile = path.Join(kc.CertDirectory, "kubelet.key")

		canReadCertAndKey, err := certutil.CanReadCertAndKey(kc.TLSCertFile, kc.TLSPrivateKeyFile)
		if err != nil {
			return nil, err
		}
		if !canReadCertAndKey {
			cert, key, err := certutil.GenerateSelfSignedCertKey(nodeutil.GetHostname(kc.HostnameOverride), nil, nil)
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
	tlsOptions := &server.TLSOptions{
		Config: &tls.Config{
			// Can't use SSLv3 because of POODLE and BEAST
			// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
			// Can't use TLSv1.1 because of RC4 cipher usage
			MinVersion: tls.VersionTLS12,
		},
		CertFile: kc.TLSCertFile,
		KeyFile:  kc.TLSPrivateKeyFile,
	}

	if len(kc.Authentication.X509.ClientCAFile) > 0 {
		clientCAs, err := cert.NewPool(kc.Authentication.X509.ClientCAFile)
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
	if s.RequireKubeConfig {
		// Ignores the values of s.APIServerList
		return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
			&clientcmd.ClientConfigLoadingRules{ExplicitPath: s.KubeConfig.Value()},
			&clientcmd.ConfigOverrides{},
		).ClientConfig()
	}
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: s.KubeConfig.Value()},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: s.APIServerList[0]}},
	).ClientConfig()
}

// createClientConfig creates a client configuration from the command line
// arguments. If --kubeconfig is explicitly set, it will be used. If it is
// not set, we attempt to load the default kubeconfig file, and if we cannot,
// we fall back to the default client with no auth - this fallback does not, in
// and of itself, constitute an error.
func createClientConfig(s *options.KubeletServer) (*restclient.Config, error) {
	if s.RequireKubeConfig {
		return kubeconfigClientConfig(s)
	}

	// TODO: handle a new --standalone flag that bypasses kubeconfig loading and returns no error.
	// DEPRECATED: all subsequent code is deprecated
	if len(s.APIServerList) == 0 {
		return nil, fmt.Errorf("no api servers specified")
	}
	// TODO: adapt Kube client to support LB over several servers
	if len(s.APIServerList) > 1 {
		glog.Infof("Multiple api servers specified.  Picking first one")
	}

	if s.KubeConfig.Provided() {
		return kubeconfigClientConfig(s)
	}
	// If KubeConfig was not provided, try to load the default file, then fall back
	// to a default auth config.
	clientConfig, err := kubeconfigClientConfig(s)
	if err != nil {
		glog.Warningf("Could not load kubeconfig file %s: %v. Using default client config instead.", s.KubeConfig, err)

		authInfo := &clientauth.Info{}
		authConfig, err := authInfo.MergeWithConfig(restclient.Config{})
		if err != nil {
			return nil, err
		}
		authConfig.Host = s.APIServerList[0]
		clientConfig = &authConfig
	}
	return clientConfig, nil
}

// CreateAPIServerClientConfig generates a client.Config from command line flags,
// including api-server-list, via createClientConfig and then injects chaos into
// the configuration via addChaosToClientConfig. This func is exported to support
// integration with third party kubelet extensions (e.g. kubernetes-mesos).
func CreateAPIServerClientConfig(s *options.KubeletServer) (*restclient.Config, error) {
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
func RunKubelet(kubeCfg *componentconfig.KubeletConfiguration, kubeDeps *kubelet.KubeletDeps, runOnce bool, standaloneMode bool) error {
	hostname := nodeutil.GetHostname(kubeCfg.HostnameOverride)
	// Query the cloud provider for our node name, default to hostname if kcfg.Cloud == nil
	nodeName, err := getNodeName(kubeDeps.Cloud, hostname)
	if err != nil {
		return err
	}

	eventBroadcaster := record.NewBroadcaster()
	kubeDeps.Recorder = eventBroadcaster.NewRecorder(v1.EventSource{Component: "kubelet", Host: string(nodeName)})
	eventBroadcaster.StartLogging(glog.V(3).Infof)
	if kubeDeps.EventClient != nil {
		glog.V(4).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeDeps.EventClient.Events("")})
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
	k, err := builder(kubeCfg, kubeDeps, standaloneMode)
	if err != nil {
		return fmt.Errorf("failed to create kubelet: %v", err)
	}

	// NewMainKubelet should have set up a pod source config if one didn't exist
	// when the builder was run. This is just a precaution.
	if kubeDeps.PodConfig == nil {
		return fmt.Errorf("failed to create kubelet, pod source config was nil!")
	}
	podCfg := kubeDeps.PodConfig

	rlimit.RlimitNumFiles(uint64(kubeCfg.MaxOpenFiles))

	// TODO(dawnchen): remove this once we deprecated old debian containervm images.
	// This is a workaround for issue: https://github.com/opencontainers/runc/issues/726
	// The current chosen number is consistent with most of other os dist.
	const maxKeysPath = "/proc/sys/kernel/keys/root_maxkeys"
	const minKeys uint64 = 1000000
	key, err := ioutil.ReadFile(maxKeysPath)
	if err != nil {
		glog.Errorf("Cannot read keys quota in %s", maxKeysPath)
	} else {
		fields := strings.Fields(string(key))
		nKey, _ := strconv.ParseUint(fields[0], 10, 64)
		if nKey < minKeys {
			glog.Infof("Setting keys quota in %s to %d", maxKeysPath, minKeys)
			err = ioutil.WriteFile(maxKeysPath, []byte(fmt.Sprintf("%d", uint64(minKeys))), 0644)
			if err != nil {
				glog.Warningf("Failed to update %s: %v", maxKeysPath, err)
			}
		}
	}
	const maxBytesPath = "/proc/sys/kernel/keys/root_maxbytes"
	const minBytes uint64 = 25000000
	bytes, err := ioutil.ReadFile(maxBytesPath)
	if err != nil {
		glog.Errorf("Cannot read keys bytes in %s", maxBytesPath)
	} else {
		fields := strings.Fields(string(bytes))
		nByte, _ := strconv.ParseUint(fields[0], 10, 64)
		if nByte < minBytes {
			glog.Infof("Setting keys bytes in %s to %d", maxBytesPath, minBytes)
			err = ioutil.WriteFile(maxBytesPath, []byte(fmt.Sprintf("%d", uint64(minBytes))), 0644)
			if err != nil {
				glog.Warningf("Failed to update %s: %v", maxBytesPath, err)
			}
		}
	}

	// process pods and exit.
	if runOnce {
		if _, err := k.RunOnce(podCfg.Updates()); err != nil {
			return fmt.Errorf("runonce failed: %v", err)
		}
		glog.Infof("Started kubelet %s as runonce", version.Get().String())
	} else {
		startKubelet(k, podCfg, kubeCfg, kubeDeps)
		glog.Infof("Started kubelet %s", version.Get().String())
	}
	return nil
}

func startKubelet(k kubelet.KubeletBootstrap, podCfg *config.PodConfig, kubeCfg *componentconfig.KubeletConfiguration, kubeDeps *kubelet.KubeletDeps) {
	// start the kubelet
	go wait.Until(func() { k.Run(podCfg.Updates()) }, 0, wait.NeverStop)

	// start the kubelet server
	if kubeCfg.EnableServer {
		go wait.Until(func() {
			k.ListenAndServe(net.ParseIP(kubeCfg.Address), uint(kubeCfg.Port), kubeDeps.TLSOptions, kubeDeps.Auth, kubeCfg.EnableDebuggingHandlers)
		}, 0, wait.NeverStop)
	}
	if kubeCfg.ReadOnlyPort > 0 {
		go wait.Until(func() {
			k.ListenAndServeReadOnly(net.ParseIP(kubeCfg.Address), uint(kubeCfg.ReadOnlyPort))
		}, 0, wait.NeverStop)
	}
}

func CreateAndInitKubelet(kubeCfg *componentconfig.KubeletConfiguration, kubeDeps *kubelet.KubeletDeps, standaloneMode bool) (k kubelet.KubeletBootstrap, err error) {
	// TODO: block until all sources have delivered at least one update to the channel, or break the sync loop
	// up into "per source" synchronizations

	k, err = kubelet.NewMainKubelet(kubeCfg, kubeDeps, standaloneMode)
	if err != nil {
		return nil, err
	}

	k.BirthCry()

	k.StartGarbageCollection()

	return k, nil
}
