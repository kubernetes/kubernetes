/*
Copyright 2014 The Kubernetes Authors.

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

// Package app does all of the work necessary to configure and run a
// Kubernetes app process.
package app

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	goruntime "runtime"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientgoclientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/userspace"
	"k8s.io/kubernetes/pkg/proxy/winuserspace"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/configz"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilnetsh "k8s.io/kubernetes/pkg/util/netsh"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/resourcecontainer"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	"k8s.io/kubernetes/pkg/version/verflag"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

const (
	proxyModeUserspace = "userspace"
	proxyModeIPTables  = "iptables"
)

// checkKnownProxyMode returns true if proxyMode is valid.
func checkKnownProxyMode(proxyMode string) bool {
	switch proxyMode {
	case "", proxyModeUserspace, proxyModeIPTables:
		return true
	}
	return false
}

// Options contains everything necessary to create and run a proxy server.
type Options struct {
	// ConfigFile is the location of the proxy server's configuration file.
	ConfigFile string
	// WriteConfigTo is the path where the default configuration will be written.
	WriteConfigTo string
	// CleanupAndExit, when true, makes the proxy server clean up iptables rules, then exit.
	CleanupAndExit bool

	// config is the proxy server's configuration object.
	config *componentconfig.KubeProxyConfiguration

	// The fields below here are placeholders for flags that can't be directly mapped into
	// componentconfig.KubeProxyConfiguration.
	//
	// TODO remove these fields once the deprecated flags are removed.

	// master is used to override the kubeconfig's URL to the apiserver.
	master string
	// healthzPort is the port to be used by the healthz server.
	healthzPort int32

	scheme *runtime.Scheme
	codecs serializer.CodecFactory
}

// AddFlags adds flags to fs and binds them to options.
func AddFlags(options *Options, fs *pflag.FlagSet) {
	fs.StringVar(&options.ConfigFile, "config", options.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&options.WriteConfigTo, "write-config-to", options.WriteConfigTo, "If set, write the default configuration values to this file and exit.")
	fs.BoolVar(&options.CleanupAndExit, "cleanup-iptables", options.CleanupAndExit, "If true cleanup iptables rules and exit.")

	// All flags below here are deprecated and will eventually be removed.

	fs.Var(componentconfig.IPVar{Val: &options.config.BindAddress}, "bind-address", "The IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&options.master, "master", options.master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.Int32Var(&options.healthzPort, "healthz-port", options.healthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.Var(componentconfig.IPVar{Val: &options.config.HealthzBindAddress}, "healthz-bind-address", "The IP address and port for the health check server to serve on (set to 0.0.0.0 for all interfaces)")
	fs.Int32Var(options.config.OOMScoreAdj, "oom-score-adj", util.Int32PtrDerefOr(options.config.OOMScoreAdj, int32(qos.KubeProxyOOMScoreAdj)), "The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
	fs.StringVar(&options.config.ResourceContainer, "resource-container", options.config.ResourceContainer, "Absolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).")
	fs.MarkDeprecated("resource-container", "This feature will be removed in a later release.")
	fs.StringVar(&options.config.ClientConnection.KubeConfigFile, "kubeconfig", options.config.ClientConnection.KubeConfigFile, "Path to kubeconfig file with authorization information (the master location is set by the master flag).")
	fs.Var(componentconfig.PortRangeVar{Val: &options.config.PortRange}, "proxy-port-range", "Range of host ports (beginPort-endPort, inclusive) that may be consumed in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.")
	fs.StringVar(&options.config.HostnameOverride, "hostname-override", options.config.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.Var(&options.config.Mode, "proxy-mode", "Which proxy mode to use: 'userspace' (older) or 'iptables' (faster). If blank, use the best-available proxy (currently iptables).  If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are insufficient, this always falls back to the userspace proxy.")
	fs.Int32Var(options.config.IPTables.MasqueradeBit, "iptables-masquerade-bit", util.Int32PtrDerefOr(options.config.IPTables.MasqueradeBit, 14), "If using the pure iptables proxy, the bit of the fwmark space to mark packets requiring SNAT with.  Must be within the range [0, 31].")
	fs.DurationVar(&options.config.IPTables.SyncPeriod.Duration, "iptables-sync-period", options.config.IPTables.SyncPeriod.Duration, "The maximum interval of how often iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.DurationVar(&options.config.IPTables.MinSyncPeriod.Duration, "iptables-min-sync-period", options.config.IPTables.MinSyncPeriod.Duration, "The minimum interval of how often the iptables rules can be refreshed as endpoints and services change (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&options.config.ConfigSyncPeriod.Duration, "config-sync-period", options.config.ConfigSyncPeriod.Duration, "How often configuration from the apiserver is refreshed.  Must be greater than 0.")
	fs.BoolVar(&options.config.IPTables.MasqueradeAll, "masquerade-all", options.config.IPTables.MasqueradeAll, "If using the pure iptables proxy, SNAT everything (this not commonly needed)")
	fs.StringVar(&options.config.ClusterCIDR, "cluster-cidr", options.config.ClusterCIDR, "The CIDR range of pods in the cluster. It is used to bridge traffic coming from outside of the cluster. If not provided, no off-cluster bridging will be performed.")
	fs.StringVar(&options.config.ClientConnection.ContentType, "kube-api-content-type", options.config.ClientConnection.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&options.config.ClientConnection.QPS, "kube-api-qps", options.config.ClientConnection.QPS, "QPS to use while talking with kubernetes apiserver")
	fs.IntVar(&options.config.ClientConnection.Burst, "kube-api-burst", options.config.ClientConnection.Burst, "Burst to use while talking with kubernetes apiserver")
	fs.DurationVar(&options.config.UDPIdleTimeout.Duration, "udp-timeout", options.config.UDPIdleTimeout.Duration, "How long an idle UDP connection will be kept open (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxy-mode=userspace")
	fs.Int32Var(&options.config.Conntrack.Max, "conntrack-max", options.config.Conntrack.Max,
		"Maximum number of NAT connections to track (0 to leave as-is). This overrides conntrack-max-per-core and conntrack-min.")
	fs.MarkDeprecated("conntrack-max", "This feature will be removed in a later release.")
	fs.Int32Var(&options.config.Conntrack.MaxPerCore, "conntrack-max-per-core", options.config.Conntrack.MaxPerCore,
		"Maximum number of NAT connections to track per CPU core (0 to leave the limit as-is and ignore conntrack-min).")
	fs.Int32Var(&options.config.Conntrack.Min, "conntrack-min", options.config.Conntrack.Min,
		"Minimum number of conntrack entries to allocate, regardless of conntrack-max-per-core (set conntrack-max-per-core=0 to leave the limit as-is).")
	fs.DurationVar(&options.config.Conntrack.TCPEstablishedTimeout.Duration, "conntrack-tcp-timeout-established", options.config.Conntrack.TCPEstablishedTimeout.Duration, "Idle timeout for established TCP connections (0 to leave as-is)")
	fs.DurationVar(
		&options.config.Conntrack.TCPCloseWaitTimeout.Duration, "conntrack-tcp-timeout-close-wait",
		options.config.Conntrack.TCPCloseWaitTimeout.Duration,
		"NAT timeout for TCP connections in the CLOSE_WAIT state")
	fs.BoolVar(&options.config.EnableProfiling, "profiling", options.config.EnableProfiling, "If true enables profiling via web interface on /debug/pprof handler.")

	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

func NewOptions() (*Options, error) {
	o := &Options{
		config:      new(componentconfig.KubeProxyConfiguration),
		healthzPort: 10256,
	}

	o.scheme = runtime.NewScheme()
	o.codecs = serializer.NewCodecFactory(o.scheme)

	if err := componentconfig.AddToScheme(o.scheme); err != nil {
		return nil, err
	}
	if err := v1alpha1.AddToScheme(o.scheme); err != nil {
		return nil, err
	}

	return o, nil
}

// Complete completes all the required options.
func (o *Options) Complete() error {
	if len(o.ConfigFile) == 0 && len(o.WriteConfigTo) == 0 {
		glog.Warning("WARNING: all flags other than --config, --write-config-to, and --cleanup-iptables are deprecated. Please begin using a config file ASAP.")
		o.applyDeprecatedHealthzPortToConfig()
	}

	return nil
}

// Validate validates all the required options.
func (o *Options) Validate(args []string) error {
	if len(args) != 0 {
		return errors.New("no arguments are supported")
	}

	return nil
}

func (o *Options) Run() error {
	config := o.config

	if len(o.WriteConfigTo) > 0 {
		return o.writeConfigFile()
	}

	if len(o.ConfigFile) > 0 {
		if c, err := o.loadConfigFromFile(o.ConfigFile); err != nil {
			return err
		} else {
			config = c
			// Make sure we apply the feature gate settings in the config file.
			utilfeature.DefaultFeatureGate.Set(config.FeatureGates)
		}
	}

	proxyServer, err := NewProxyServer(config, o.CleanupAndExit, o.scheme, o.master)
	if err != nil {
		return err
	}

	return proxyServer.Run()
}

func (o *Options) writeConfigFile() error {
	var encoder runtime.Encoder
	mediaTypes := o.codecs.SupportedMediaTypes()
	for _, info := range mediaTypes {
		if info.MediaType == "application/yaml" {
			encoder = info.Serializer
			break
		}
	}
	if encoder == nil {
		return errors.New("unable to locate yaml encoder")
	}
	encoder = json.NewYAMLSerializer(json.DefaultMetaFactory, o.scheme, o.scheme)
	encoder = o.codecs.EncoderForVersion(encoder, v1alpha1.SchemeGroupVersion)

	configFile, err := os.Create(o.WriteConfigTo)
	if err != nil {
		return err
	}
	defer configFile.Close()

	if err := encoder.Encode(o.config, configFile); err != nil {
		return err
	}

	fmt.Printf("Wrote configuration to: %s\n", o.WriteConfigTo)

	return nil
}

// applyDeprecatedHealthzPortToConfig sets o.config.HealthzBindAddress from
// flags passed on the command line based on the following rules:
//
// 1. If --healthz-port is 0, disable the healthz server.
// 2. Otherwise, use the value of --healthz-port for the port portion of
//    o.config.HealthzBindAddress
func (o *Options) applyDeprecatedHealthzPortToConfig() {
	if o.healthzPort == 0 {
		o.config.HealthzBindAddress = ""
		return
	}

	index := strings.Index(o.config.HealthzBindAddress, ":")
	if index != -1 {
		o.config.HealthzBindAddress = o.config.HealthzBindAddress[0:index]
	}

	o.config.HealthzBindAddress = fmt.Sprintf("%s:%d", o.config.HealthzBindAddress, o.healthzPort)
}

// loadConfigFromFile loads the contents of file and decodes it as a
// KubeProxyConfiguration object.
func (o *Options) loadConfigFromFile(file string) (*componentconfig.KubeProxyConfiguration, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return o.loadConfig(data)
}

// loadConfig decodes data as a KubeProxyConfiguration object.
func (o *Options) loadConfig(data []byte) (*componentconfig.KubeProxyConfiguration, error) {
	configObj, gvk, err := o.codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	config, ok := configObj.(*componentconfig.KubeProxyConfiguration)
	if !ok {
		return nil, fmt.Errorf("got unexpected config type: %v", gvk)
	}
	return config, nil
}

func (o *Options) applyDefaults(in *componentconfig.KubeProxyConfiguration) (*componentconfig.KubeProxyConfiguration, error) {
	external, err := o.scheme.ConvertToVersion(in, v1alpha1.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	o.scheme.Default(external)

	internal, err := o.scheme.ConvertToVersion(external, componentconfig.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	out := internal.(*componentconfig.KubeProxyConfiguration)

	return out, nil
}

// NewProxyCommand creates a *cobra.Command object with default parameters
func NewProxyCommand() *cobra.Command {
	opts, err := NewOptions()
	if err != nil {
		glog.Fatalf("Unable to initialize command options: %v", err)
	}

	cmd := &cobra.Command{
		Use: "kube-proxy",
		Long: `The Kubernetes network proxy runs on each node. This
reflects services as defined in the Kubernetes API on each node and can do simple
TCP,UDP stream forwarding or round robin TCP,UDP forwarding across a set of backends.
Service cluster ips and ports are currently found through Docker-links-compatible
environment variables specifying ports opened by the service proxy. There is an optional
addon that provides cluster DNS for these cluster IPs. The user must create a service
with the apiserver API to configure the proxy.`,
		Run: func(cmd *cobra.Command, args []string) {
			verflag.PrintAndExitIfRequested()
			cmdutil.CheckErr(opts.Complete())
			cmdutil.CheckErr(opts.Validate(args))
			cmdutil.CheckErr(opts.Run())
		},
	}

	opts.config, err = opts.applyDefaults(opts.config)
	if err != nil {
		glog.Fatalf("unable to create flag defaults: %v", err)
	}

	flags := cmd.Flags()
	AddFlags(opts, flags)

	cmd.MarkFlagFilename("config", "yaml", "yml", "json")

	return cmd
}

// ProxyServer represents all the parameters required to start the Kubernetes proxy server. All
// fields are required.
type ProxyServer struct {
	Client                 clientset.Interface
	EventClient            v1core.EventsGetter
	IptInterface           utiliptables.Interface
	Proxier                proxy.ProxyProvider
	Broadcaster            record.EventBroadcaster
	Recorder               record.EventRecorder
	ConntrackConfiguration componentconfig.KubeProxyConntrackConfiguration
	Conntracker            Conntracker // if nil, ignored
	ProxyMode              string
	NodeRef                *clientv1.ObjectReference
	CleanupAndExit         bool
	MetricsBindAddress     string
	EnableProfiling        bool
	OOMScoreAdj            *int32
	ResourceContainer      string
	ConfigSyncPeriod       time.Duration
	ServiceEventHandler    proxyconfig.ServiceHandler
	EndpointsEventHandler  proxyconfig.EndpointsHandler
	HealthzServer          *healthcheck.HealthzServer
}

// createClients creates a kube client and an event client from the given config and masterOverride.
// TODO remove masterOverride when CLI flags are removed.
func createClients(config componentconfig.ClientConnectionConfiguration, masterOverride string) (clientset.Interface, v1core.EventsGetter, error) {
	if len(config.KubeConfigFile) == 0 && len(masterOverride) == 0 {
		glog.Warningf("Neither --kubeconfig nor --master was specified. Using default API client. This might not work.")
	}

	// This creates a client, first loading any specified kubeconfig
	// file, and then overriding the Master flag, if non-empty.
	kubeConfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: config.KubeConfigFile},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: masterOverride}}).ClientConfig()
	if err != nil {
		return nil, nil, err
	}

	kubeConfig.AcceptContentTypes = config.AcceptContentTypes
	kubeConfig.ContentType = config.ContentType
	kubeConfig.QPS = config.QPS
	//TODO make config struct use int instead of int32?
	kubeConfig.Burst = int(config.Burst)

	client, err := clientset.NewForConfig(kubeConfig)
	if err != nil {
		return nil, nil, err
	}

	eventClient, err := clientgoclientset.NewForConfig(kubeConfig)
	if err != nil {
		return nil, nil, err
	}

	return client, eventClient, nil
}

// NewProxyServer returns a new ProxyServer.
func NewProxyServer(config *componentconfig.KubeProxyConfiguration, cleanupAndExit bool, scheme *runtime.Scheme, master string) (*ProxyServer, error) {
	if config == nil {
		return nil, errors.New("config is required")
	}

	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(config)
	} else {
		return nil, fmt.Errorf("unable to register configz: %s", err)
	}

	protocol := utiliptables.ProtocolIpv4
	if net.ParseIP(config.BindAddress).To4() == nil {
		protocol = utiliptables.ProtocolIpv6
	}

	var netshInterface utilnetsh.Interface
	var iptInterface utiliptables.Interface
	var dbus utildbus.Interface

	// Create a iptables utils.
	execer := exec.New()

	if goruntime.GOOS == "windows" {
		netshInterface = utilnetsh.New(execer)
	} else {
		dbus = utildbus.New()
		iptInterface = utiliptables.New(execer, dbus, protocol)
	}

	// We omit creation of pretty much everything if we run in cleanup mode
	if cleanupAndExit {
		return &ProxyServer{IptInterface: iptInterface, CleanupAndExit: cleanupAndExit}, nil
	}

	client, eventClient, err := createClients(config.ClientConnection, master)
	if err != nil {
		return nil, err
	}

	// Create event recorder
	hostname := nodeutil.GetHostname(config.HostnameOverride)
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme, clientv1.EventSource{Component: "kube-proxy", Host: hostname})

	var healthzServer *healthcheck.HealthzServer
	if len(config.HealthzBindAddress) > 0 {
		healthzServer = healthcheck.NewDefaultHealthzServer(config.HealthzBindAddress, 2*config.IPTables.SyncPeriod.Duration)
	}

	var proxier proxy.ProxyProvider
	var serviceEventHandler proxyconfig.ServiceHandler
	var endpointsEventHandler proxyconfig.EndpointsHandler

	proxyMode := getProxyMode(string(config.Mode), iptInterface, iptables.LinuxKernelCompatTester{})
	if proxyMode == proxyModeIPTables {
		glog.V(0).Info("Using iptables Proxier.")
		var nodeIP net.IP
		if config.BindAddress != "0.0.0.0" {
			nodeIP = net.ParseIP(config.BindAddress)
		} else {
			nodeIP = getNodeIP(client, hostname)
		}
		if config.IPTables.MasqueradeBit == nil {
			// MasqueradeBit must be specified or defaulted.
			return nil, fmt.Errorf("unable to read IPTables MasqueradeBit from config")
		}

		// TODO this has side effects that should only happen when Run() is invoked.
		proxierIPTables, err := iptables.NewProxier(
			iptInterface,
			utilsysctl.New(),
			execer,
			config.IPTables.SyncPeriod.Duration,
			config.IPTables.MinSyncPeriod.Duration,
			config.IPTables.MasqueradeAll,
			int(*config.IPTables.MasqueradeBit),
			config.ClusterCIDR,
			hostname,
			nodeIP,
			recorder,
			healthzServer,
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
		iptables.RegisterMetrics()
		proxier = proxierIPTables
		serviceEventHandler = proxierIPTables
		endpointsEventHandler = proxierIPTables
		// No turning back. Remove artifacts that might still exist from the userspace Proxier.
		glog.V(0).Info("Tearing down userspace rules.")
		// TODO this has side effects that should only happen when Run() is invoked.
		userspace.CleanupLeftovers(iptInterface)
	} else {
		glog.V(0).Info("Using userspace Proxier.")
		if goruntime.GOOS == "windows" {
			// This is a proxy.LoadBalancer which NewProxier needs but has methods we don't need for
			// our config.EndpointsConfigHandler.
			loadBalancer := winuserspace.NewLoadBalancerRR()
			// set EndpointsHandler to our loadBalancer
			endpointsEventHandler = loadBalancer
			proxierUserspace, err := winuserspace.NewProxier(
				loadBalancer,
				net.ParseIP(config.BindAddress),
				netshInterface,
				*utilnet.ParsePortRangeOrDie(config.PortRange),
				// TODO @pires replace below with default values, if applicable
				config.IPTables.SyncPeriod.Duration,
				config.UDPIdleTimeout.Duration,
			)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}
			serviceEventHandler = proxierUserspace
			proxier = proxierUserspace
		} else {
			// This is a proxy.LoadBalancer which NewProxier needs but has methods we don't need for
			// our config.EndpointsConfigHandler.
			loadBalancer := userspace.NewLoadBalancerRR()
			// set EndpointsConfigHandler to our loadBalancer
			endpointsEventHandler = loadBalancer

			// TODO this has side effects that should only happen when Run() is invoked.
			proxierUserspace, err := userspace.NewProxier(
				loadBalancer,
				net.ParseIP(config.BindAddress),
				iptInterface,
				execer,
				*utilnet.ParsePortRangeOrDie(config.PortRange),
				config.IPTables.SyncPeriod.Duration,
				config.IPTables.MinSyncPeriod.Duration,
				config.UDPIdleTimeout.Duration,
			)
			if err != nil {
				return nil, fmt.Errorf("unable to create proxier: %v", err)
			}
			serviceEventHandler = proxierUserspace
			proxier = proxierUserspace
		}
		// Remove artifacts from the pure-iptables Proxier, if not on Windows.
		if goruntime.GOOS != "windows" {
			glog.V(0).Info("Tearing down pure-iptables proxy rules.")
			// TODO this has side effects that should only happen when Run() is invoked.
			iptables.CleanupLeftovers(iptInterface)
		}
	}

	// Add iptables reload function, if not on Windows.
	if goruntime.GOOS != "windows" {
		iptInterface.AddReloadFunc(proxier.Sync)
	}

	nodeRef := &clientv1.ObjectReference{
		Kind:      "Node",
		Name:      hostname,
		UID:       types.UID(hostname),
		Namespace: "",
	}

	return &ProxyServer{
		Client:                 client,
		EventClient:            eventClient,
		IptInterface:           iptInterface,
		Proxier:                proxier,
		Broadcaster:            eventBroadcaster,
		Recorder:               recorder,
		ConntrackConfiguration: config.Conntrack,
		Conntracker:            &realConntracker{},
		ProxyMode:              proxyMode,
		NodeRef:                nodeRef,
		MetricsBindAddress:     config.MetricsBindAddress,
		EnableProfiling:        config.EnableProfiling,
		OOMScoreAdj:            config.OOMScoreAdj,
		ResourceContainer:      config.ResourceContainer,
		ConfigSyncPeriod:       config.ConfigSyncPeriod.Duration,
		ServiceEventHandler:    serviceEventHandler,
		EndpointsEventHandler:  endpointsEventHandler,
		HealthzServer:          healthzServer,
	}, nil
}

// Run runs the specified ProxyServer.  This should never exit (unless CleanupAndExit is set).
func (s *ProxyServer) Run() error {
	// remove iptables rules and exit
	if s.CleanupAndExit {
		encounteredError := userspace.CleanupLeftovers(s.IptInterface)
		encounteredError = iptables.CleanupLeftovers(s.IptInterface) || encounteredError
		if encounteredError {
			return errors.New("encountered an error while tearing down rules.")
		}
		return nil
	}

	// TODO(vmarmol): Use container config for this.
	var oomAdjuster *oom.OOMAdjuster
	if s.OOMScoreAdj != nil {
		oomAdjuster = oom.NewOOMAdjuster()
		if err := oomAdjuster.ApplyOOMScoreAdj(0, int(*s.OOMScoreAdj)); err != nil {
			glog.V(2).Info(err)
		}
	}

	if len(s.ResourceContainer) != 0 {
		// Run in its own container.
		if err := resourcecontainer.RunInResourceContainer(s.ResourceContainer); err != nil {
			glog.Warningf("Failed to start in resource-only container %q: %v", s.ResourceContainer, err)
		} else {
			glog.V(2).Infof("Running in resource-only container %q", s.ResourceContainer)
		}
	}

	if s.Broadcaster != nil && s.EventClient != nil {
		s.Broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: s.EventClient.Events("")})
	}

	// Start up a healthz server if requested
	if s.HealthzServer != nil {
		s.HealthzServer.Run()
	}

	// Start up a metrics server if requested
	if len(s.MetricsBindAddress) > 0 {
		mux := http.NewServeMux()
		healthz.InstallHandler(mux)
		mux.HandleFunc("/proxyMode", func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, "%s", s.ProxyMode)
		})
		mux.Handle("/metrics", prometheus.Handler())
		if s.EnableProfiling {
			mux.HandleFunc("/debug/pprof/", pprof.Index)
			mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
			mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
			mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		}
		configz.InstallHandler(mux)
		go wait.Until(func() {
			err := http.ListenAndServe(s.MetricsBindAddress, mux)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("starting metrics server failed: %v", err))
			}
		}, 5*time.Second, wait.NeverStop)
	}

	// Tune conntrack, if requested
	if s.Conntracker != nil && goruntime.GOOS != "windows" {
		max, err := getConntrackMax(s.ConntrackConfiguration)
		if err != nil {
			return err
		}
		if max > 0 {
			err := s.Conntracker.SetMax(max)
			if err != nil {
				if err != readOnlySysFSError {
					return err
				}
				// readOnlySysFSError is caused by a known docker issue (https://github.com/docker/docker/issues/24000),
				// the only remediation we know is to restart the docker daemon.
				// Here we'll send an node event with specific reason and message, the
				// administrator should decide whether and how to handle this issue,
				// whether to drain the node and restart docker.
				// TODO(random-liu): Remove this when the docker bug is fixed.
				const message = "DOCKER RESTART NEEDED (docker issue #24000): /sys is read-only: " +
					"cannot modify conntrack limits, problems may arise later."
				s.Recorder.Eventf(s.NodeRef, api.EventTypeWarning, err.Error(), message)
			}
		}

		if s.ConntrackConfiguration.TCPEstablishedTimeout.Duration > 0 {
			timeout := int(s.ConntrackConfiguration.TCPEstablishedTimeout.Duration / time.Second)
			if err := s.Conntracker.SetTCPEstablishedTimeout(timeout); err != nil {
				return err
			}
		}

		if s.ConntrackConfiguration.TCPCloseWaitTimeout.Duration > 0 {
			timeout := int(s.ConntrackConfiguration.TCPCloseWaitTimeout.Duration / time.Second)
			if err := s.Conntracker.SetTCPCloseWaitTimeout(timeout); err != nil {
				return err
			}
		}
	}

	informerFactory := informers.NewSharedInformerFactory(s.Client, s.ConfigSyncPeriod)

	// Create configs (i.e. Watches for Services and Endpoints)
	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.
	serviceConfig := proxyconfig.NewServiceConfig(informerFactory.Core().InternalVersion().Services(), s.ConfigSyncPeriod)
	serviceConfig.RegisterEventHandler(s.ServiceEventHandler)
	go serviceConfig.Run(wait.NeverStop)

	endpointsConfig := proxyconfig.NewEndpointsConfig(informerFactory.Core().InternalVersion().Endpoints(), s.ConfigSyncPeriod)
	endpointsConfig.RegisterEventHandler(s.EndpointsEventHandler)
	go endpointsConfig.Run(wait.NeverStop)

	// This has to start after the calls to NewServiceConfig and NewEndpointsConfig because those
	// functions must configure their shared informer event handlers first.
	go informerFactory.Start(wait.NeverStop)

	// Birth Cry after the birth is successful
	s.birthCry()

	// Just loop forever for now...
	s.Proxier.SyncLoop()
	return nil
}

func getConntrackMax(config componentconfig.KubeProxyConntrackConfiguration) (int, error) {
	if config.Max > 0 {
		if config.MaxPerCore > 0 {
			return -1, fmt.Errorf("invalid config: Conntrack Max and Conntrack MaxPerCore are mutually exclusive")
		}
		glog.V(3).Infof("getConntrackMax: using absolute conntrack-max (deprecated)")
		return int(config.Max), nil
	}
	if config.MaxPerCore > 0 {
		floor := int(config.Min)
		scaled := int(config.MaxPerCore) * goruntime.NumCPU()
		if scaled > floor {
			glog.V(3).Infof("getConntrackMax: using scaled conntrack-max-per-core")
			return scaled, nil
		}
		glog.V(3).Infof("getConntrackMax: using conntrack-min")
		return floor, nil
	}
	return 0, nil
}

func getProxyMode(proxyMode string, iptver iptables.IPTablesVersioner, kcompat iptables.KernelCompatTester) string {
	if proxyMode == proxyModeUserspace {
		return proxyModeUserspace
	}

	if len(proxyMode) > 0 && proxyMode != proxyModeIPTables {
		glog.Warningf("Flag proxy-mode=%q unknown, assuming iptables proxy", proxyMode)
	}

	return tryIPTablesProxy(iptver, kcompat)
}

func tryIPTablesProxy(iptver iptables.IPTablesVersioner, kcompat iptables.KernelCompatTester) string {
	// guaranteed false on error, error only necessary for debugging
	useIPTablesProxy, err := iptables.CanUseIPTablesProxier(iptver, kcompat)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("can't determine whether to use iptables proxy, using userspace proxier: %v", err))
		return proxyModeUserspace
	}
	if useIPTablesProxy {
		return proxyModeIPTables
	}
	// Fallback.
	glog.V(1).Infof("Can't use iptables proxy, using userspace proxier")
	return proxyModeUserspace
}

func (s *ProxyServer) birthCry() {
	s.Recorder.Eventf(s.NodeRef, api.EventTypeNormal, "Starting", "Starting kube-proxy.")
}

func getNodeIP(client clientset.Interface, hostname string) net.IP {
	var nodeIP net.IP
	node, err := client.Core().Nodes().Get(hostname, metav1.GetOptions{})
	if err != nil {
		glog.Warningf("Failed to retrieve node info: %v", err)
		return nil
	}
	nodeIP, err = nodeutil.InternalGetNodeHostIP(node)
	if err != nil {
		glog.Warningf("Failed to retrieve node IP: %v", err)
		return nil
	}
	return nodeIP
}
