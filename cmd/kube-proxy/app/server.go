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
	_ "net/http/pprof"
	"runtime"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/userspace"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/config"
	"k8s.io/kubernetes/pkg/util/configz"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/resourcecontainer"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

const (
	proxyModeUserspace              = "userspace"
	proxyModeIPTables               = "iptables"
	ExperimentalProxyModeAnnotation = "net.experimental.kubernetes.io/proxy-mode"
	betaProxyModeAnnotation         = "net.beta.kubernetes.io/proxy-mode"
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
	// CleanupAndExit, when true, makes the proxy server clean up iptables rules, then exit.
	CleanupAndExit bool

	// config is the proxy server's config.
	config *componentconfig.KubeProxyConfiguration

	// The fields below here are placeholders for flags that can't be directly
	// mapped into componentconfig.KubeProxyConfiguration.
	//
	// TODO remove these fields once the deprecate flags are removed.

	// master is used to override the kubeconfig's url to the apiserver.
	master string
	// healthzPort is the port to be used by the healthz server.
	healthzPort int32
}

// applyDefaults applies default values to in.
func applyDefaults(in *componentconfig.KubeProxyConfiguration) (*componentconfig.KubeProxyConfiguration, error) {
	external, err := api.Scheme.ConvertToVersion(in, v1alpha1.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	internal, err := api.Scheme.ConvertToVersion(external, componentconfig.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	out := internal.(*componentconfig.KubeProxyConfiguration)

	return out, nil
}

// AddFlags adds flags to fs and binds them to options.
func AddFlags(options *Options, fs *pflag.FlagSet) error {
	fs.StringVar(&options.ConfigFile, "config", options.ConfigFile, "The config file to use")
	fs.BoolVar(&options.CleanupAndExit, "cleanup-iptables", options.CleanupAndExit, "If true cleanup iptables rules and exit.")

	// All flags below here is deprecated and will eventually be removed
	fs.Var(componentconfig.IPVar{Val: &options.config.BindAddress}, "bind-address", "The IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&options.master, "master", options.master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.Int32Var(&options.healthzPort, "healthz-port", options.healthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.Var(componentconfig.IPVar{Val: &options.config.HealthzBindAddress}, "healthz-bind-address", "The IP address for the health check server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)")
	fs.Int32Var(options.config.OOMScoreAdj, "oom-score-adj", util.Int32PtrDerefOr(options.config.OOMScoreAdj, int32(qos.KubeProxyOOMScoreAdj)), "The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
	fs.StringVar(&options.config.ResourceContainer, "resource-container", options.config.ResourceContainer, "Absolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).")
	fs.MarkDeprecated("resource-container", "This feature will be removed in a later release.")
	fs.StringVar(&options.config.ClientConnection.KubeConfigFile, "kubeconfig", options.config.ClientConnection.KubeConfigFile, "Path to kubeconfig file with authorization information (the master location is set by the master flag).")
	fs.Var(componentconfig.PortRangeVar{Val: &options.config.PortRange}, "proxy-port-range", "Range of host ports (beginPort-endPort, inclusive) that may be consumed in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.")
	fs.StringVar(&options.config.HostnameOverride, "hostname-override", options.config.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.Var(&options.config.Mode, "proxy-mode", "Which proxy mode to use: 'userspace' (older) or 'iptables' (faster). If blank, look at the Node object on the Kubernetes API and respect the '"+ExperimentalProxyModeAnnotation+"' annotation if provided.  Otherwise use the best-available proxy (currently iptables).  If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are insufficient, this always falls back to the userspace proxy.")
	fs.Int32Var(options.config.IPTables.MasqueradeBit, "iptables-masquerade-bit", util.Int32PtrDerefOr(options.config.IPTables.MasqueradeBit, 14), "If using the pure iptables proxy, the bit of the fwmark space to mark packets requiring SNAT with.  Must be within the range [0, 31].")
	fs.DurationVar(&options.config.IPTables.SyncPeriod.Duration, "iptables-sync-period", options.config.IPTables.SyncPeriod.Duration, "How often iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.DurationVar(&options.config.ConfigSyncPeriod.Duration, "config-sync-period", options.config.ConfigSyncPeriod.Duration, "How often configuration from the apiserver is refreshed.  Must be greater than 0.")
	fs.BoolVar(&options.config.IPTables.MasqueradeAll, "masquerade-all", options.config.IPTables.MasqueradeAll, "If using the pure iptables proxy, SNAT everything")
	fs.StringVar(&options.config.ClusterCIDR, "cluster-cidr", options.config.ClusterCIDR, "The CIDR range of pods in the cluster. It is used to bridge traffic coming from outside of the cluster. If not provided, no off-cluster bridging will be performed.")
	fs.StringVar(&options.config.ClientConnection.ContentType, "kube-api-content-type", options.config.ClientConnection.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&options.config.ClientConnection.QPS, "kube-api-qps", options.config.ClientConnection.QPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&options.config.ClientConnection.Burst, "kube-api-burst", options.config.ClientConnection.Burst, "Burst to use while talking with kubernetes apiserver")
	fs.DurationVar(&options.config.UDPIdleTimeout.Duration, "udp-timeout", options.config.UDPIdleTimeout.Duration, "How long an idle UDP connection will be kept open (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxy-mode=userspace")
	fs.Int32Var(&options.config.Conntrack.Max, "conntrack-max", options.config.Conntrack.Max,
		"Maximum number of NAT connections to track (0 to leave as-is). This overrides conntrack-max-per-core and conntrack-min.")
	fs.MarkDeprecated("conntrack-max", "This feature will be removed in a later release.")
	fs.Int32Var(&options.config.Conntrack.MaxPerCore, "conntrack-max-per-core", options.config.Conntrack.MaxPerCore,
		"Maximum number of NAT connections to track per CPU core (0 to leave the limit as-is and ignore conntrack-min).")
	fs.Int32Var(&options.config.Conntrack.Min, "conntrack-min", options.config.Conntrack.Min,
		"Minimum number of conntrack entries to allocate, regardless of conntrack-max-per-core (set conntrack-max-per-core=0 to leave the limit as-is).")
	fs.DurationVar(&options.config.Conntrack.TCPEstablishedTimeout.Duration, "conntrack-tcp-timeout-established", options.config.Conntrack.TCPEstablishedTimeout.Duration, "Idle timeout for established TCP connections (0 to leave as-is)")

	config.DefaultFeatureGate.AddFlag(fs)

	return nil
}

// Complete completes all the required options.
func (o Options) Complete() error {
	return nil
}

// Validate validates all the required options.
func (o Options) Validate(args []string) error {
	if len(args) != 0 {
		return errors.New("no arguments are supported")
	}

	return nil
}

// Run creates and runs a proxy server.
func (o Options) Run() error {
	if len(o.ConfigFile) == 0 {
		glog.Warning("WARNING: all flags other than --config and --cleanup-iptables are deprecated. Please move to using a config file ASAP.")
		o.applyDeprecatedHealthzPortToConfig()
	} else {
		if c, err := loadConfigFromFile(o.ConfigFile); err != nil {
			return err
		} else {
			o.config = c
			// Make sure we apply the feature gate settings in the config file!
			config.DefaultFeatureGate.Set(o.config.FeatureGates)
		}
	}

	proxyServer, err := o.NewProxyServer()
	if err != nil {
		return err
	}

	return proxyServer.Run()
}

// applyDeprecatedHealthzPortToConfig sets o.config.HealthzBindAddress from
// flags passed on the command line based on the following rules:
//
// 1. If --healthz-port is 0, disable the healthz server.
// 2. Otherwise, use the value of --healthz-port for the port portion of
//    o.config.HealthzBindAddress
func (o Options) applyDeprecatedHealthzPortToConfig() {
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
func loadConfigFromFile(file string) (*componentconfig.KubeProxyConfiguration, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	configObj, gvk, err := api.Codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	config, ok := configObj.(*componentconfig.KubeProxyConfiguration)
	if !ok {
		return nil, fmt.Errorf("got unexpected config type: %v", gvk)
	}
	return config, nil
}

// NewProxyCommand creates a *cobra.Command object with default parameters
func NewProxyCommand() *cobra.Command {
	opts := Options{
		config:      new(componentconfig.KubeProxyConfiguration),
		healthzPort: 10249,
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
		Run: func(c *cobra.Command, args []string) {
			cmdutil.CheckErr(opts.Complete())
			cmdutil.CheckErr(opts.Validate(args))
			cmdutil.CheckErr(opts.Run())
		},
	}

	var err error
	opts.config, err = applyDefaults(opts.config)
	if err != nil {
		glog.Fatalf("unable to create flag defaults: %v", err)
	}

	flags := cmd.Flags()
	AddFlags(&opts, flags)

	cmd.MarkFlagFilename("config", "yaml", "yml", "json")

	return cmd
}

// ProxyServer represents all the parameters required to start the Kuberetes
// proxy server. All fields are required.
type ProxyServer struct {
	Client                 clientset.Interface
	IptInterface           utiliptables.Interface
	Proxier                proxy.ProxyProvider
	ServiceConfigHandler   proxyconfig.ServiceConfigHandler
	EndpointsConfigHandler proxyconfig.EndpointsConfigHandler
	Broadcaster            record.EventBroadcaster
	Recorder               record.EventRecorder
	ConntrackConfiguration componentconfig.KubeProxyConntrackConfiguration
	Conntracker            Conntracker // if nil, ignored
	ProxyMode              string
	NodeRef                *api.ObjectReference
	CleanupAndExit         bool
	HealthzBindAddress     string
	OOMScoreAdj            *int32
	ResourceContainer      string
	ConfigSyncPeriod       time.Duration
}

// createKubeClient creates a client from the given config and masterOverride.
// TODO remove masterOverride when cli flags are removed.
func createKubeClient(config componentconfig.ClientConnectionConfiguration, masterOverride string) (clientset.Interface, error) {
	// This creates a client using the specified kubeconfig file and optional
	// master override.
	clientConfig, err := clientcmd.BuildConfigFromFlags(masterOverride, config.KubeConfigFile)
	if err != nil {
		return nil, err
	}

	clientConfig.AcceptContentTypes = config.AcceptContentTypes
	clientConfig.ContentType = config.ContentType
	clientConfig.QPS = config.QPS
	//TODO make config struct use int instead of int32?
	clientConfig.Burst = int(config.Burst)

	return clientset.NewForConfig(clientConfig)
}

// NewProxyServer returns a new ProxyServer.
func (o Options) NewProxyServer() (*ProxyServer, error) {
	if o.config == nil {
		return nil, errors.New("config must be set")
	}

	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(o.config)
	} else {
		// TODO should this return the error or is it ok to proceed?
		utilruntime.HandleError(fmt.Errorf("unable to register configz: %s", err))
	}

	glog.V(4).Infof("Using configuration: %#v", o.config)

	protocol := utiliptables.ProtocolIpv4
	if net.ParseIP(o.config.BindAddress).To4() == nil {
		protocol = utiliptables.ProtocolIpv6
	}

	// Create a iptables utils.
	execer := exec.New()
	dbus := utildbus.New()
	iptInterface := utiliptables.New(execer, dbus, protocol)

	proxyServer := &ProxyServer{
		IptInterface: iptInterface,
	}

	// We omit creation of pretty much everything if we run in cleanup mode
	if o.CleanupAndExit {
		return proxyServer, nil
	}

	client, err := createKubeClient(o.config.ClientConnection, o.master)
	if err != nil {
		return nil, err
	}

	// Create event recorder
	hostname := nodeutil.GetHostname(o.config.HostnameOverride)
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "kube-proxy", Host: hostname})

	var proxier proxy.ProxyProvider
	var endpointsHandler proxyconfig.EndpointsConfigHandler

	proxyMode := getProxyMode(string(o.config.Mode), client.Core().Nodes(), hostname, iptInterface, iptables.LinuxKernelCompatTester{})
	if proxyMode == proxyModeIPTables {
		glog.V(0).Info("Using iptables Proxier.")
		if o.config.IPTables.MasqueradeBit == nil {
			// MasqueradeBit must be specified or defaulted.
			return nil, fmt.Errorf("unable to read IPTables MasqueradeBit from config")
		}
		// TODO this has side effects that should only happen when Run() is invoked.
		proxierIPTables, err := iptables.NewProxier(iptInterface, utilsysctl.New(), execer, o.config.IPTables.SyncPeriod.Duration, o.config.IPTables.MasqueradeAll, int(*o.config.IPTables.MasqueradeBit), o.config.ClusterCIDR, hostname, getNodeIP(client, hostname))
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
		proxier = proxierIPTables
		endpointsHandler = proxierIPTables

		// No turning back. Remove artifacts that might still exist from the userspace Proxier.
		glog.V(0).Info("Tearing down userspace rules.")
		// TODO this has side effects that should only happen when Run() is invoked.
		userspace.CleanupLeftovers(iptInterface)
	} else {
		glog.V(0).Info("Using userspace Proxier.")
		// This is a proxy.LoadBalancer which NewProxier needs but has methods we don't need for
		// our config.EndpointsConfigHandler.
		loadBalancer := userspace.NewLoadBalancerRR()
		// set EndpointsConfigHandler to our loadBalancer
		endpointsHandler = loadBalancer

		// TODO this has side effects that should only happen when Run() is invoked.
		proxierUserspace, err := userspace.NewProxier(
			loadBalancer,
			net.ParseIP(o.config.BindAddress),
			iptInterface,
			*utilnet.ParsePortRangeOrDie(o.config.PortRange),
			o.config.IPTables.SyncPeriod.Duration,
			o.config.UDPIdleTimeout.Duration,
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
		proxier = proxierUserspace

		// Remove artifacts from the pure-iptables Proxier.
		glog.V(0).Info("Tearing down pure-iptables proxy rules.")
		// TODO this has side effects that should only happen when Run() is invoked.
		iptables.CleanupLeftovers(iptInterface)
	}
	iptInterface.AddReloadFunc(proxier.Sync)

	nodeRef := &api.ObjectReference{
		Kind:      "Node",
		Name:      hostname,
		UID:       types.UID(hostname),
		Namespace: "",
	}

	return &ProxyServer{
		Client:                 client,
		IptInterface:           iptInterface,
		Proxier:                proxier,
		ServiceConfigHandler:   proxier,
		EndpointsConfigHandler: endpointsHandler,
		Broadcaster:            eventBroadcaster,
		Recorder:               recorder,
		ConntrackConfiguration: o.config.Conntrack,
		Conntracker:            realConntracker{},
		ProxyMode:              proxyMode,
		NodeRef:                nodeRef,
		HealthzBindAddress:     o.config.HealthzBindAddress,
		OOMScoreAdj:            o.config.OOMScoreAdj,
		ResourceContainer:      o.config.ResourceContainer,
		ConfigSyncPeriod:       o.config.ConfigSyncPeriod.Duration,
	}, nil
}

// Run runs the specified ProxyServer. This should never exit (unless CleanupAndExit is set).
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
	if s.OOMScoreAdj != nil {
		oomAdjuster := oom.NewOOMAdjuster()
		if err := oomAdjuster.ApplyOOMScoreAdj(0, int(*s.OOMScoreAdj)); err != nil {
			glog.V(2).Info(err)
		}
	}

	if len(s.ResourceContainer) > 0 {
		// Run in its own container.
		if err := resourcecontainer.RunInResourceContainer(s.ResourceContainer); err != nil {
			glog.Warningf("Failed to start in resource-only container %q: %v", s.ResourceContainer, err)
		} else {
			glog.V(2).Infof("Running in resource-only container %q", s.ResourceContainer)
		}
	}

	s.Broadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: s.Client.Core().Events("")})

	// Start up a webserver if requested
	if len(s.HealthzBindAddress) > 0 {
		http.HandleFunc("/proxyMode", func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, "%s", s.ProxyMode)
		})
		configz.InstallHandler(http.DefaultServeMux)
		go wait.Until(func() {
			err := http.ListenAndServe(s.HealthzBindAddress, nil)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("starting health server failed: %v", err))
			}
		}, 5*time.Second, wait.NeverStop)
	}

	// Tune conntrack, if requested
	if s.Conntracker != nil {
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
				const message = "DOCKER RESTART NEEDED (docker issue #24000): /sys is read-only: can't raise conntrack limits, problems may arise later."
				s.Recorder.Eventf(s.NodeRef, api.EventTypeWarning, err.Error(), message)
			}
		}
		if s.ConntrackConfiguration.TCPEstablishedTimeout.Duration > 0 {
			if err := s.Conntracker.SetTCPEstablishedTimeout(int(s.ConntrackConfiguration.TCPEstablishedTimeout.Duration / time.Second)); err != nil {
				return err
			}
		}
	}

	// Create configs (i.e. Watches for Services and Endpoints)
	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.
	serviceConfig := proxyconfig.NewServiceConfig()
	serviceConfig.RegisterHandler(s.ServiceConfigHandler)

	endpointsConfig := proxyconfig.NewEndpointsConfig()
	endpointsConfig.RegisterHandler(s.EndpointsConfigHandler)

	proxyconfig.NewSourceAPI(
		s.Client.Core().RESTClient(),
		s.ConfigSyncPeriod,
		serviceConfig.Channel("api"),
		endpointsConfig.Channel("api"),
	)

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
		glog.V(3).Infof("getConntrackMax: using absolute conntrax-max (deprecated)")
		return int(config.Max), nil
	}
	if config.MaxPerCore > 0 {
		floor := int(config.Min)
		scaled := int(config.MaxPerCore) * runtime.NumCPU()
		if scaled > floor {
			glog.V(3).Infof("getConntrackMax: using scaled conntrax-max-per-core")
			return scaled, nil
		}
		glog.V(3).Infof("getConntrackMax: using conntrax-min")
		return floor, nil
	}
	return 0, nil
}

type nodeGetter interface {
	Get(hostname string) (*api.Node, error)
}

func getProxyMode(proxyMode string, client nodeGetter, hostname string, iptver iptables.IPTablesVersioner, kcompat iptables.KernelCompatTester) string {
	if proxyMode == proxyModeUserspace {
		return proxyModeUserspace
	} else if proxyMode == proxyModeIPTables {
		return tryIPTablesProxy(iptver, kcompat)
	} else if len(proxyMode) > 0 {
		glog.Warningf("proxy-mode=%q unknown, assuming iptables proxy", proxyMode)
		return tryIPTablesProxy(iptver, kcompat)
	}
	// proxyMode == "" - choose the best option.
	if client == nil {
		utilruntime.HandleError(errors.New("nodeGetter is nil: assuming iptables proxy"))
		return tryIPTablesProxy(iptver, kcompat)
	}
	node, err := client.Get(hostname)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("can't get Node %q, assuming iptables proxy, err: %v", hostname, err))
		return tryIPTablesProxy(iptver, kcompat)
	}
	if node == nil {
		utilruntime.HandleError(fmt.Errorf("got nil Node %q, assuming iptables proxy", hostname))
		return tryIPTablesProxy(iptver, kcompat)
	}
	proxyMode, found := node.Annotations[betaProxyModeAnnotation]
	if found {
		glog.V(1).Infof("Found beta annotation %q = %q", betaProxyModeAnnotation, proxyMode)
	} else {
		// We already published some information about this annotation with the "experimental" name, so we will respect it.
		proxyMode, found = node.Annotations[ExperimentalProxyModeAnnotation]
		if found {
			glog.V(1).Infof("Found experimental annotation %q = %q", ExperimentalProxyModeAnnotation, proxyMode)
		}
	}
	if proxyMode == proxyModeUserspace {
		glog.V(1).Infof("Annotation demands userspace proxy")
		return proxyModeUserspace
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
	node, err := client.Core().Nodes().Get(hostname)
	if err != nil {
		glog.Warningf("Failed to retrieve node info: %v", err)
		return nil
	}
	nodeIP, err = nodeutil.GetNodeHostIP(node)
	if err != nil {
		glog.Warningf("Failed to retrieve node IP: %v", err)
		return nil
	}
	return nodeIP
}
