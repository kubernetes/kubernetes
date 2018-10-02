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
	"os"
	goruntime "runtime"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	apimachineryconfig "k8s.io/apimachinery/pkg/apis/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/apiserver/pkg/server/routes"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/flag"
	informers "k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/tools/record"
	"k8s.io/kube-proxy/config/v1alpha1"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/proxy"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	"k8s.io/kubernetes/pkg/proxy/apis/config/validation"
	"k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
	"k8s.io/kubernetes/pkg/proxy/userspace"
	"k8s.io/kubernetes/pkg/util/configz"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/resourcecontainer"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/version/verflag"
	"k8s.io/utils/exec"
	utilpointer "k8s.io/utils/pointer"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

const (
	proxyModeUserspace   = "userspace"
	proxyModeIPTables    = "iptables"
	proxyModeIPVS        = "ipvs"
	proxyModeKernelspace = "kernelspace"
)

// Options contains everything necessary to create and run a proxy server.
type Options struct {
	// ConfigFile is the location of the proxy server's configuration file.
	ConfigFile string
	// WriteConfigTo is the path where the default configuration will be written.
	WriteConfigTo string
	// CleanupAndExit, when true, makes the proxy server clean up iptables rules, then exit.
	CleanupAndExit bool
	// CleanupIPVS, when true, makes the proxy server clean up ipvs rules before running.
	CleanupIPVS bool
	// WindowsService should be set to true if kube-proxy is running as a service on Windows.
	// Its corresponding flag only gets registered in Windows builds
	WindowsService bool
	// config is the proxy server's configuration object.
	config *kubeproxyconfig.KubeProxyConfiguration

	// The fields below here are placeholders for flags that can't be directly mapped into
	// config.KubeProxyConfiguration.
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
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	o.addOSFlags(fs)
	fs.StringVar(&o.ConfigFile, "config", o.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&o.WriteConfigTo, "write-config-to", o.WriteConfigTo, "If set, write the default configuration values to this file and exit.")
	fs.BoolVar(&o.CleanupAndExit, "cleanup-iptables", o.CleanupAndExit, "If true cleanup iptables and ipvs rules and exit.")
	fs.MarkDeprecated("cleanup-iptables", "This flag is replaced by --cleanup.")
	fs.BoolVar(&o.CleanupAndExit, "cleanup", o.CleanupAndExit, "If true cleanup iptables and ipvs rules and exit.")
	fs.BoolVar(&o.CleanupIPVS, "cleanup-ipvs", o.CleanupIPVS, "If true make kube-proxy cleanup ipvs rules before running.  Default is true")

	// All flags below here are deprecated and will eventually be removed.

	fs.Var(utilflag.IPVar{Val: &o.config.BindAddress}, "bind-address", "The IP address for the proxy server to serve on (set to `0.0.0.0` for all IPv4 interfaces and `::` for all IPv6 interfaces)")
	fs.StringVar(&o.master, "master", o.master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.Int32Var(&o.healthzPort, "healthz-port", o.healthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.Var(utilflag.IPVar{Val: &o.config.HealthzBindAddress}, "healthz-bind-address", "The IP address and port for the health check server to serve on (set to `0.0.0.0` for all IPv4 interfaces and `::` for all IPv6 interfaces)")
	fs.Var(utilflag.IPVar{Val: &o.config.MetricsBindAddress}, "metrics-bind-address", "The IP address and port for the metrics server to serve on (set to `0.0.0.0` for all IPv4 interfaces and `::` for all IPv6 interfaces)")
	fs.Int32Var(o.config.OOMScoreAdj, "oom-score-adj", utilpointer.Int32PtrDerefOr(o.config.OOMScoreAdj, int32(qos.KubeProxyOOMScoreAdj)), "The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
	fs.StringVar(&o.config.ResourceContainer, "resource-container", o.config.ResourceContainer, "Absolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).")
	fs.MarkDeprecated("resource-container", "This feature will be removed in a later release.")
	fs.StringVar(&o.config.ClientConnection.Kubeconfig, "kubeconfig", o.config.ClientConnection.Kubeconfig, "Path to kubeconfig file with authorization information (the master location is set by the master flag).")
	fs.Var(utilflag.PortRangeVar{Val: &o.config.PortRange}, "proxy-port-range", "Range of host ports (beginPort-endPort, single port or beginPort+offset, inclusive) that may be consumed in order to proxy service traffic. If (unspecified, 0, or 0-0) then ports will be randomly chosen.")
	fs.StringVar(&o.config.HostnameOverride, "hostname-override", o.config.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.Var(&o.config.Mode, "proxy-mode", "Which proxy mode to use: 'userspace' (older) or 'iptables' (faster) or 'ipvs' (experimental). If blank, use the best-available proxy (currently iptables).  If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are insufficient, this always falls back to the userspace proxy.")
	fs.Int32Var(o.config.IPTables.MasqueradeBit, "iptables-masquerade-bit", utilpointer.Int32PtrDerefOr(o.config.IPTables.MasqueradeBit, 14), "If using the pure iptables proxy, the bit of the fwmark space to mark packets requiring SNAT with.  Must be within the range [0, 31].")
	fs.DurationVar(&o.config.IPTables.SyncPeriod.Duration, "iptables-sync-period", o.config.IPTables.SyncPeriod.Duration, "The maximum interval of how often iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.DurationVar(&o.config.IPTables.MinSyncPeriod.Duration, "iptables-min-sync-period", o.config.IPTables.MinSyncPeriod.Duration, "The minimum interval of how often the iptables rules can be refreshed as endpoints and services change (e.g. '5s', '1m', '2h22m').")
	fs.DurationVar(&o.config.IPVS.SyncPeriod.Duration, "ipvs-sync-period", o.config.IPVS.SyncPeriod.Duration, "The maximum interval of how often ipvs rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.DurationVar(&o.config.IPVS.MinSyncPeriod.Duration, "ipvs-min-sync-period", o.config.IPVS.MinSyncPeriod.Duration, "The minimum interval of how often the ipvs rules can be refreshed as endpoints and services change (e.g. '5s', '1m', '2h22m').")
	fs.StringSliceVar(&o.config.IPVS.ExcludeCIDRs, "ipvs-exclude-cidrs", o.config.IPVS.ExcludeCIDRs, "A comma-separated list of CIDR's which the ipvs proxier should not touch when cleaning up IPVS rules.")
	fs.DurationVar(&o.config.ConfigSyncPeriod.Duration, "config-sync-period", o.config.ConfigSyncPeriod.Duration, "How often configuration from the apiserver is refreshed.  Must be greater than 0.")
	fs.BoolVar(&o.config.IPTables.MasqueradeAll, "masquerade-all", o.config.IPTables.MasqueradeAll, "If using the pure iptables proxy, SNAT all traffic sent via Service cluster IPs (this not commonly needed)")
	fs.StringVar(&o.config.ClusterCIDR, "cluster-cidr", o.config.ClusterCIDR, "The CIDR range of pods in the cluster. When configured, traffic sent to a Service cluster IP from outside this range will be masqueraded and traffic sent from pods to an external LoadBalancer IP will be directed to the respective cluster IP instead")
	fs.StringVar(&o.config.ClientConnection.ContentType, "kube-api-content-type", o.config.ClientConnection.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&o.config.ClientConnection.QPS, "kube-api-qps", o.config.ClientConnection.QPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&o.config.ClientConnection.Burst, "kube-api-burst", o.config.ClientConnection.Burst, "Burst to use while talking with kubernetes apiserver")
	fs.DurationVar(&o.config.UDPIdleTimeout.Duration, "udp-timeout", o.config.UDPIdleTimeout.Duration, "How long an idle UDP connection will be kept open (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxy-mode=userspace")
	if o.config.Conntrack.Max == nil {
		o.config.Conntrack.Max = utilpointer.Int32Ptr(0)
	}
	fs.Int32Var(o.config.Conntrack.Max, "conntrack-max", *o.config.Conntrack.Max,
		"Maximum number of NAT connections to track (0 to leave as-is). This overrides conntrack-max-per-core and conntrack-min.")
	fs.MarkDeprecated("conntrack-max", "This feature will be removed in a later release.")
	fs.Int32Var(o.config.Conntrack.MaxPerCore, "conntrack-max-per-core", *o.config.Conntrack.MaxPerCore,
		"Maximum number of NAT connections to track per CPU core (0 to leave the limit as-is and ignore conntrack-min).")
	fs.Int32Var(o.config.Conntrack.Min, "conntrack-min", *o.config.Conntrack.Min,
		"Minimum number of conntrack entries to allocate, regardless of conntrack-max-per-core (set conntrack-max-per-core=0 to leave the limit as-is).")
	fs.DurationVar(&o.config.Conntrack.TCPEstablishedTimeout.Duration, "conntrack-tcp-timeout-established", o.config.Conntrack.TCPEstablishedTimeout.Duration, "Idle timeout for established TCP connections (0 to leave as-is)")
	fs.DurationVar(
		&o.config.Conntrack.TCPCloseWaitTimeout.Duration, "conntrack-tcp-timeout-close-wait",
		o.config.Conntrack.TCPCloseWaitTimeout.Duration,
		"NAT timeout for TCP connections in the CLOSE_WAIT state")
	fs.BoolVar(&o.config.EnableProfiling, "profiling", o.config.EnableProfiling, "If true enables profiling via web interface on /debug/pprof handler.")
	fs.StringVar(&o.config.IPVS.Scheduler, "ipvs-scheduler", o.config.IPVS.Scheduler, "The ipvs scheduler type when proxy mode is ipvs")
	fs.StringSliceVar(&o.config.NodePortAddresses, "nodeport-addresses", o.config.NodePortAddresses,
		"A string slice of values which specify the addresses to use for NodePorts. Values may be valid IP blocks (e.g. 1.2.3.0/24, 1.2.3.4/32). The default empty string slice ([]) means to use all local addresses.")
	fs.Var(flag.NewMapStringBool(&o.config.FeatureGates), "feature-gates", "A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(utilfeature.DefaultFeatureGate.KnownFeatures(), "\n"))
}

func NewOptions() *Options {
	return &Options{
		config:      new(kubeproxyconfig.KubeProxyConfiguration),
		healthzPort: ports.ProxyHealthzPort,
		scheme:      scheme.Scheme,
		codecs:      scheme.Codecs,
		CleanupIPVS: true,
	}
}

// Complete completes all the required options.
func (o *Options) Complete() error {
	if len(o.ConfigFile) == 0 && len(o.WriteConfigTo) == 0 {
		glog.Warning("WARNING: all flags other than --config, --write-config-to, and --cleanup are deprecated. Please begin using a config file ASAP.")
		o.applyDeprecatedHealthzPortToConfig()
	}

	// Load the config file here in Complete, so that Validate validates the fully-resolved config.
	if len(o.ConfigFile) > 0 {
		if c, err := o.loadConfigFromFile(o.ConfigFile); err != nil {
			return err
		} else {
			o.config = c
		}
	}

	err := utilfeature.DefaultFeatureGate.SetFromMap(o.config.FeatureGates)
	if err != nil {
		return err
	}

	return nil
}

// Validate validates all the required options.
func (o *Options) Validate(args []string) error {
	if len(args) != 0 {
		return errors.New("no arguments are supported")
	}

	if errs := validation.Validate(o.config); len(errs) != 0 {
		return errs.ToAggregate()
	}

	return nil
}

func (o *Options) Run() error {
	if len(o.WriteConfigTo) > 0 {
		return o.writeConfigFile()
	}

	proxyServer, err := NewProxyServer(o)
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

	glog.Infof("Wrote configuration to: %s\n", o.WriteConfigTo)

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
func (o *Options) loadConfigFromFile(file string) (*kubeproxyconfig.KubeProxyConfiguration, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return o.loadConfig(data)
}

// loadConfig decodes data as a KubeProxyConfiguration object.
func (o *Options) loadConfig(data []byte) (*kubeproxyconfig.KubeProxyConfiguration, error) {
	configObj, gvk, err := o.codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	config, ok := configObj.(*kubeproxyconfig.KubeProxyConfiguration)
	if !ok {
		return nil, fmt.Errorf("got unexpected config type: %v", gvk)
	}
	return config, nil
}

func (o *Options) ApplyDefaults(in *kubeproxyconfig.KubeProxyConfiguration) (*kubeproxyconfig.KubeProxyConfiguration, error) {
	external, err := o.scheme.ConvertToVersion(in, v1alpha1.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	o.scheme.Default(external)

	internal, err := o.scheme.ConvertToVersion(external, kubeproxyconfig.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	out := internal.(*kubeproxyconfig.KubeProxyConfiguration)

	return out, nil
}

// NewProxyCommand creates a *cobra.Command object with default parameters
func NewProxyCommand() *cobra.Command {
	opts := NewOptions()

	cmd := &cobra.Command{
		Use: "kube-proxy",
		Long: `The Kubernetes network proxy runs on each node. This
reflects services as defined in the Kubernetes API on each node and can do simple
TCP, UDP, and SCTP stream forwarding or round robin TCP, UDP, and SCTP forwarding across a set of backends.
Service cluster IPs and ports are currently found through Docker-links-compatible
environment variables specifying ports opened by the service proxy. There is an optional
addon that provides cluster DNS for these cluster IPs. The user must create a service
with the apiserver API to configure the proxy.`,
		Run: func(cmd *cobra.Command, args []string) {
			verflag.PrintAndExitIfRequested()
			utilflag.PrintFlags(cmd.Flags())

			if err := initForOS(opts.WindowsService); err != nil {
				glog.Fatalf("failed OS init: %v", err)
			}

			if err := opts.Complete(); err != nil {
				glog.Fatalf("failed complete: %v", err)
			}
			if err := opts.Validate(args); err != nil {
				glog.Fatalf("failed validate: %v", err)
			}
			glog.Fatal(opts.Run())
		},
	}

	var err error
	opts.config, err = opts.ApplyDefaults(opts.config)
	if err != nil {
		glog.Fatalf("unable to create flag defaults: %v", err)
	}

	opts.AddFlags(cmd.Flags())

	cmd.MarkFlagFilename("config", "yaml", "yml", "json")

	return cmd
}

// ProxyServer represents all the parameters required to start the Kubernetes proxy server. All
// fields are required.
type ProxyServer struct {
	Client                 clientset.Interface
	EventClient            v1core.EventsGetter
	IptInterface           utiliptables.Interface
	IpvsInterface          utilipvs.Interface
	IpsetInterface         utilipset.Interface
	execer                 exec.Interface
	Proxier                proxy.ProxyProvider
	Broadcaster            record.EventBroadcaster
	Recorder               record.EventRecorder
	ConntrackConfiguration kubeproxyconfig.KubeProxyConntrackConfiguration
	Conntracker            Conntracker // if nil, ignored
	ProxyMode              string
	NodeRef                *v1.ObjectReference
	CleanupAndExit         bool
	CleanupIPVS            bool
	MetricsBindAddress     string
	EnableProfiling        bool
	OOMScoreAdj            *int32
	ResourceContainer      string
	ConfigSyncPeriod       time.Duration
	ServiceEventHandler    config.ServiceHandler
	EndpointsEventHandler  config.EndpointsHandler
	HealthzServer          *healthcheck.HealthzServer
}

// createClients creates a kube client and an event client from the given config and masterOverride.
// TODO remove masterOverride when CLI flags are removed.
func createClients(config apimachineryconfig.ClientConnectionConfiguration, masterOverride string) (clientset.Interface, v1core.EventsGetter, error) {
	var kubeConfig *rest.Config
	var err error

	if len(config.Kubeconfig) == 0 && len(masterOverride) == 0 {
		glog.Info("Neither kubeconfig file nor master URL was specified. Falling back to in-cluster config.")
		kubeConfig, err = rest.InClusterConfig()
	} else {
		// This creates a client, first loading any specified kubeconfig
		// file, and then overriding the Master flag, if non-empty.
		kubeConfig, err = clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
			&clientcmd.ClientConfigLoadingRules{ExplicitPath: config.Kubeconfig},
			&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: masterOverride}}).ClientConfig()
	}
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

	eventClient, err := clientset.NewForConfig(kubeConfig)
	if err != nil {
		return nil, nil, err
	}

	return client, eventClient.CoreV1(), nil
}

// Run runs the specified ProxyServer.  This should never exit (unless CleanupAndExit is set).
func (s *ProxyServer) Run() error {
	// To help debugging, immediately log version
	glog.Infof("Version: %+v", version.Get())
	// remove iptables rules and exit
	if s.CleanupAndExit {
		encounteredError := userspace.CleanupLeftovers(s.IptInterface)
		encounteredError = iptables.CleanupLeftovers(s.IptInterface) || encounteredError
		encounteredError = ipvs.CleanupLeftovers(s.IpvsInterface, s.IptInterface, s.IpsetInterface, s.CleanupIPVS) || encounteredError
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
		mux := mux.NewPathRecorderMux("kube-proxy")
		healthz.InstallHandler(mux)
		mux.HandleFunc("/proxyMode", func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, "%s", s.ProxyMode)
		})
		mux.Handle("/metrics", prometheus.Handler())
		if s.EnableProfiling {
			routes.Profiling{}.Install(mux)
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
	// Conntracker is always nil for windows
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
				const message = "DOCKER RESTART NEEDED (docker issue #24000): /sys is read-only: " +
					"cannot modify conntrack limits, problems may arise later."
				s.Recorder.Eventf(s.NodeRef, api.EventTypeWarning, err.Error(), message)
			}
		}

		if s.ConntrackConfiguration.TCPEstablishedTimeout != nil && s.ConntrackConfiguration.TCPEstablishedTimeout.Duration > 0 {
			timeout := int(s.ConntrackConfiguration.TCPEstablishedTimeout.Duration / time.Second)
			if err := s.Conntracker.SetTCPEstablishedTimeout(timeout); err != nil {
				return err
			}
		}

		if s.ConntrackConfiguration.TCPCloseWaitTimeout != nil && s.ConntrackConfiguration.TCPCloseWaitTimeout.Duration > 0 {
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
	serviceConfig := config.NewServiceConfig(informerFactory.Core().V1().Services(), s.ConfigSyncPeriod)
	serviceConfig.RegisterEventHandler(s.ServiceEventHandler)
	go serviceConfig.Run(wait.NeverStop)

	endpointsConfig := config.NewEndpointsConfig(informerFactory.Core().V1().Endpoints(), s.ConfigSyncPeriod)
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

func (s *ProxyServer) birthCry() {
	s.Recorder.Eventf(s.NodeRef, api.EventTypeNormal, "Starting", "Starting kube-proxy.")
}

func getConntrackMax(config kubeproxyconfig.KubeProxyConntrackConfiguration) (int, error) {
	if config.Max != nil && *config.Max > 0 {
		if config.MaxPerCore != nil && *config.MaxPerCore > 0 {
			return -1, fmt.Errorf("invalid config: Conntrack Max and Conntrack MaxPerCore are mutually exclusive")
		}
		glog.V(3).Infof("getConntrackMax: using absolute conntrack-max (deprecated)")
		return int(*config.Max), nil
	}
	if config.MaxPerCore != nil && *config.MaxPerCore > 0 {
		floor := 0
		if config.Min != nil {
			floor = int(*config.Min)
		}
		scaled := int(*config.MaxPerCore) * goruntime.NumCPU()
		if scaled > floor {
			glog.V(3).Infof("getConntrackMax: using scaled conntrack-max-per-core")
			return scaled, nil
		}
		glog.V(3).Infof("getConntrackMax: using conntrack-min")
		return floor, nil
	}
	return 0, nil
}

func getNodeIP(client clientset.Interface, hostname string) net.IP {
	var nodeIP net.IP
	node, err := client.CoreV1().Nodes().Get(hostname, metav1.GetOptions{})
	if err != nil {
		glog.Warningf("Failed to retrieve node info: %v", err)
		return nil
	}
	nodeIP, err = utilnode.GetNodeHostIP(node)
	if err != nil {
		glog.Warningf("Failed to retrieve node IP: %v", err)
		return nil
	}
	return nodeIP
}
