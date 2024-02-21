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
	"context"
	goflag "flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/spf13/cobra"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/apiserver/pkg/server/routes"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/tools/events"
	cliflag "k8s.io/component-base/cli/flag"
	componentbaseconfig "k8s.io/component-base/config"
	"k8s.io/component-base/configz"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/component-base/metrics"
	metricsfeatures "k8s.io/component-base/metrics/features"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/prometheus/slis"
	"k8s.io/component-base/version"
	"k8s.io/component-base/version/verflag"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/apis"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyconfigscheme "k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	"k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	netutils "k8s.io/utils/net"
)

func init() {
	utilruntime.Must(metricsfeatures.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
	utilruntime.Must(logsapi.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
}

// proxyRun defines the interface to run a specified ProxyServer
type proxyRun interface {
	Run() error
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
		RunE: func(cmd *cobra.Command, args []string) error {
			verflag.PrintAndExitIfRequested()

			if err := initForOS(opts.config.Windows.RunAsService); err != nil {
				return fmt.Errorf("failed os init: %w", err)
			}

			if err := opts.Complete(cmd.Flags()); err != nil {
				return fmt.Errorf("failed complete: %w", err)
			}

			logs.InitLogs()
			if err := logsapi.ValidateAndApplyAsField(&opts.config.Logging, utilfeature.DefaultFeatureGate, field.NewPath("logging")); err != nil {
				return fmt.Errorf("initialize logging: %v", err)
			}

			cliflag.PrintFlags(cmd.Flags())

			if err := opts.Validate(); err != nil {
				return fmt.Errorf("failed validate: %w", err)
			}
			// add feature enablement metrics
			utilfeature.DefaultMutableFeatureGate.AddMetrics()
			if err := opts.Run(); err != nil {
				opts.logger.Error(err, "Error running ProxyServer")
				return err
			}

			return nil
		},
		Args: func(cmd *cobra.Command, args []string) error {
			for _, arg := range args {
				if len(arg) > 0 {
					return fmt.Errorf("%q does not take any arguments, got %q", cmd.CommandPath(), args)
				}
			}
			return nil
		},
	}

	fs := cmd.Flags()
	opts.AddFlags(fs)
	fs.AddGoFlagSet(goflag.CommandLine) // for --boot-id-file and --machine-id-file

	_ = cmd.MarkFlagFilename("config", "yaml", "yml", "json")

	return cmd
}

// ProxyServer represents all the parameters required to start the Kubernetes proxy server. All
// fields are required.
type ProxyServer struct {
	Config *kubeproxyconfig.KubeProxyConfiguration

	Client          clientset.Interface
	Broadcaster     events.EventBroadcaster
	Recorder        events.EventRecorder
	NodeRef         *v1.ObjectReference
	HealthzServer   *healthcheck.ProxierHealthServer
	Hostname        string
	PrimaryIPFamily v1.IPFamily
	NodeIPs         map[v1.IPFamily]net.IP

	podCIDRs []string // only used for LocalModeNodeCIDR

	Proxier proxy.Provider

	logger klog.Logger
}

// newProxyServer creates a ProxyServer based on the given config
func newProxyServer(logger klog.Logger, config *kubeproxyconfig.KubeProxyConfiguration, master string, initOnly bool) (*ProxyServer, error) {
	s := &ProxyServer{
		Config: config,
		logger: logger,
	}

	cz, err := configz.New(kubeproxyconfig.GroupName)
	if err != nil {
		return nil, fmt.Errorf("unable to register configz: %s", err)
	}
	cz.Set(config)

	if len(config.ShowHiddenMetricsForVersion) > 0 {
		metrics.SetShowHidden()
	}

	s.Hostname, err = nodeutil.GetHostname(config.HostnameOverride)
	if err != nil {
		return nil, err
	}

	s.Client, err = createClient(logger, config.ClientConnection, master)
	if err != nil {
		return nil, err
	}

	rawNodeIPs := getNodeIPs(logger, s.Client, s.Hostname)
	s.PrimaryIPFamily, s.NodeIPs = detectNodeIPs(logger, rawNodeIPs, config.NodeIPOverride)

	s.Broadcaster = events.NewBroadcaster(&events.EventSinkImpl{Interface: s.Client.EventsV1()})
	s.Recorder = s.Broadcaster.NewRecorder(proxyconfigscheme.Scheme, "kube-proxy")

	s.NodeRef = &v1.ObjectReference{
		Kind:      "Node",
		Name:      s.Hostname,
		UID:       types.UID(s.Hostname),
		Namespace: "",
	}

	if len(config.HealthzBindAddresses) > 0 {
		s.HealthzServer = healthcheck.NewProxierHealthServer(config.HealthzBindAddresses, config.HealthzBindPort, 2*config.SyncPeriod.Duration)
	}

	err = s.platformSetup()
	if err != nil {
		return nil, err
	}

	ipv4Supported, ipv6Supported, dualStackSupported, err := s.platformCheckSupported()
	if err != nil {
		return nil, err
	} else if (s.PrimaryIPFamily == v1.IPv4Protocol && !ipv4Supported) || (s.PrimaryIPFamily == v1.IPv6Protocol && !ipv6Supported) {
		return nil, fmt.Errorf("no support for primary IP family %q", s.PrimaryIPFamily)
	} else if dualStackSupported {
		logger.Info("kube-proxy running in dual-stack mode", "primary ipFamily", s.PrimaryIPFamily)
	} else {
		logger.Info("kube-proxy running in single-stack mode", "ipFamily", s.PrimaryIPFamily)
	}

	err, fatal := checkIPConfig(s, dualStackSupported)
	if err != nil {
		if fatal || (config.ConfigHardFail != nil && *config.ConfigHardFail) {
			return nil, fmt.Errorf("kube-proxy configuration is incorrect: %v", err)
		}
		logger.Error(err, "Kube-proxy configuration may be incomplete or incorrect")
	}

	s.Proxier, err = s.createProxier(config, dualStackSupported, initOnly)
	if err != nil {
		return nil, err
	}

	return s, nil
}

// checkIPConfig confirms that s has proper configuration for its primary IP family.
func checkIPConfig(s *ProxyServer, dualStackSupported bool) (error, bool) {
	var errors []error
	var badFamily netutils.IPFamily

	if s.PrimaryIPFamily == v1.IPv4Protocol {
		badFamily = netutils.IPv6
	} else {
		badFamily = netutils.IPv4
	}

	var clusterType string
	if dualStackSupported {
		clusterType = fmt.Sprintf("%s-primary", s.PrimaryIPFamily)
	} else {
		clusterType = fmt.Sprintf("%s-only", s.PrimaryIPFamily)
	}

	// Historically, we did not check most of the config options, so we cannot
	// retroactively make IP family mismatches in those options be fatal. When we add
	// new options to check here, we should make problems with those options be fatal.
	fatal := false

	if len(s.Config.DetectLocal.ClusterCIDRs) > 0 {
		if badCIDRs(s.Config.DetectLocal.ClusterCIDRs, badFamily) {
			errors = append(errors, fmt.Errorf("cluster is %s but clusterCIDRs contains only IPv%s addresses", clusterType, badFamily))
			if s.Config.DetectLocalMode == kubeproxyconfig.LocalModeClusterCIDR && !dualStackSupported {
				// This has always been a fatal error
				fatal = true
			}
		}
	}

	if badCIDRs(s.Config.NodePortAddresses, badFamily) {
		errors = append(errors, fmt.Errorf("cluster is %s but nodePortAddresses contains only IPv%s addresses", clusterType, badFamily))
	}

	if badCIDRs(s.podCIDRs, badFamily) {
		errors = append(errors, fmt.Errorf("cluster is %s but node.spec.podCIDRs contains only IPv%s addresses", clusterType, badFamily))
		if s.Config.DetectLocalMode == kubeproxyconfig.LocalModeNodeCIDR {
			// This has always been a fatal error
			fatal = true
		}
	}

	if netutils.IPFamilyOfString(s.Config.Winkernel.SourceVip) == badFamily {
		errors = append(errors, fmt.Errorf("cluster is %s but winkernel.sourceVip is IPv%s", clusterType, badFamily))
	}

	// In some cases, wrong-IP-family is only a problem when the secondary IP family
	// isn't present at all.
	if !dualStackSupported {
		if badCIDRs(s.Config.IPVS.ExcludeCIDRs, badFamily) {
			errors = append(errors, fmt.Errorf("cluster is %s but ipvs.excludeCIDRs contains only IPv%s addresses", clusterType, badFamily))
		}

		if badBindAddresses(s.Config.HealthzBindAddresses, badFamily) {
			errors = append(errors, fmt.Errorf("cluster is %s but healthzBindAddresses contains only IPv%s addresses", clusterType, badFamily))
		}
		if badBindAddresses(s.Config.MetricsBindAddresses, badFamily) {
			errors = append(errors, fmt.Errorf("cluster is %s but metricsBindAddresses contains only IPv%s addresses", clusterType, badFamily))
		}
	}

	return utilerrors.NewAggregate(errors), fatal
}

// badCIDRs returns true if cidrs is a non-empty list of CIDRs, all of wrongFamily.
func badCIDRs(cidrs []string, wrongFamily netutils.IPFamily) bool {
	if len(cidrs) == 0 {
		return false
	}
	for _, cidr := range cidrs {
		if netutils.IPFamilyOfCIDRString(cidr) != wrongFamily {
			return false
		}
	}
	return true
}

// badBindAddresses returns true if cidrs is a non-empty list of CIDRs, all of wrongFamily.
// Unspecified addresses '0.0.0.0' and '::' are not treated as part of either family.
func badBindAddresses(addresses []string, wrongFamily netutils.IPFamily) bool {
	if len(addresses) == 0 {
		return false
	}
	for _, address := range addresses {
		ip, _, _ := netutils.ParseCIDRSloppy(address)
		if netutils.IPFamilyOf(ip) != wrongFamily || ip.IsUnspecified() {
			return false
		}
	}
	return true
}

// createClient creates a kube client from the given config and masterOverride.
// TODO remove masterOverride when CLI flags are removed.
func createClient(logger klog.Logger, config componentbaseconfig.ClientConnectionConfiguration, masterOverride string) (clientset.Interface, error) {
	var kubeConfig *rest.Config
	var err error

	if len(config.Kubeconfig) == 0 && len(masterOverride) == 0 {
		logger.Info("Neither kubeconfig file nor master URL was specified, falling back to in-cluster config")
		kubeConfig, err = rest.InClusterConfig()
	} else {
		// This creates a client, first loading any specified kubeconfig
		// file, and then overriding the Master flag, if non-empty.
		kubeConfig, err = clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
			&clientcmd.ClientConfigLoadingRules{ExplicitPath: config.Kubeconfig},
			&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: masterOverride}}).ClientConfig()
	}
	if err != nil {
		return nil, err
	}

	kubeConfig.AcceptContentTypes = config.AcceptContentTypes
	kubeConfig.ContentType = config.ContentType
	kubeConfig.QPS = config.QPS
	kubeConfig.Burst = int(config.Burst)

	client, err := clientset.NewForConfig(kubeConfig)
	if err != nil {
		return nil, err
	}

	return client, nil
}

func serveMetrics(logger klog.Logger, cidrStrings []string, port int32, proxyMode kubeproxyconfig.ProxyMode, enableProfiling bool, bindHardFail bool) chan error {
	proxyMux := mux.NewPathRecorderMux("kube-proxy")
	healthz.InstallHandler(proxyMux)
	slis.SLIMetricsWithReset{}.Install(proxyMux)

	proxyMux.HandleFunc("/proxyMode", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		fmt.Fprintf(w, "%s", proxyMode)
	})

	proxyMux.Handle("/metrics", legacyregistry.Handler())

	if enableProfiling {
		routes.Profiling{}.Install(proxyMux)
		routes.DebugFlags{}.Install(proxyMux, "v", routes.StringFlagPutHandler(logs.GlogSetter))
	}

	configz.InstallHandler(proxyMux)

	var nodeIPs []net.IP
	for _, family := range []v1.IPFamily{v1.IPv4Protocol, v1.IPv6Protocol} {
		npa := proxyutil.NewNodePortAddresses(family, cidrStrings, nil)
		ips, err := npa.GetNodeIPs(proxyutil.RealNetwork{})
		if err != nil {
			logger.Error(err, "failed to get get node ips for metrics server")
		}
		nodeIPs = append(nodeIPs, ips...)
	}
	if len(nodeIPs) == 0 {
		logger.Info("failed to get any node ip matching metricsBindAddresses", "metricsBindAddresses", cidrStrings)
	}

	errCh := make(chan error)
	for _, nodeIP := range nodeIPs {
		stopCh := make(chan struct{})
		addr := net.JoinHostPort(nodeIP.String(), strconv.Itoa(int(port)))
		fn := func() {
			err := http.ListenAndServe(addr, proxyMux)
			if err != nil {
				err = fmt.Errorf("starting metrics server failed: %w", err)
				utilruntime.HandleError(err)
				if err != nil && bindHardFail {
					stopCh <- struct{}{}
					errCh <- err
				}
			}
		}
		go wait.Until(fn, 5*time.Second, stopCh)
	}
	return errCh
}

// Run runs the specified ProxyServer. This should never exit (unless CleanupAndExit is set).
// TODO: At the moment, Run() cannot return a nil error, otherwise it's caller will never exit. Update callers of Run to handle nil errors.
func (s *ProxyServer) Run() error {
	// To help debugging, immediately log version
	s.logger.Info("Version info", "version", version.Get())

	s.logger.Info("Golang settings", "GOGC", os.Getenv("GOGC"), "GOMAXPROCS", os.Getenv("GOMAXPROCS"), "GOTRACEBACK", os.Getenv("GOTRACEBACK"))

	// TODO(vmarmol): Use container config for this.
	var oomAdjuster *oom.OOMAdjuster
	if s.Config.Linux.OOMScoreAdj != nil {
		oomAdjuster = oom.NewOOMAdjuster()
		if err := oomAdjuster.ApplyOOMScoreAdj(0, int(*s.Config.Linux.OOMScoreAdj)); err != nil {
			s.logger.V(2).Info("Failed to apply OOMScore", "err", err)
		}
	}

	if s.Broadcaster != nil {
		stopCh := make(chan struct{})
		s.Broadcaster.StartRecordingToSink(stopCh)
	}

	// TODO(thockin): make it possible for healthz and metrics to be on the same port.
	// Start up a healthz server if requested
	var healthzErrCh chan error
	if s.HealthzServer != nil {
		healthzErrCh = s.HealthzServer.Run(s.Config.BindAddressHardFail)
	}

	// Start up a metrics server if requested
	var metricsErrCh chan error
	if len(s.Config.MetricsBindAddresses) > 0 {
		metricsErrCh = serveMetrics(s.logger, s.Config.MetricsBindAddresses, s.Config.MetricsBindPort, s.Config.Mode, s.Config.EnableProfiling, s.Config.BindAddressHardFail)
	}

	noProxyName, err := labels.NewRequirement(apis.LabelServiceProxyName, selection.DoesNotExist, nil)
	if err != nil {
		return err
	}

	noHeadlessEndpoints, err := labels.NewRequirement(v1.IsHeadlessService, selection.DoesNotExist, nil)
	if err != nil {
		return err
	}

	labelSelector := labels.NewSelector()
	labelSelector = labelSelector.Add(*noProxyName, *noHeadlessEndpoints)

	// Make informers that filter out objects that want a non-default service proxy.
	informerFactory := informers.NewSharedInformerFactoryWithOptions(s.Client, s.Config.ConfigSyncPeriod.Duration,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.LabelSelector = labelSelector.String()
		}))

	// Create configs (i.e. Watches for Services, EndpointSlices and ServiceCIDRs)
	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.
	serviceConfig := config.NewServiceConfig(informerFactory.Core().V1().Services(), s.Config.ConfigSyncPeriod.Duration)
	serviceConfig.RegisterEventHandler(s.Proxier)
	go serviceConfig.Run(wait.NeverStop)

	endpointSliceConfig := config.NewEndpointSliceConfig(informerFactory.Discovery().V1().EndpointSlices(), s.Config.ConfigSyncPeriod.Duration)
	endpointSliceConfig.RegisterEventHandler(s.Proxier)
	go endpointSliceConfig.Run(wait.NeverStop)

	if utilfeature.DefaultFeatureGate.Enabled(features.MultiCIDRServiceAllocator) {
		serviceCIDRConfig := config.NewServiceCIDRConfig(informerFactory.Networking().V1alpha1().ServiceCIDRs(), s.Config.ConfigSyncPeriod.Duration)
		serviceCIDRConfig.RegisterEventHandler(s.Proxier)
		go serviceCIDRConfig.Run(wait.NeverStop)
	}
	// This has to start after the calls to NewServiceConfig because that
	// function must configure its shared informer event handlers first.
	informerFactory.Start(wait.NeverStop)

	// Make an informer that selects for our nodename.
	currentNodeInformerFactory := informers.NewSharedInformerFactoryWithOptions(s.Client, s.Config.ConfigSyncPeriod.Duration,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", s.NodeRef.Name).String()
		}))
	nodeConfig := config.NewNodeConfig(currentNodeInformerFactory.Core().V1().Nodes(), s.Config.ConfigSyncPeriod.Duration)
	// https://issues.k8s.io/111321
	if s.Config.DetectLocalMode == kubeproxyconfig.LocalModeNodeCIDR {
		nodeConfig.RegisterEventHandler(proxy.NewNodePodCIDRHandler(s.podCIDRs))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.KubeProxyDrainingTerminatingNodes) {
		nodeConfig.RegisterEventHandler(&proxy.NodeEligibleHandler{
			HealthServer: s.HealthzServer,
		})
	}
	nodeConfig.RegisterEventHandler(s.Proxier)

	go nodeConfig.Run(wait.NeverStop)

	// This has to start after the calls to NewNodeConfig because that must
	// configure the shared informer event handler first.
	currentNodeInformerFactory.Start(wait.NeverStop)

	// Birth Cry after the birth is successful
	s.birthCry()

	go s.Proxier.SyncLoop()

	select {
	case err = <-healthzErrCh:
		s.Recorder.Eventf(s.NodeRef, nil, api.EventTypeWarning, "FailedToStartProxierHealthcheck", "StartKubeProxy", err.Error())
	case err = <-metricsErrCh:
		s.Recorder.Eventf(s.NodeRef, nil, api.EventTypeWarning, "FailedToStartMetricServer", "StartKubeProxy", err.Error())
	}
	return err
}

func (s *ProxyServer) birthCry() {
	s.Recorder.Eventf(s.NodeRef, nil, api.EventTypeNormal, "Starting", "StartKubeProxy", "")
}

// detectNodeIPs returns the proxier's "node IP" or IPs, and the IP family to use if the
// node turns out to be incapable of dual-stack. (Note that kube-proxy normally runs as
// dual-stack if the backend is capable of supporting both IP families, regardless of
// whether the node is *actually* configured as dual-stack or not.)

// (Note that on Linux, the node IPs are used only to determine whether a given
// LoadBalancerSourceRanges value matches the node or not. In particular, they are *not*
// used for NodePort handling.)
//
// The order of precedence is:
//  1. if bindAddress is not 0.0.0.0 or ::, then it is used as the primary IP.
//  2. if rawNodeIPs is not empty, then its address(es) is/are used
//  3. otherwise the node IPs are 127.0.0.1 and ::1
func detectNodeIPs(logger klog.Logger, rawNodeIPs []net.IP, nodeIPOverride []string) (v1.IPFamily, map[v1.IPFamily]net.IP) {
	primaryFamily := v1.IPv4Protocol
	nodeIPs := map[v1.IPFamily]net.IP{
		v1.IPv4Protocol: net.IPv4(127, 0, 0, 1),
		v1.IPv6Protocol: net.IPv6loopback,
	}

	if len(rawNodeIPs) > 0 {
		if !netutils.IsIPv4(rawNodeIPs[0]) {
			primaryFamily = v1.IPv6Protocol
		}
		nodeIPs[primaryFamily] = rawNodeIPs[0]
		if len(rawNodeIPs) > 1 {
			// If more than one address is returned, they are guaranteed to be of different families
			family := v1.IPv4Protocol
			if !netutils.IsIPv4(rawNodeIPs[1]) {
				family = v1.IPv6Protocol
			}
			nodeIPs[family] = rawNodeIPs[1]
		}
	}

	// If a bindAddress is passed, override the primary IP
	for _, addr := range nodeIPOverride {
		nodeIP := netutils.ParseIPSloppy(addr)
		if nodeIP != nil && !nodeIP.IsUnspecified() {
			if netutils.IsIPv4(nodeIP) {
				primaryFamily = v1.IPv4Protocol
			} else {
				primaryFamily = v1.IPv6Protocol
			}
			nodeIPs[primaryFamily] = nodeIP
		}
	}

	if nodeIPs[primaryFamily].IsLoopback() {
		logger.Info("Can't determine this node's IP, assuming loopback; if this is incorrect, please set the --bind-address/--node-ip-override flag")
	}
	return primaryFamily, nodeIPs
}

// getNodeIP returns IPs for the node with the provided name.  If
// required, it will wait for the node to be created.
func getNodeIPs(logger klog.Logger, client clientset.Interface, name string) []net.IP {
	var nodeIPs []net.IP
	backoff := wait.Backoff{
		Steps:    6,
		Duration: 1 * time.Second,
		Factor:   2.0,
		Jitter:   0.2,
	}

	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		node, err := client.CoreV1().Nodes().Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			logger.Error(err, "Failed to retrieve node info")
			return false, nil
		}
		nodeIPs, err = utilnode.GetNodeHostIPs(node)
		if err != nil {
			logger.Error(err, "Failed to retrieve node IPs")
			return false, nil
		}
		return true, nil
	})
	if err == nil {
		logger.Info("Successfully retrieved node IP(s)", "IPs", nodeIPs)
	}
	return nodeIPs
}
