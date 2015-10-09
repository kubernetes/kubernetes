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

// Package app does all of the work necessary to configure and run a
// Kubernetes app process.
package app

import (
	"errors"
	"net"
	"net/http"
	_ "net/http/pprof"
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	kubeclient "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/userspace"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// ProxyServerConfig contains configures and runs a Kubernetes proxy server
type ProxyServerConfig struct {
	BindAddress        net.IP
	HealthzPort        int
	HealthzBindAddress net.IP
	OOMScoreAdj        int
	ResourceContainer  string
	Master             string
	Kubeconfig         string
	PortRange          util.PortRange
	HostnameOverride   string
	ProxyMode          string
	SyncPeriod         time.Duration
	nodeRef            *api.ObjectReference // Reference to this node.
	MasqueradeAll      bool
	CleanupAndExit     bool
	KubeApiQps         float32
	KubeApiBurst       int
}

type ProxyServer struct {
	Config       *ProxyServerConfig
	IptInterface utiliptables.Interface
	Proxier      proxy.ProxyProvider
	Recorder     record.EventRecorder
}

// AddFlags adds flags for a specific ProxyServer to the specified FlagSet
func (s *ProxyServerConfig) AddFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.BindAddress, "bind-address", s.BindAddress, "The IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.IntVar(&s.HealthzPort, "healthz-port", s.HealthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.IPVar(&s.HealthzBindAddress, "healthz-bind-address", s.HealthzBindAddress, "The IP address for the health check server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)")
	fs.IntVar(&s.OOMScoreAdj, "oom-score-adj", s.OOMScoreAdj, "The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
	fs.StringVar(&s.ResourceContainer, "resource-container", s.ResourceContainer, "Absolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization information (the master location is set by the master flag).")
	fs.Var(&s.PortRange, "proxy-port-range", "Range of host ports (beginPort-endPort, inclusive) that may be consumed in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.")
	fs.StringVar(&s.HostnameOverride, "hostname-override", s.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.StringVar(&s.ProxyMode, "proxy-mode", "", "Which proxy mode to use: 'userspace' (older, stable) or 'iptables' (experimental). If blank, look at the Node object on the Kubernetes API and respect the '"+experimentalProxyModeAnnotation+"' annotation if provided.  Otherwise use the best-available proxy (currently userspace, but may change in future versions).  If the iptables proxy is selected, regardless of how, but the system's kernel or iptables versions are insufficient, this always falls back to the userspace proxy.")
	fs.DurationVar(&s.SyncPeriod, "iptables-sync-period", s.SyncPeriod, "How often iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.BoolVar(&s.MasqueradeAll, "masquerade-all", false, "If using the pure iptables proxy, SNAT everything")
	fs.BoolVar(&s.CleanupAndExit, "cleanup-iptables", false, "If true cleanup iptables rules and exit.")
	fs.Float32Var(&s.KubeApiQps, "kube-api-qps", s.KubeApiQps, "QPS to use while talking with kubernetes apiserver")
	fs.IntVar(&s.KubeApiBurst, "kube-api-burst", s.KubeApiBurst, "Burst to use while talking with kubernetes apiserver")
}

const (
	proxyModeUserspace              = "userspace"
	proxyModeIptables               = "iptables"
	experimentalProxyModeAnnotation = "net.experimental.kubernetes.io/proxy-mode"
	betaProxyModeAnnotation         = "net.beta.kubernetes.io/proxy-mode"
)

func checkKnownProxyMode(proxyMode string) bool {
	switch proxyMode {
	case "", proxyModeUserspace, proxyModeIptables:
		return true
	}
	return false
}

func NewProxyConfig() *ProxyServerConfig {
	return &ProxyServerConfig{
		BindAddress:        net.ParseIP("0.0.0.0"),
		HealthzPort:        10249,
		HealthzBindAddress: net.ParseIP("127.0.0.1"),
		OOMScoreAdj:        qos.KubeProxyOOMScoreAdj,
		ResourceContainer:  "/kube-proxy",
		SyncPeriod:         30 * time.Second,
		KubeApiQps:         5.0,
		KubeApiBurst:       10,
	}
}

func NewProxyServer(
	config *ProxyServerConfig,
	iptInterface utiliptables.Interface,
	proxier proxy.ProxyProvider,
	recorder record.EventRecorder,
) (*ProxyServer, error) {
	return &ProxyServer{
		Config:       config,
		IptInterface: iptInterface,
		Proxier:      proxier,
		Recorder:     recorder,
	}, nil
}

// NewProxyServerDefault creates a new ProxyServer object with default parameters.
func NewProxyServerDefault(config *ProxyServerConfig) (*ProxyServer, error) {
	protocol := utiliptables.ProtocolIpv4
	if config.BindAddress.To4() == nil {
		protocol = utiliptables.ProtocolIpv6
	}

	// Create a iptables utils.
	execer := exec.New()
	dbus := utildbus.New()
	iptInterface := utiliptables.New(execer, dbus, protocol)

	// We ommit creation of pretty much everything if we run in cleanup mode
	if config.CleanupAndExit {
		return &ProxyServer{
			Config:       config,
			IptInterface: iptInterface,
		}, nil
	}

	// TODO(vmarmol): Use container config for this.
	var oomAdjuster *oom.OOMAdjuster
	if config.OOMScoreAdj != 0 {
		oomAdjuster = oom.NewOOMAdjuster()
		if err := oomAdjuster.ApplyOOMScoreAdj(0, config.OOMScoreAdj); err != nil {
			glog.V(2).Info(err)
		}
	}

	if config.ResourceContainer != "" {
		// Run in its own container.
		if err := util.RunInResourceContainer(config.ResourceContainer); err != nil {
			glog.Warningf("Failed to start in resource-only container %q: %v", config.ResourceContainer, err)
		} else {
			glog.V(2).Infof("Running in resource-only container %q", config.ResourceContainer)
		}
	}

	// Create a Kube Client
	// define api config source
	if config.Kubeconfig == "" && config.Master == "" {
		glog.Warningf("Neither --kubeconfig nor --master was specified.  Using default API client.  This might not work.")
	}
	// This creates a client, first loading any specified kubeconfig
	// file, and then overriding the Master flag, if non-empty.
	kubeconfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: config.Kubeconfig},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: config.Master}}).ClientConfig()
	if err != nil {
		return nil, err
	}

	// Override kubeconfig qps/burst settings from flags
	kubeconfig.QPS = config.KubeApiQps
	kubeconfig.Burst = config.KubeApiBurst

	client, err := kubeclient.New(kubeconfig)
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}

	// Create event recorder
	hostname := nodeutil.GetHostname(config.HostnameOverride)
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "kube-proxy", Host: hostname})
	eventBroadcaster.StartRecordingToSink(client.Events(""))

	var proxier proxy.ProxyProvider
	var endpointsHandler proxyconfig.EndpointsConfigHandler

	useIptablesProxy := false
	if mayTryIptablesProxy(config.ProxyMode, client.Nodes(), hostname) {
		var err error
		// guaranteed false on error, error only necessary for debugging
		useIptablesProxy, err = iptables.ShouldUseIptablesProxier()
		if err != nil {
			glog.Errorf("Can't determine whether to use iptables proxy, using userspace proxier: %v", err)
		}
	}

	if useIptablesProxy {
		glog.V(2).Info("Using iptables Proxier.")
		proxierIptables, err := iptables.NewProxier(iptInterface, execer, config.SyncPeriod, config.MasqueradeAll)
		if err != nil {
			glog.Fatalf("Unable to create proxier: %v", err)
		}
		proxier = proxierIptables
		endpointsHandler = proxierIptables
		// No turning back. Remove artifacts that might still exist from the userspace Proxier.
		glog.V(2).Info("Tearing down userspace rules. Errors here are acceptable.")
		userspace.CleanupLeftovers(iptInterface)
	} else {
		glog.V(2).Info("Using userspace Proxier.")
		// This is a proxy.LoadBalancer which NewProxier needs but has methods we don't need for
		// our config.EndpointsConfigHandler.
		loadBalancer := userspace.NewLoadBalancerRR()
		// set EndpointsConfigHandler to our loadBalancer
		endpointsHandler = loadBalancer

		proxierUserspace, err := userspace.NewProxier(loadBalancer, config.BindAddress, iptInterface, config.PortRange, config.SyncPeriod)
		if err != nil {
			glog.Fatalf("Unable to create proxier: %v", err)
		}
		proxier = proxierUserspace
		// Remove artifacts from the pure-iptables Proxier.
		glog.V(2).Info("Tearing down pure-iptables proxy rules. Errors here are acceptable.")
		iptables.CleanupLeftovers(iptInterface)
	}
	iptInterface.AddReloadFunc(proxier.Sync)

	// Create configs (i.e. Watches for Services and Endpoints)
	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.
	serviceConfig := proxyconfig.NewServiceConfig()
	serviceConfig.RegisterHandler(proxier)

	endpointsConfig := proxyconfig.NewEndpointsConfig()
	endpointsConfig.RegisterHandler(endpointsHandler)

	proxyconfig.NewSourceAPI(
		client,
		30*time.Second,
		serviceConfig.Channel("api"),
		endpointsConfig.Channel("api"),
	)

	config.nodeRef = &api.ObjectReference{
		Kind:      "Node",
		Name:      hostname,
		UID:       types.UID(hostname),
		Namespace: "",
	}
	return NewProxyServer(config, iptInterface, proxier, recorder)
}

// Run runs the specified ProxyServer.  This should never exit (unless CleanupAndExit is set).
func (s *ProxyServer) Run(_ []string) error {
	// remove iptables rules and exit
	if s.Config.CleanupAndExit {
		encounteredError := userspace.CleanupLeftovers(s.IptInterface)
		encounteredError = iptables.CleanupLeftovers(s.IptInterface) || encounteredError
		if encounteredError {
			return errors.New("Encountered an error while tearing down rules.")
		}
		return nil
	}

	// Birth Cry after the birth is successful
	s.birthCry()

	// Start up Healthz service if requested
	if s.Config.HealthzPort > 0 {
		go util.Until(func() {
			err := http.ListenAndServe(s.Config.HealthzBindAddress.String()+":"+strconv.Itoa(s.Config.HealthzPort), nil)
			if err != nil {
				glog.Errorf("Starting health server failed: %v", err)
			}
		}, 5*time.Second, util.NeverStop)
	}

	// Just loop forever for now...
	s.Proxier.SyncLoop()
	return nil
}

type nodeGetter interface {
	Get(hostname string) (*api.Node, error)
}

func mayTryIptablesProxy(proxyMode string, client nodeGetter, hostname string) bool {
	if proxyMode == proxyModeIptables {
		glog.V(1).Infof("Flag proxy-mode allows iptables proxy")
		return true
	} else if proxyMode != "" {
		glog.V(1).Infof("Flag proxy-mode=%q forbids iptables proxy", proxyMode)
		return false
	}
	// proxyMode == "" - choose the best option.
	if client == nil {
		glog.Errorf("Not trying iptables proxy: nodeGetter is nil")
		return false
	}
	node, err := client.Get(hostname)
	if err != nil {
		glog.Errorf("Not trying iptables proxy: can't get Node %q: %v", hostname, err)
		return false
	}
	if node == nil {
		glog.Errorf("Not trying iptables proxy: got nil Node %q", hostname)
		return false
	}
	proxyMode, found := node.Annotations[betaProxyModeAnnotation]
	if found {
		glog.V(1).Infof("Found beta annotation %q = %q", betaProxyModeAnnotation, proxyMode)
	} else {
		// We already published some information about this annotation with the "experimental" name, so we will respect it.
		proxyMode, found = node.Annotations[experimentalProxyModeAnnotation]
		if found {
			glog.V(1).Infof("Found experimental annotation %q = %q", experimentalProxyModeAnnotation, proxyMode)
		}
	}
	if proxyMode == proxyModeIptables {
		glog.V(1).Infof("Annotation allows iptables proxy")
		return true
	}
	glog.V(1).Infof("Not trying iptables proxy: %+v", node)
	return false
}

func (s *ProxyServer) birthCry() {
	s.Recorder.Eventf(s.Config.nodeRef, "Starting", "Starting kube-proxy.")
}
