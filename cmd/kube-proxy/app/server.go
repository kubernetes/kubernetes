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
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/proxy/userspace"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// ProxyServer contains configures and runs a Kubernetes proxy server
type ProxyServer struct {
	BindAddress         net.IP
	HealthzPort         int
	HealthzBindAddress  net.IP
	OOMScoreAdj         int
	ResourceContainer   string
	Master              string
	Kubeconfig          string
	PortRange           util.PortRange
	Recorder            record.EventRecorder
	HostnameOverride    string
	ForceUserspaceProxy bool
	SyncPeriod          time.Duration
	nodeRef             *api.ObjectReference // Reference to this node.
	MasqueradeAll       bool
	CleanupAndExit      bool
}

// NewProxyServer creates a new ProxyServer object with default parameters
func NewProxyServer() *ProxyServer {
	return &ProxyServer{
		BindAddress:        net.ParseIP("0.0.0.0"),
		HealthzPort:        10249,
		HealthzBindAddress: net.ParseIP("127.0.0.1"),
		OOMScoreAdj:        qos.KubeProxyOomScoreAdj,
		ResourceContainer:  "/kube-proxy",
		SyncPeriod:         5 * time.Second,
	}
}

// AddFlags adds flags for a specific ProxyServer to the specified FlagSet
func (s *ProxyServer) AddFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.BindAddress, "bind-address", s.BindAddress, "The IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.IntVar(&s.HealthzPort, "healthz-port", s.HealthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.IPVar(&s.HealthzBindAddress, "healthz-bind-address", s.HealthzBindAddress, "The IP address for the health check server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)")
	fs.IntVar(&s.OOMScoreAdj, "oom-score-adj", s.OOMScoreAdj, "The oom-score-adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
	fs.StringVar(&s.ResourceContainer, "resource-container", s.ResourceContainer, "Absolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization information (the master location is set by the master flag).")
	fs.Var(&s.PortRange, "proxy-port-range", "Range of host ports (beginPort-endPort, inclusive) that may be consumed in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.")
	fs.StringVar(&s.HostnameOverride, "hostname-override", s.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.BoolVar(&s.ForceUserspaceProxy, "legacy-userspace-proxy", true, "Use the legacy userspace proxy (instead of the pure iptables proxy).")
	fs.DurationVar(&s.SyncPeriod, "iptables-sync-period", 5*time.Second, "How often iptables rules are refreshed (e.g. '5s', '1m', '2h22m').  Must be greater than 0.")
	fs.BoolVar(&s.MasqueradeAll, "masquerade-all", false, "If using the pure iptables proxy, SNAT everything")
	fs.BoolVar(&s.CleanupAndExit, "cleanup-iptables", false, "If true cleanup iptables rules and exit.")
}

// Run runs the specified ProxyServer.  This should never exit (unless CleanupAndExit is set).
func (s *ProxyServer) Run(_ []string) error {
	protocol := utiliptables.ProtocolIpv4
	if s.BindAddress.To4() == nil {
		protocol = utiliptables.ProtocolIpv6
	}

	// remove iptables rules and exit
	if s.CleanupAndExit {
		execer := exec.New()
		ipt := utiliptables.New(execer, protocol)
		encounteredError := userspace.CleanupLeftovers(ipt)
		encounteredError = iptables.CleanupLeftovers(ipt) || encounteredError
		if encounteredError {
			return errors.New("Encountered an error while tearing down rules.")
		}
		return nil
	}

	// TODO(vmarmol): Use container config for this.
	oomAdjuster := oom.NewOomAdjuster()
	if err := oomAdjuster.ApplyOomScoreAdj(0, s.OOMScoreAdj); err != nil {
		glog.V(2).Info(err)
	}

	// Run in its own container.
	if err := util.RunInResourceContainer(s.ResourceContainer); err != nil {
		glog.Warningf("Failed to start in resource-only container %q: %v", s.ResourceContainer, err)
	} else {
		glog.V(2).Infof("Running in resource-only container %q", s.ResourceContainer)
	}

	// define api config source
	if s.Kubeconfig == "" && s.Master == "" {
		glog.Warningf("Neither --kubeconfig nor --master was specified.  Using default API client.  This might not work.")
	}

	// This creates a client, first loading any specified kubeconfig
	// file, and then overriding the Master flag, if non-empty.
	kubeconfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: s.Kubeconfig},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: s.Master}}).ClientConfig()
	if err != nil {
		return err
	}

	client, err := client.New(kubeconfig)
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}

	// Add event recorder
	Hostname := nodeutil.GetHostname(s.HostnameOverride)
	eventBroadcaster := record.NewBroadcaster()
	s.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: "kube-proxy", Host: Hostname})
	eventBroadcaster.StartRecordingToSink(client.Events(""))

	s.nodeRef = &api.ObjectReference{
		Kind:      "Node",
		Name:      Hostname,
		UID:       types.UID(Hostname),
		Namespace: "",
	}

	serviceConfig := config.NewServiceConfig()
	endpointsConfig := config.NewEndpointsConfig()

	var proxier proxy.ProxyProvider
	var endpointsHandler config.EndpointsConfigHandler

	shouldUseIptables := false
	if !s.ForceUserspaceProxy {
		var err error
		// guaranteed false on error, error only necessary for debugging
		shouldUseIptables, err = iptables.ShouldUseIptablesProxier()
		if err != nil {
			glog.Errorf("Can't determine whether to use iptables proxy, using userspace proxier: %v", err)
		}
	}

	if shouldUseIptables {
		glog.V(2).Info("Using iptables Proxier.")

		execer := exec.New()
		ipt := utiliptables.New(execer, protocol)
		proxierIptables, err := iptables.NewProxier(ipt, execer, s.SyncPeriod, s.MasqueradeAll)
		if err != nil {
			glog.Fatalf("Unable to create proxier: %v", err)
		}
		proxier = proxierIptables
		endpointsHandler = proxierIptables
		// No turning back. Remove artifacts that might still exist from the userspace Proxier.
		glog.V(2).Info("Tearing down userspace rules. Errors here are acceptable.")
		userspace.CleanupLeftovers(ipt)

	} else {
		glog.V(2).Info("Using userspace Proxier.")
		// This is a proxy.LoadBalancer which NewProxier needs but has methods we don't need for
		// our config.EndpointsConfigHandler.
		loadBalancer := userspace.NewLoadBalancerRR()
		// set EndpointsConfigHandler to our loadBalancer
		endpointsHandler = loadBalancer

		execer := exec.New()
		ipt := utiliptables.New(execer, protocol)
		proxierUserspace, err := userspace.NewProxier(loadBalancer, s.BindAddress, ipt, s.PortRange, s.SyncPeriod)
		if err != nil {
			glog.Fatalf("Unable to create proxier: %v", err)
		}
		proxier = proxierUserspace
		// Remove artifacts from the pure-iptables Proxier.
		glog.V(2).Info("Tearing down pure-iptables proxy rules. Errors here are acceptable.")
		iptables.CleanupLeftovers(ipt)
	}

	// Birth Cry after the birth is successful
	s.birthCry()

	// Wire proxier to handle changes to services
	serviceConfig.RegisterHandler(proxier)
	// And wire endpointsHandler to handle changes to endpoints to services
	endpointsConfig.RegisterHandler(endpointsHandler)

	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.

	config.NewSourceAPI(
		client,
		30*time.Second,
		serviceConfig.Channel("api"),
		endpointsConfig.Channel("api"),
	)

	if s.HealthzPort > 0 {
		go util.Until(func() {
			err := http.ListenAndServe(s.HealthzBindAddress.String()+":"+strconv.Itoa(s.HealthzPort), nil)
			if err != nil {
				glog.Errorf("Starting health server failed: %v", err)
			}
		}, 5*time.Second, util.NeverStop)
	}

	// Just loop forever for now...
	proxier.SyncLoop()
	return nil
}

func (s *ProxyServer) birthCry() {
	s.Recorder.Eventf(s.nodeRef, "Starting", "Starting kube-proxy.")
}
