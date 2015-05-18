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
	"net"
	"net/http"
	_ "net/http/pprof"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/proxy"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/proxy/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/iptables"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// ProxyServer contains configures and runs a Kubernetes proxy server
type ProxyServer struct {
	BindAddress        util.IP
	HealthzPort        int
	HealthzBindAddress util.IP
	OOMScoreAdj        int
	ResourceContainer  string
	Master             string
	Kubeconfig         string
	PortRange          util.PortRange
}

// NewProxyServer creates a new ProxyServer object with default parameters
func NewProxyServer() *ProxyServer {
	return &ProxyServer{
		BindAddress:        util.IP(net.ParseIP("0.0.0.0")),
		HealthzPort:        10249,
		HealthzBindAddress: util.IP(net.ParseIP("127.0.0.1")),
		OOMScoreAdj:        -899,
		ResourceContainer:  "/kube-proxy",
	}
}

// AddFlags adds flags for a specific ProxyServer to the specified FlagSet
func (s *ProxyServer) AddFlags(fs *pflag.FlagSet) {
	fs.Var(&s.BindAddress, "bind-address", "The IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.IntVar(&s.HealthzPort, "healthz-port", s.HealthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.Var(&s.HealthzBindAddress, "healthz-bind-address", "The IP address for the health check server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)")
	fs.IntVar(&s.OOMScoreAdj, "oom-score-adj", s.OOMScoreAdj, "The oom_score_adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
	fs.StringVar(&s.ResourceContainer, "resource-container", s.ResourceContainer, "Absolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization information (the master location is set by the master flag).")
	fs.Var(&s.PortRange, "proxy-port-range", "Range of host ports (beginPort-endPort, inclusive) that may be consumed in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.")
}

// Run runs the specified ProxyServer.  This should never exit.
func (s *ProxyServer) Run(_ []string) error {
	// TODO(vmarmol): Use container config for this.
	if err := util.ApplyOomScoreAdj(0, s.OOMScoreAdj); err != nil {
		glog.V(2).Info(err)
	}

	// Run in its own container.
	if err := util.RunInResourceContainer(s.ResourceContainer); err != nil {
		glog.Warningf("Failed to start in resource-only container %q: %v", s.ResourceContainer, err)
	} else {
		glog.V(2).Infof("Running in resource-only container %q", s.ResourceContainer)
	}

	serviceConfig := config.NewServiceConfig()
	endpointsConfig := config.NewEndpointsConfig()

	protocol := iptables.ProtocolIpv4
	if net.IP(s.BindAddress).To4() == nil {
		protocol = iptables.ProtocolIpv6
	}
	loadBalancer := proxy.NewLoadBalancerRR()
	proxier, err := proxy.NewProxier(loadBalancer, net.IP(s.BindAddress), iptables.New(exec.New(), protocol), s.PortRange)
	if err != nil {
		glog.Fatalf("Unable to create proxer: %v", err)
	}

	// Wire proxier to handle changes to services
	serviceConfig.RegisterHandler(proxier)
	// And wire loadBalancer to handle changes to endpoints to services
	endpointsConfig.RegisterHandler(loadBalancer)

	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.

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

	config.NewSourceAPI(
		client.Services(api.NamespaceAll),
		client.Endpoints(api.NamespaceAll),
		30*time.Second,
		serviceConfig.Channel("api"),
		endpointsConfig.Channel("api"),
	)

	if s.HealthzPort > 0 {
		go util.Forever(func() {
			err := http.ListenAndServe(s.HealthzBindAddress.String()+":"+strconv.Itoa(s.HealthzPort), nil)
			if err != nil {
				glog.Errorf("Starting health server failed: %v", err)
			}
		}, 5*time.Second)
	}

	// Just loop forever for now...
	proxier.SyncLoop()
	return nil
}
