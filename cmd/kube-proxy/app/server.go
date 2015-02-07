/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/healthz"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/hyperkube"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/proxy"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/proxy/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/iptables"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// ProxyServer contains configures and runs a Kubernetes proxy server
type ProxyServer struct {
	EtcdServerList util.StringList
	EtcdConfigFile string
	BindAddress    util.IP
	ClientConfig   client.Config
	HealthzPort    int
	OOMScoreAdj    int
}

// NewProxyServer creates a new ProxyServer object with default parameters
func NewProxyServer() *ProxyServer {
	return &ProxyServer{
		BindAddress: util.IP(net.ParseIP("0.0.0.0")),
		HealthzPort: 10249,
		OOMScoreAdj: -899,
	}
}

// NewHyperkubeServer creates a new hyperkube Server object that includes the
// description and flags.
func NewHyperkubeServer() *hyperkube.Server {
	s := NewProxyServer()

	hks := hyperkube.Server{
		SimpleUsage: "proxy",
		Long: `The Kubernetes proxy server is responsible for taking traffic directed at
services and forwarding it to the appropriate pods.  It generally runs on
nodes next to the Kubelet and proxies traffic from local pods to remote pods.
It is also used when handling incoming external traffic.`,
		Run: func(_ *hyperkube.Server, args []string) error {
			return s.Run(args)
		},
	}
	s.AddFlags(hks.Flags())
	return &hks
}

// AddFlags adds flags for a specific ProxyServer to the specified FlagSet
func (s *ProxyServer) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.EtcdConfigFile, "etcd_config", s.EtcdConfigFile, "The config file for the etcd client. Mutually exclusive with -etcd_servers")
	fs.Var(&s.EtcdServerList, "etcd_servers", "List of etcd servers to watch (http://ip:port), comma separated (optional). Mutually exclusive with -etcd_config")
	fs.Var(&s.BindAddress, "bind_address", "The IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces)")
	client.BindClientConfigFlags(fs, &s.ClientConfig)
	fs.IntVar(&s.HealthzPort, "healthz_port", s.HealthzPort, "The port to bind the health check server. Use 0 to disable.")
	fs.IntVar(&s.OOMScoreAdj, "oom_score_adj", s.OOMScoreAdj, "The oom_score_adj value for kube-proxy process. Values must be within the range [-1000, 1000]")
}

// Run runs the specified ProxyServer.  This should never exit.
func (s *ProxyServer) Run(_ []string) error {
	if err := util.ApplyOomScoreAdj(0, s.OOMScoreAdj); err != nil {
		glog.Info(err)
	}

	serviceConfig := config.NewServiceConfig()
	endpointsConfig := config.NewEndpointsConfig()

	protocol := iptables.ProtocolIpv4
	if net.IP(s.BindAddress).To4() == nil {
		protocol = iptables.ProtocolIpv6
	}
	loadBalancer := proxy.NewLoadBalancerRR()
	proxier := proxy.NewProxier(loadBalancer, net.IP(s.BindAddress), iptables.New(exec.New(), protocol))
	if proxier == nil {
		glog.Fatalf("failed to create proxier, aborting")
	}

	// Wire proxier to handle changes to services
	serviceConfig.RegisterHandler(proxier)
	// And wire loadBalancer to handle changes to endpoints to services
	endpointsConfig.RegisterHandler(loadBalancer)

	// Note: RegisterHandler() calls need to happen before creation of Sources because sources
	// only notify on changes, and the initial update (on process start) may be lost if no handlers
	// are registered yet.

	// define api config source
	if s.ClientConfig.Host != "" {
		glog.Infof("Using API calls to get config %v", s.ClientConfig.Host)
		client, err := client.New(&s.ClientConfig)
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
	} else {
		var etcdClient *etcd.Client

		// Set up etcd client
		if len(s.EtcdServerList) > 0 {
			// Set up logger for etcd client
			etcd.SetLogger(util.NewLogger("etcd "))
			etcdClient = etcd.NewClient(s.EtcdServerList)
		} else if s.EtcdConfigFile != "" {
			// Set up logger for etcd client
			etcd.SetLogger(util.NewLogger("etcd "))
			var err error
			etcdClient, err = etcd.NewClientFromFile(s.EtcdConfigFile)

			if err != nil {
				glog.Fatalf("Error with etcd config file: %v", err)
			}
		}

		// Create a configuration source that handles configuration from etcd.
		if etcdClient != nil {
			glog.Infof("Using etcd servers %v", etcdClient.GetCluster())

			config.NewConfigSourceEtcd(etcdClient,
				serviceConfig.Channel("etcd"),
				endpointsConfig.Channel("etcd"))
		}
	}

	if s.HealthzPort > 0 {
		go util.Forever(func() {
			err := http.ListenAndServe(s.BindAddress.String()+":"+strconv.Itoa(s.HealthzPort), nil)
			if err != nil {
				glog.Errorf("Starting health server failed: %v", err)
			}
		}, 5*time.Second)
	}

	// Just loop forever for now...
	proxier.SyncLoop()
	return nil
}
