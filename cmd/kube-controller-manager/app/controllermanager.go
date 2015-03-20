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

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
package app

import (
	"net"
	"net/http"
	"net/http/pprof"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	nodeControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/controller"
	replicationControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/namespace"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/resourcequota"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// CMServer is the main context object for the controller manager.
type CMServer struct {
	Port                    int
	Address                 util.IP
	ClientConfig            client.Config
	CloudProvider           string
	CloudConfigFile         string
	MinionRegexp            string
	NodeSyncPeriod          time.Duration
	ResourceQuotaSyncPeriod time.Duration
	NamespaceSyncPeriod     time.Duration
	RegisterRetryCount      int
	MachineList             util.StringList
	SyncNodeList            bool
	SyncNodeStatus          bool
	PodEvictionTimeout      time.Duration

	// TODO: Discover these by pinging the host machines, and rip out these params.
	NodeMilliCPU int64
	NodeMemory   resource.Quantity

	KubeletConfig   client.KubeletConfig
	EnableProfiling bool
}

// NewCMServer creates a new CMServer with a default config.
func NewCMServer() *CMServer {
	s := CMServer{
		Port:                    ports.ControllerManagerPort,
		Address:                 util.IP(net.ParseIP("127.0.0.1")),
		NodeSyncPeriod:          10 * time.Second,
		ResourceQuotaSyncPeriod: 10 * time.Second,
		NamespaceSyncPeriod:     10 * time.Second,
		RegisterRetryCount:      10,
		PodEvictionTimeout:      5 * time.Minute,
		NodeMilliCPU:            1000,
		NodeMemory:              resource.MustParse("3Gi"),
		SyncNodeList:            true,
		SyncNodeStatus:          false,
		KubeletConfig: client.KubeletConfig{
			Port:        ports.KubeletPort,
			EnableHttps: false,
		},
	}
	return &s
}

// AddFlags adds flags for a specific CMServer to the specified FlagSet
func (s *CMServer) AddFlags(fs *pflag.FlagSet) {
	fs.IntVar(&s.Port, "port", s.Port, "The port that the controller-manager's http service runs on")
	fs.Var(&s.Address, "address", "The IP address to serve on (set to 0.0.0.0 for all interfaces)")
	client.BindClientConfigFlags(fs, &s.ClientConfig)
	fs.StringVar(&s.CloudProvider, "cloud_provider", s.CloudProvider, "The provider for cloud services.  Empty string for no provider.")
	fs.StringVar(&s.CloudConfigFile, "cloud_config", s.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	fs.StringVar(&s.MinionRegexp, "minion_regexp", s.MinionRegexp, "If non empty, and --cloud_provider is specified, a regular expression for matching minion VMs.")
	fs.DurationVar(&s.NodeSyncPeriod, "node_sync_period", s.NodeSyncPeriod, ""+
		"The period for syncing nodes from cloudprovider. Longer periods will result in "+
		"fewer calls to cloud provider, but may delay addition of new nodes to cluster.")
	fs.DurationVar(&s.ResourceQuotaSyncPeriod, "resource_quota_sync_period", s.ResourceQuotaSyncPeriod, "The period for syncing quota usage status in the system")
	fs.DurationVar(&s.NamespaceSyncPeriod, "namespace_sync_period", s.NamespaceSyncPeriod, "The period for syncing namespace life-cycle updates")
	fs.DurationVar(&s.PodEvictionTimeout, "pod_eviction_timeout", s.PodEvictionTimeout, "The grace peroid for deleting pods on failed nodes.")
	fs.IntVar(&s.RegisterRetryCount, "register_retry_count", s.RegisterRetryCount, ""+
		"The number of retries for initial node registration.  Retry interval equals node_sync_period.")
	fs.Var(&s.MachineList, "machines", "List of machines to schedule onto, comma separated.")
	fs.BoolVar(&s.SyncNodeList, "sync_nodes", s.SyncNodeList, "If true, and --cloud_provider is specified, sync nodes from the cloud provider. Default true.")
	fs.BoolVar(&s.SyncNodeStatus, "sync_node_status", s.SyncNodeStatus, "Should node controler send probes to kubelets and update NodeStatus.")
	// TODO: Discover these by pinging the host machines, and rip out these flags.
	// TODO: in the meantime, use resource.QuantityFlag() instead of these
	fs.Int64Var(&s.NodeMilliCPU, "node_milli_cpu", s.NodeMilliCPU, "The amount of MilliCPU provisioned on each node")
	fs.Var(resource.NewQuantityFlagValue(&s.NodeMemory), "node_memory", "The amount of memory (in bytes) provisioned on each node")
	client.BindKubeletClientConfigFlags(fs, &s.KubeletConfig)
	fs.BoolVar(&s.EnableProfiling, "profiling", false, "Enable profiling via web interface host:port/debug/pprof/")
}

func (s *CMServer) verifyMinionFlags() {
	if !s.SyncNodeList && s.MinionRegexp != "" {
		glog.Info("--minion_regexp is ignored by --sync_nodes=false")
	}
	if s.CloudProvider == "" || s.MinionRegexp == "" {
		if len(s.MachineList) == 0 {
			glog.Info("No machines specified!")
		}
		return
	}
	if len(s.MachineList) != 0 {
		glog.Info("--machines is overwritten by --minion_regexp")
	}
}

// Run runs the CMServer.  This should never exit.
func (s *CMServer) Run(_ []string) error {
	s.verifyMinionFlags()

	if len(s.ClientConfig.Host) == 0 {
		glog.Fatal("usage: controller-manager --master <master>")
	}

	kubeClient, err := client.New(&s.ClientConfig)
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}

	go func() {
		if s.EnableProfiling {
			mux := http.NewServeMux()
			mux.HandleFunc("/debug/pprof/", pprof.Index)
			mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
			mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		}
		http.ListenAndServe(net.JoinHostPort(s.Address.String(), strconv.Itoa(s.Port)), nil)
	}()

	endpoints := service.NewEndpointController(kubeClient)
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*10)

	controllerManager := replicationControllerPkg.NewReplicationManager(kubeClient)
	controllerManager.Run(replicationControllerPkg.DefaultSyncPeriod)

	kubeletClient, err := client.NewKubeletClient(&s.KubeletConfig)
	if err != nil {
		glog.Fatalf("Failure to start kubelet client: %v", err)
	}

	cloud := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	nodeResources := &api.NodeResources{
		Capacity: api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(s.NodeMilliCPU, resource.DecimalSI),
			api.ResourceMemory: s.NodeMemory,
		},
	}

	nodeController := nodeControllerPkg.NewNodeController(cloud, s.MinionRegexp, s.MachineList, nodeResources,
		kubeClient, kubeletClient, s.RegisterRetryCount, s.PodEvictionTimeout)
	nodeController.Run(s.NodeSyncPeriod, s.SyncNodeList, s.SyncNodeStatus)

	resourceQuotaManager := resourcequota.NewResourceQuotaManager(kubeClient)
	resourceQuotaManager.Run(s.ResourceQuotaSyncPeriod)

	namespaceManager := namespace.NewNamespaceManager(kubeClient)
	namespaceManager.Run(s.NamespaceSyncPeriod)

	select {}
	return nil
}
