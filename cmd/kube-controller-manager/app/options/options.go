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

// Package options provides the flags used for the controller manager.
//
// CAUTION: If you update code in this file, you may need to also update code
//          in contrib/mesos/pkg/controllermanager/controllermanager.go
package options

import (
	"net"
	"time"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/leaderelection"
	daemonoptions "k8s.io/kubernetes/pkg/controller/daemon/options"
	deploymentoptions "k8s.io/kubernetes/pkg/controller/deployment/options"
	endpointoptions "k8s.io/kubernetes/pkg/controller/endpoint/options"
	gcoptions "k8s.io/kubernetes/pkg/controller/gc/options"
	joboptions "k8s.io/kubernetes/pkg/controller/job/options"
	namespaceoptions "k8s.io/kubernetes/pkg/controller/namespace/options"
	nodeoptions "k8s.io/kubernetes/pkg/controller/node/options"
	pvoptions "k8s.io/kubernetes/pkg/controller/persistentvolume/options"
	hpaoptions "k8s.io/kubernetes/pkg/controller/podautoscaler/options"
	replicationoptions "k8s.io/kubernetes/pkg/controller/replication/options"
	resourcequotaoptions "k8s.io/kubernetes/pkg/controller/resourcequota/options"
	serviceoptions "k8s.io/kubernetes/pkg/controller/service/options"
	serviceaccountoptions "k8s.io/kubernetes/pkg/controller/serviceaccount/options"

	"k8s.io/kubernetes/pkg/master/ports"

	"github.com/spf13/pflag"
)

// CMServer is the main context object for the controller manager.
type CMServer struct {
	Address         net.IP
	CloudConfigFile string
	CloudProvider   string
	ClusterName     string
	EnableProfiling bool
	KubeAPIBurst    int
	KubeAPIQPS      float32
	Kubeconfig      string
	Master          string
	MinResyncPeriod time.Duration
	Port            int
	RootCAFile      string

	DaemonControllerOptions           daemonoptions.DaemonControllerOptions
	DeploymentControllerOptions       deploymentoptions.DeploymentControllerOptions
	EndpointControllerOptions         endpointoptions.EndpointControllerOptions
	GarbageCollectorOptions           gcoptions.GarbageCollectorOptions
	JobControllerOptions              joboptions.JobControllerOptions
	LeaderElection                    componentconfig.LeaderElectionConfiguration
	NamespaceControllerOptions        namespaceoptions.NamespaceControllerOptions
	NodeControllerOptions             nodeoptions.NodeControllerOptions
	PersistentVolumeControllerOptions pvoptions.PersistentVolumeControllerOptions
	PodAutoscalerOptions              hpaoptions.PodAutoscalerOptions
	ReplicationControllerOptions      replicationoptions.ReplicationControllerOptions
	ResourceQuotaControllerOptions    resourcequotaoptions.ResourceQuotaControllerOptions
	ServiceControllerOptions          serviceoptions.ServiceControllerOptions
	ServiceAccountControllerOptions   serviceaccountoptions.ServiceAccountControllerOptions

	// TODO: split into different rates for different components (?)
	NodeSyncPeriod time.Duration

	// deprecated
	DeploymentControllerSyncPeriod time.Duration
	RegisterRetryCount             int
}

// NewCMServer creates a new CMServer with a default config.
func NewCMServer() *CMServer {
	s := CMServer{
		Address:         net.ParseIP("0.0.0.0"),
		ClusterName:     "kubernetes",
		KubeAPIQPS:      20.0,
		KubeAPIBurst:    30,
		MinResyncPeriod: 12 * time.Hour,
		Port:            ports.ControllerManagerPort,

		NodeSyncPeriod: 10 * time.Second,

		LeaderElection:                    leaderelection.DefaultLeaderElectionConfiguration(),
		DeploymentControllerOptions:       deploymentoptions.NewDeploymentControllerOptions(),
		DaemonControllerOptions:           daemonoptions.NewDaemonControllerOptions(),
		EndpointControllerOptions:         endpointoptions.NewEndpointControllerOptions(),
		GarbageCollectorOptions:           gcoptions.NewGarbageCollectorOptions(),
		JobControllerOptions:              joboptions.NewJobControllerOptions(),
		NamespaceControllerOptions:        namespaceoptions.NewNamespaceControllerOptions(),
		NodeControllerOptions:             nodeoptions.NewNodeControllerOptions(),
		PersistentVolumeControllerOptions: pvoptions.NewPersistentVolumeControllerOptions(),
		PodAutoscalerOptions:              hpaoptions.NewPodAutoscalerOptions(),
		ReplicationControllerOptions:      replicationoptions.NewReplicationControllerOptions(),
		ResourceQuotaControllerOptions:    resourcequotaoptions.NewResourceQuotaControllerOptions(),
		ServiceControllerOptions:          serviceoptions.NewServiceControllerOptions(),
		ServiceAccountControllerOptions:   serviceaccountoptions.NewServiceAccountControllerOptions(),
	}
	return &s
}

// AddFlags adds flags for a specific CMServer to the specified FlagSet
func (s *CMServer) AddFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.Address, "address", s.Address, "The IP address to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider, "The provider for cloud services.  Empty string for no provider.")
	fs.StringVar(&s.ClusterName, "cluster-name", s.ClusterName, "The instance prefix for the cluster")
	fs.BoolVar(&s.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/")
	fs.IntVar(&s.KubeAPIBurst, "kube-api-burst", s.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver")
	fs.Float32Var(&s.KubeAPIQPS, "kube-api-qps", s.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.DurationVar(&s.MinResyncPeriod, "min-resync-period", s.MinResyncPeriod, "The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod")
	fs.IntVar(&s.Port, "port", s.Port, "The port that the controller-manager's http service runs on")
	fs.StringVar(&s.RootCAFile, "root-ca-file", s.RootCAFile, "If set, this root certificate authority will be included in service account's token secret. This must be a valid PEM-encoded CA bundle.")

	fs.DurationVar(&s.NodeSyncPeriod, "node-sync-period", s.NodeSyncPeriod, ""+
		"The period for syncing nodes from cloudprovider. Longer periods will result in "+
		"fewer calls to cloud provider, but may delay addition of new nodes to cluster.")
	s.LeaderElection.AddFlags(fs)
	s.DaemonControllerOptions.AddFlags(fs)
	s.DeploymentControllerOptions.AddFlags(fs)
	s.EndpointControllerOptions.AddFlags(fs)
	s.GarbageCollectorOptions.AddFlags(fs)
	s.JobControllerOptions.AddFlags(fs)
	s.NamespaceControllerOptions.AddFlags(fs)
	s.NodeControllerOptions.AddFlags(fs)
	s.PersistentVolumeControllerOptions.AddFlags(fs)
	s.PodAutoscalerOptions.AddFlags(fs)
	s.ReplicationControllerOptions.AddFlags(fs)
	s.ResourceQuotaControllerOptions.AddFlags(fs)
	s.ServiceControllerOptions.AddFlags(fs)
	s.ServiceAccountControllerOptions.AddFlags(fs)
	fs.DurationVar(&s.DeploymentControllerSyncPeriod, "deployment-controller-sync-period", 0, "Period for syncing the deployments.")
	fs.MarkDeprecated("deployment-controller-sync-period", "This flag is currently no-op and will be deleted.")
	fs.IntVar(&s.RegisterRetryCount, "register-retry-count", 0, ""+
		"The number of retries for initial node registration.  Retry interval equals node-sync-period.")
	fs.MarkDeprecated("register-retry-count", "This flag is currently no-op and will be deleted.")
}
