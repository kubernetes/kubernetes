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

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
//
// CAUTION: If you update code in this file, you may need to also update code
//          in contrib/mesos/pkg/controllermanager/controllermanager.go
package app

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/http/pprof"
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controller/deployment"
	endpointcontroller "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/gc"
	"k8s.io/kubernetes/pkg/controller/job"
	namespacecontroller "k8s.io/kubernetes/pkg/controller/namespace"
	nodecontroller "k8s.io/kubernetes/pkg/controller/node"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/persistentvolume"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	routecontroller "k8s.io/kubernetes/pkg/controller/route"
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
	"k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// CMServer is the main context object for the controller manager.
type CMServer struct {
	Port                              int
	Address                           net.IP
	CloudProvider                     string
	CloudConfigFile                   string
	ConcurrentEndpointSyncs           int
	ConcurrentRCSyncs                 int
	ConcurrentDSCSyncs                int
	ConcurrentJobSyncs                int
	ConcurrentResourceQuotaSyncs      int
	ServiceSyncPeriod                 time.Duration
	NodeSyncPeriod                    time.Duration
	ResourceQuotaSyncPeriod           time.Duration
	NamespaceSyncPeriod               time.Duration
	PVClaimBinderSyncPeriod           time.Duration
	VolumeConfigFlags                 VolumeConfigFlags
	TerminatedPodGCThreshold          int
	HorizontalPodAutoscalerSyncPeriod time.Duration
	DeploymentControllerSyncPeriod    time.Duration
	MinResyncPeriod                   time.Duration
	RegisterRetryCount                int
	NodeMonitorGracePeriod            time.Duration
	NodeStartupGracePeriod            time.Duration
	NodeMonitorPeriod                 time.Duration
	NodeStatusUpdateRetry             int
	PodEvictionTimeout                time.Duration
	DeletingPodsQps                   float32
	DeletingPodsBurst                 int
	ServiceAccountKeyFile             string
	RootCAFile                        string

	ClusterName       string
	ClusterCIDR       net.IPNet
	AllocateNodeCIDRs bool
	EnableProfiling   bool

	Master       string
	Kubeconfig   string
	KubeAPIQPS   float32
	KubeAPIBurst int
}

// NewCMServer creates a new CMServer with a default config.
func NewCMServer() *CMServer {
	s := CMServer{
		Port:                              ports.ControllerManagerPort,
		Address:                           net.ParseIP("127.0.0.1"),
		ConcurrentEndpointSyncs:           5,
		ConcurrentRCSyncs:                 5,
		ConcurrentDSCSyncs:                2,
		ConcurrentJobSyncs:                5,
		ConcurrentResourceQuotaSyncs:      5,
		ServiceSyncPeriod:                 5 * time.Minute,
		NodeSyncPeriod:                    10 * time.Second,
		ResourceQuotaSyncPeriod:           5 * time.Minute,
		NamespaceSyncPeriod:               5 * time.Minute,
		PVClaimBinderSyncPeriod:           10 * time.Minute,
		HorizontalPodAutoscalerSyncPeriod: 30 * time.Second,
		DeploymentControllerSyncPeriod:    30 * time.Second,
		MinResyncPeriod:                   12 * time.Hour,
		RegisterRetryCount:                10,
		PodEvictionTimeout:                5 * time.Minute,
		NodeMonitorGracePeriod:            40 * time.Second,
		NodeStartupGracePeriod:            60 * time.Second,
		NodeMonitorPeriod:                 5 * time.Second,
		ClusterName:                       "kubernetes",
		TerminatedPodGCThreshold:          12500,
		VolumeConfigFlags: VolumeConfigFlags{
			// default values here
			PersistentVolumeRecyclerMinimumTimeoutNFS:        300,
			PersistentVolumeRecyclerIncrementTimeoutNFS:      30,
			PersistentVolumeRecyclerMinimumTimeoutHostPath:   60,
			PersistentVolumeRecyclerIncrementTimeoutHostPath: 30,
			EnableHostPathProvisioning:                       false,
		},
		KubeAPIQPS:   20.0,
		KubeAPIBurst: 30,
	}
	return &s
}

// NewControllerManagerCommand creates a *cobra.Command object with default parameters
func NewControllerManagerCommand() *cobra.Command {
	s := NewCMServer()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "kube-controller-manager",
		Long: `The Kubernetes controller manager is a daemon that embeds
the core control loops shipped with Kubernetes. In applications of robotics and
automation, a control loop is a non-terminating loop that regulates the state of
the system. In Kubernetes, a controller is a control loop that watches the shared
state of the cluster through the apiserver and makes changes attempting to move the
current state towards the desired state. Examples of controllers that ship with
Kubernetes today are the replication controller, endpoints controller, namespace
controller, and serviceaccounts controller.`,
		Run: func(cmd *cobra.Command, args []string) {
		},
	}

	return cmd
}

// VolumeConfigFlags is used to bind CLI flags to variables.  This top-level struct contains *all* enumerated
// CLI flags meant to configure all volume plugins.  From this config, the binary will create many instances
// of volume.VolumeConfig which are then passed to the appropriate plugin. The ControllerManager binary is the only
// part of the code which knows what plugins are supported and which CLI flags correspond to each plugin.
type VolumeConfigFlags struct {
	PersistentVolumeRecyclerMinimumTimeoutNFS           int
	PersistentVolumeRecyclerPodTemplateFilePathNFS      string
	PersistentVolumeRecyclerIncrementTimeoutNFS         int
	PersistentVolumeRecyclerPodTemplateFilePathHostPath string
	PersistentVolumeRecyclerMinimumTimeoutHostPath      int
	PersistentVolumeRecyclerIncrementTimeoutHostPath    int
	EnableHostPathProvisioning                          bool
}

// AddFlags adds flags for a specific CMServer to the specified FlagSet
func (s *CMServer) AddFlags(fs *pflag.FlagSet) {
	fs.IntVar(&s.Port, "port", s.Port, "The port that the controller-manager's http service runs on")
	fs.IPVar(&s.Address, "address", s.Address, "The IP address to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider, "The provider for cloud services.  Empty string for no provider.")
	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	fs.IntVar(&s.ConcurrentEndpointSyncs, "concurrent-endpoint-syncs", s.ConcurrentEndpointSyncs, "The number of endpoint syncing operations that will be done concurrently. Larger number = faster endpoint updating, but more CPU (and network) load")
	fs.IntVar(&s.ConcurrentRCSyncs, "concurrent_rc_syncs", s.ConcurrentRCSyncs, "The number of replication controllers that are allowed to sync concurrently. Larger number = more reponsive replica management, but more CPU (and network) load")
	fs.IntVar(&s.ConcurrentResourceQuotaSyncs, "concurrent-resource-quota-syncs", s.ConcurrentResourceQuotaSyncs, "The number of resource quotas that are allowed to sync concurrently. Larger number = more responsive quota management, but more CPU (and network) load")
	fs.DurationVar(&s.ServiceSyncPeriod, "service-sync-period", s.ServiceSyncPeriod, "The period for syncing services with their external load balancers")
	fs.DurationVar(&s.NodeSyncPeriod, "node-sync-period", s.NodeSyncPeriod, ""+
		"The period for syncing nodes from cloudprovider. Longer periods will result in "+
		"fewer calls to cloud provider, but may delay addition of new nodes to cluster.")
	fs.DurationVar(&s.ResourceQuotaSyncPeriod, "resource-quota-sync-period", s.ResourceQuotaSyncPeriod, "The period for syncing quota usage status in the system")
	fs.DurationVar(&s.NamespaceSyncPeriod, "namespace-sync-period", s.NamespaceSyncPeriod, "The period for syncing namespace life-cycle updates")
	fs.DurationVar(&s.PVClaimBinderSyncPeriod, "pvclaimbinder-sync-period", s.PVClaimBinderSyncPeriod, "The period for syncing persistent volumes and persistent volume claims")
	fs.DurationVar(&s.MinResyncPeriod, "min-resync-period", s.MinResyncPeriod, "The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod")
	fs.StringVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathNFS, "pv-recycler-pod-template-filepath-nfs", s.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathNFS, "The file path to a pod definition used as a template for NFS persistent volume recycling")
	fs.IntVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutNFS, "pv-recycler-minimum-timeout-nfs", s.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutNFS, "The minimum ActiveDeadlineSeconds to use for an NFS Recycler pod")
	fs.IntVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutNFS, "pv-recycler-increment-timeout-nfs", s.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutNFS, "the increment of time added per Gi to ActiveDeadlineSeconds for an NFS scrubber pod")
	fs.StringVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathHostPath, "pv-recycler-pod-template-filepath-hostpath", s.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathHostPath, "The file path to a pod definition used as a template for HostPath persistent volume recycling. This is for development and testing only and will not work in a multi-node cluster.")
	fs.IntVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutHostPath, "pv-recycler-minimum-timeout-hostpath", s.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutHostPath, "The minimum ActiveDeadlineSeconds to use for a HostPath Recycler pod.  This is for development and testing only and will not work in a multi-node cluster.")
	fs.IntVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutHostPath, "pv-recycler-timeout-increment-hostpath", s.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutHostPath, "the increment of time added per Gi to ActiveDeadlineSeconds for a HostPath scrubber pod.  This is for development and testing only and will not work in a multi-node cluster.")
	fs.BoolVar(&s.VolumeConfigFlags.EnableHostPathProvisioning, "enable-hostpath-provisioner", s.VolumeConfigFlags.EnableHostPathProvisioning, "Enable HostPath PV provisioning when running without a cloud provider. This allows testing and development of provisioning features.  HostPath provisioning is not supported in any way, won't work in a multi-node cluster, and should not be used for anything other than testing or development.")
	fs.IntVar(&s.TerminatedPodGCThreshold, "terminated-pod-gc-threshold", s.TerminatedPodGCThreshold, "Number of terminated pods that can exist before the terminated pod garbage collector starts deleting terminated pods. If <= 0, the terminated pod garbage collector is disabled.")
	fs.DurationVar(&s.HorizontalPodAutoscalerSyncPeriod, "horizontal-pod-autoscaler-sync-period", s.HorizontalPodAutoscalerSyncPeriod, "The period for syncing the number of pods in horizontal pod autoscaler.")
	fs.DurationVar(&s.DeploymentControllerSyncPeriod, "deployment-controller-sync-period", s.DeploymentControllerSyncPeriod, "Period for syncing the deployments.")
	fs.DurationVar(&s.PodEvictionTimeout, "pod-eviction-timeout", s.PodEvictionTimeout, "The grace period for deleting pods on failed nodes.")
	fs.Float32Var(&s.DeletingPodsQps, "deleting-pods-qps", 0.1, "Number of nodes per second on which pods are deleted in case of node failure.")
	fs.IntVar(&s.DeletingPodsBurst, "deleting-pods-burst", 10, "Number of nodes on which pods are bursty deleted in case of node failure. For more details look into RateLimiter.")
	fs.IntVar(&s.RegisterRetryCount, "register-retry-count", s.RegisterRetryCount, ""+
		"The number of retries for initial node registration.  Retry interval equals node-sync-period.")
	fs.MarkDeprecated("register-retry-count", "This flag is currently no-op and will be deleted.")
	fs.DurationVar(&s.NodeMonitorGracePeriod, "node-monitor-grace-period", s.NodeMonitorGracePeriod,
		"Amount of time which we allow running Node to be unresponsive before marking it unhealty. "+
			"Must be N times more than kubelet's nodeStatusUpdateFrequency, "+
			"where N means number of retries allowed for kubelet to post node status.")
	fs.DurationVar(&s.NodeStartupGracePeriod, "node-startup-grace-period", s.NodeStartupGracePeriod,
		"Amount of time which we allow starting Node to be unresponsive before marking it unhealty.")
	fs.DurationVar(&s.NodeMonitorPeriod, "node-monitor-period", s.NodeMonitorPeriod,
		"The period for syncing NodeStatus in NodeController.")
	fs.StringVar(&s.ServiceAccountKeyFile, "service-account-private-key-file", s.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA key used to sign service account tokens.")
	fs.BoolVar(&s.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/")
	fs.StringVar(&s.ClusterName, "cluster-name", s.ClusterName, "The instance prefix for the cluster")
	fs.IPNetVar(&s.ClusterCIDR, "cluster-cidr", s.ClusterCIDR, "CIDR Range for Pods in cluster.")
	fs.BoolVar(&s.AllocateNodeCIDRs, "allocate-node-cidrs", false, "Should CIDRs for Pods be allocated and set on the cloud provider.")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&s.RootCAFile, "root-ca-file", s.RootCAFile, "If set, this root certificate authority will be included in service account's token secret. This must be a valid PEM-encoded CA bundle.")
	fs.Float32Var(&s.KubeAPIQPS, "kube-api-qps", s.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver")
	fs.IntVar(&s.KubeAPIBurst, "kube-api-burst", s.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver")
}

func (s *CMServer) ResyncPeriod() time.Duration {
	factor := rand.Float64() + 1
	return time.Duration(float64(s.MinResyncPeriod.Nanoseconds()) * factor)
}

func clientForUserAgentOrDie(config client.Config, userAgent string) *client.Client {
	fullUserAgent := client.DefaultKubernetesUserAgent() + "/" + userAgent
	config.UserAgent = fullUserAgent
	kubeClient, err := client.New(&config)
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}
	return kubeClient
}

// Run runs the CMServer.  This should never exit.
func (s *CMServer) Run(_ []string) error {
	kubeconfig, err := clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
	if err != nil {
		return err
	}

	// Override kubeconfig qps/burst settings from flags
	kubeconfig.QPS = s.KubeAPIQPS
	kubeconfig.Burst = s.KubeAPIBurst

	kubeClient, err := client.New(kubeconfig)
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}

	go func() {
		mux := http.NewServeMux()
		healthz.InstallHandler(mux)
		if s.EnableProfiling {
			mux.HandleFunc("/debug/pprof/", pprof.Index)
			mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
			mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		}
		mux.Handle("/metrics", prometheus.Handler())

		server := &http.Server{
			Addr:    net.JoinHostPort(s.Address.String(), strconv.Itoa(s.Port)),
			Handler: mux,
		}
		glog.Fatal(server.ListenAndServe())
	}()

	go endpointcontroller.NewEndpointController(clientForUserAgentOrDie(*kubeconfig, "endpoint-controller"), s.ResyncPeriod).
		Run(s.ConcurrentEndpointSyncs, util.NeverStop)

	go replicationcontroller.NewReplicationManager(
		clientForUserAgentOrDie(*kubeconfig, "replication-controller"),
		s.ResyncPeriod,
		replicationcontroller.BurstReplicas,
	).Run(s.ConcurrentRCSyncs, util.NeverStop)

	if s.TerminatedPodGCThreshold > 0 {
		go gc.New(clientForUserAgentOrDie(*kubeconfig, "garbage-collector"), s.ResyncPeriod, s.TerminatedPodGCThreshold).
			Run(util.NeverStop)
	}

	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}

	nodeController := nodecontroller.NewNodeController(cloud, clientForUserAgentOrDie(*kubeconfig, "node-controller"),
		s.PodEvictionTimeout, util.NewTokenBucketRateLimiter(s.DeletingPodsQps, s.DeletingPodsBurst),
		util.NewTokenBucketRateLimiter(s.DeletingPodsQps, s.DeletingPodsBurst),
		s.NodeMonitorGracePeriod, s.NodeStartupGracePeriod, s.NodeMonitorPeriod, &s.ClusterCIDR, s.AllocateNodeCIDRs)
	nodeController.Run(s.NodeSyncPeriod)

	serviceController := servicecontroller.New(cloud, clientForUserAgentOrDie(*kubeconfig, "service-controller"), s.ClusterName)
	if err := serviceController.Run(s.ServiceSyncPeriod, s.NodeSyncPeriod); err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	}

	if s.AllocateNodeCIDRs {
		if cloud == nil {
			glog.Warning("allocate-node-cidrs is set, but no cloud provider specified. Will not manage routes.")
		} else if routes, ok := cloud.Routes(); !ok {
			glog.Warning("allocate-node-cidrs is set, but cloud provider does not support routes. Will not manage routes.")
		} else {
			routeController := routecontroller.New(routes, clientForUserAgentOrDie(*kubeconfig, "route-controller"), s.ClusterName, &s.ClusterCIDR)
			routeController.Run(s.NodeSyncPeriod)
		}
	} else {
		glog.Infof("allocate-node-cidrs set to %v, node controller not creating routes", s.AllocateNodeCIDRs)
	}

	go resourcequotacontroller.NewResourceQuotaController(
		clientForUserAgentOrDie(*kubeconfig, "resourcequota-controller"),
		controller.StaticResyncPeriodFunc(s.ResourceQuotaSyncPeriod)).Run(s.ConcurrentResourceQuotaSyncs, util.NeverStop)

	// If apiserver is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and controller manager at the same time.
	var versionStrings []string
	err = wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		if versionStrings, err = client.ServerAPIVersions(kubeconfig); err == nil {
			return true, nil
		}
		glog.Errorf("Failed to get api versions from server: %v", err)
		return false, nil
	})
	if err != nil {
		glog.Fatalf("Failed to get api versions from server: %v", err)
	}
	versions := &unversioned.APIVersions{Versions: versionStrings}

	resourceMap, err := kubeClient.Discovery().ServerResources()
	if err != nil {
		glog.Fatalf("Failed to get supported resources from server: %v", err)
	}

	namespacecontroller.NewNamespaceController(clientForUserAgentOrDie(*kubeconfig, "namespace-controller"), versions, s.NamespaceSyncPeriod).Run()

	groupVersion := "extensions/v1beta1"
	resources, found := resourceMap[groupVersion]
	// TODO: this needs to be dynamic so users don't have to restart their controller manager if they change the apiserver
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "horizontalpodautoscalers") {
			glog.Infof("Starting horizontal pod controller.")
			hpaClient := clientForUserAgentOrDie(*kubeconfig, "horizontal-pod-autoscaler")
			metricsClient := metrics.NewHeapsterMetricsClient(
				hpaClient,
				metrics.DefaultHeapsterNamespace,
				metrics.DefaultHeapsterScheme,
				metrics.DefaultHeapsterService,
				metrics.DefaultHeapsterPort,
			)
			podautoscaler.NewHorizontalController(hpaClient, hpaClient, hpaClient, metricsClient).
				Run(s.HorizontalPodAutoscalerSyncPeriod)
		}

		if containsResource(resources, "daemonsets") {
			glog.Infof("Starting daemon set controller")
			go daemon.NewDaemonSetsController(clientForUserAgentOrDie(*kubeconfig, "daemon-set-controller"), s.ResyncPeriod).
				Run(s.ConcurrentDSCSyncs, util.NeverStop)
		}

		if containsResource(resources, "jobs") {
			glog.Infof("Starting job controller")
			go job.NewJobController(clientForUserAgentOrDie(*kubeconfig, "job-controller"), s.ResyncPeriod).
				Run(s.ConcurrentJobSyncs, util.NeverStop)
		}

		if containsResource(resources, "deployments") {
			glog.Infof("Starting deployment controller")
			deployment.New(clientForUserAgentOrDie(*kubeconfig, "deployment-controller")).
				Run(s.DeploymentControllerSyncPeriod)
		}
	}

	volumePlugins := ProbeRecyclableVolumePlugins(s.VolumeConfigFlags)
	provisioner, err := NewVolumeProvisioner(cloud, s.VolumeConfigFlags)
	if err != nil {
		glog.Fatal("A Provisioner could not be created, but one was expected. Provisioning will not work. This functionality is considered an early Alpha version.")
	}

	pvclaimBinder := persistentvolumecontroller.NewPersistentVolumeClaimBinder(clientForUserAgentOrDie(*kubeconfig, "persistent-volume-binder"), s.PVClaimBinderSyncPeriod)
	pvclaimBinder.Run()

	pvRecycler, err := persistentvolumecontroller.NewPersistentVolumeRecycler(clientForUserAgentOrDie(*kubeconfig, "persistent-volume-recycler"), s.PVClaimBinderSyncPeriod, ProbeRecyclableVolumePlugins(s.VolumeConfigFlags), cloud)
	if err != nil {
		glog.Fatalf("Failed to start persistent volume recycler: %+v", err)
	}
	pvRecycler.Run()

	if provisioner != nil {
		pvController, err := persistentvolumecontroller.NewPersistentVolumeProvisionerController(persistentvolumecontroller.NewControllerClient(clientForUserAgentOrDie(*kubeconfig, "persistent-volume-provisioner")), s.PVClaimBinderSyncPeriod, volumePlugins, provisioner, cloud)
		if err != nil {
			glog.Fatalf("Failed to start persistent volume provisioner controller: %+v", err)
		}
		pvController.Run()
	}

	var rootCA []byte

	if s.RootCAFile != "" {
		rootCA, err = ioutil.ReadFile(s.RootCAFile)
		if err != nil {
			return fmt.Errorf("error reading root-ca-file at %s: %v", s.RootCAFile, err)
		}
		if _, err := util.CertsFromPEM(rootCA); err != nil {
			return fmt.Errorf("error parsing root-ca-file at %s: %v", s.RootCAFile, err)
		}
	} else {
		rootCA = kubeconfig.CAData
	}

	if len(s.ServiceAccountKeyFile) > 0 {
		privateKey, err := serviceaccount.ReadPrivateKey(s.ServiceAccountKeyFile)
		if err != nil {
			glog.Errorf("Error reading key for service account token controller: %v", err)
		} else {
			serviceaccount.NewTokensController(
				clientForUserAgentOrDie(*kubeconfig, "tokens-controller"),
				serviceaccount.TokensControllerOptions{
					TokenGenerator: serviceaccount.JWTTokenGenerator(privateKey),
					RootCA:         rootCA,
				},
			).Run()
		}
	}

	serviceaccount.NewServiceAccountsController(
		clientForUserAgentOrDie(*kubeconfig, "service-account-controller"),
		serviceaccount.DefaultServiceAccountsControllerOptions(),
	).Run()

	select {}
}

func containsVersion(versions *unversioned.APIVersions, version string) bool {
	for ix := range versions.Versions {
		if versions.Versions[ix] == version {
			return true
		}
	}
	return false
}

func containsResource(resources *unversioned.APIResourceList, resourceName string) bool {
	for ix := range resources.APIResources {
		resource := resources.APIResources[ix]
		if resource.Name == resourceName {
			return true
		}
	}
	return false
}
