/*
Copyright 2015 The Kubernetes Authors.

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

package controllermanager

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"strconv"
	"time"

	kubecontrollermanager "k8s.io/kubernetes/cmd/kube-controller-manager/app"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	"k8s.io/kubernetes/contrib/mesos/pkg/node"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/mesos"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controller/deployment"
	endpointcontroller "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/job"
	namespacecontroller "k8s.io/kubernetes/pkg/controller/namespace"
	nodecontroller "k8s.io/kubernetes/pkg/controller/node"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/pkg/controller/podgc"
	replicaset "k8s.io/kubernetes/pkg/controller/replicaset"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	routecontroller "k8s.io/kubernetes/pkg/controller/route"
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/pkg/healthz"
	quotainstall "k8s.io/kubernetes/pkg/quota/install"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/util/crypto"
	"k8s.io/kubernetes/pkg/util/wait"

	"k8s.io/kubernetes/contrib/mesos/pkg/profile"
	kmendpoint "k8s.io/kubernetes/contrib/mesos/pkg/service"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/pflag"
)

const (
	// Jitter used when starting controller managers
	ControllerStartJitter = 1.0
)

// CMServer is the main context object for the controller manager.
type CMServer struct {
	*options.CMServer
	UseHostPortEndpoints bool
}

// NewCMServer creates a new CMServer with a default config.
func NewCMServer() *CMServer {
	s := &CMServer{
		CMServer: options.NewCMServer(),
	}
	s.CloudProvider = mesos.ProviderName
	s.UseHostPortEndpoints = true
	return s
}

// AddFlags adds flags for a specific CMServer to the specified FlagSet
func (s *CMServer) AddFlags(fs *pflag.FlagSet) {
	s.CMServer.AddFlags(fs)
	fs.BoolVar(&s.UseHostPortEndpoints, "host-port-endpoints", s.UseHostPortEndpoints, "Map service endpoints to hostIP:hostPort instead of podIP:containerPort. Default true.")
}

func (s *CMServer) resyncPeriod() time.Duration {
	factor := rand.Float64() + 1
	return time.Duration(float64(time.Hour) * 12.0 * factor)
}

func (s *CMServer) Run(_ []string) error {
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

	kubeconfig.QPS = 20.0
	kubeconfig.Burst = 30

	kubeClient, err := client.New(kubeconfig)
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}

	go func() {
		mux := http.NewServeMux()
		healthz.InstallHandler(mux)
		if s.EnableProfiling {
			profile.InstallHandler(mux)
		}
		mux.Handle("/metrics", prometheus.Handler())
		server := &http.Server{
			Addr:    net.JoinHostPort(s.Address, strconv.Itoa(int(s.Port))),
			Handler: mux,
		}
		glog.Fatal(server.ListenAndServe())
	}()

	endpoints := s.createEndpointController(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "endpoint-controller")))
	go endpoints.Run(int(s.ConcurrentEndpointSyncs), wait.NeverStop)

	go replicationcontroller.NewReplicationManagerFromClient(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "replication-controller")), s.resyncPeriod, replicationcontroller.BurstReplicas, int(s.LookupCacheSizeForRC)).
		Run(int(s.ConcurrentRCSyncs), wait.NeverStop)

	if s.TerminatedPodGCThreshold > 0 {
		go podgc.New(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "pod-garbage-collector")), s.resyncPeriod, int(s.TerminatedPodGCThreshold)).
			Run(wait.NeverStop)
	}

	//TODO(jdef) should eventually support more cloud providers here
	if s.CloudProvider != mesos.ProviderName {
		glog.Fatalf("Only provider %v is supported, you specified %v", mesos.ProviderName, s.CloudProvider)
	}
	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}
	_, clusterCIDR, _ := net.ParseCIDR(s.ClusterCIDR)
	_, serviceCIDR, _ := net.ParseCIDR(s.ServiceCIDR)
	nodeController, err := nodecontroller.NewNodeControllerFromClient(cloud, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "node-controller")),
		s.PodEvictionTimeout.Duration, s.DeletingPodsQps,
		s.NodeMonitorGracePeriod.Duration, s.NodeStartupGracePeriod.Duration, s.NodeMonitorPeriod.Duration, clusterCIDR, serviceCIDR, int(s.NodeCIDRMaskSize), s.AllocateNodeCIDRs)
	if err != nil {
		glog.Fatalf("Failed to initialize nodecontroller: %v", err)
	}
	nodeController.Run(s.NodeSyncPeriod.Duration)

	nodeStatusUpdaterController := node.NewStatusUpdater(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "node-status-controller")), s.NodeMonitorPeriod.Duration, time.Now)
	if err := nodeStatusUpdaterController.Run(wait.NeverStop); err != nil {
		glog.Fatalf("Failed to start node status update controller: %v", err)
	}

	serviceController, err := servicecontroller.New(cloud, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "service-controller")), s.ClusterName)
	if err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	} else {
		serviceController.Run(int(s.ConcurrentServiceSyncs))
	}

	if s.AllocateNodeCIDRs && s.ConfigureCloudRoutes {
		if cloud == nil {
			glog.Warning("configure-cloud-routes is set, but no cloud provider specified. Will not configure cloud provider routes.")
		} else if routes, ok := cloud.Routes(); !ok {
			glog.Warning("configure-cloud-routes is set, but cloud provider does not support routes. Will not configure cloud provider routes.")
		} else {
			routeController := routecontroller.New(routes, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "route-controller")), s.ClusterName, clusterCIDR)
			routeController.Run(s.NodeSyncPeriod.Duration)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	} else {
		glog.Infof("Will not configure cloud provider routes for allocate-node-cidrs: %v, configure-cloud-routes: %v.", s.AllocateNodeCIDRs, s.ConfigureCloudRoutes)
	}

	resourceQuotaControllerClient := clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "resource-quota-controller"))
	resourceQuotaRegistry := quotainstall.NewRegistry(resourceQuotaControllerClient)
	groupKindsToReplenish := []unversioned.GroupKind{
		api.Kind("Pod"),
		api.Kind("Service"),
		api.Kind("ReplicationController"),
		api.Kind("PersistentVolumeClaim"),
		api.Kind("Secret"),
	}
	resourceQuotaControllerOptions := &resourcequotacontroller.ResourceQuotaControllerOptions{
		KubeClient:                resourceQuotaControllerClient,
		ResyncPeriod:              controller.StaticResyncPeriodFunc(s.ResourceQuotaSyncPeriod.Duration),
		Registry:                  resourceQuotaRegistry,
		GroupKindsToReplenish:     groupKindsToReplenish,
		ReplenishmentResyncPeriod: s.resyncPeriod,
		ControllerFactory:         resourcequotacontroller.NewReplenishmentControllerFactoryFromClient(resourceQuotaControllerClient),
	}
	go resourcequotacontroller.NewResourceQuotaController(resourceQuotaControllerOptions).Run(int(s.ConcurrentResourceQuotaSyncs), wait.NeverStop)

	// If apiserver is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and controller manager at the same time.
	var versionStrings []string
	err = wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		if versionStrings, err = restclient.ServerAPIVersions(kubeconfig); err == nil {
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

	// Find the list of namespaced resources via discovery that the namespace controller must manage
	namespaceKubeClient := clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "namespace-controller"))
	namespaceClientPool := dynamic.NewClientPool(restclient.AddUserAgent(kubeconfig, "namespace-controller"), dynamic.LegacyAPIPathResolverFunc)
	groupVersionResources, err := namespaceKubeClient.Discovery().ServerPreferredNamespacedResources()
	if err != nil {
		glog.Fatalf("Failed to get supported resources from server: %v", err)
	}
	namespaceController := namespacecontroller.NewNamespaceController(namespaceKubeClient, namespaceClientPool, groupVersionResources, s.NamespaceSyncPeriod.Duration, api.FinalizerKubernetes)
	go namespaceController.Run(int(s.ConcurrentNamespaceSyncs), wait.NeverStop)

	groupVersion := "extensions/v1beta1"
	resources, found := resourceMap[groupVersion]
	// TODO(k8s): this needs to be dynamic so users don't have to restart their controller manager if they change the apiserver
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "horizontalpodautoscalers") {
			glog.Infof("Starting horizontal pod controller.")
			hpaClient := clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "horizontal-pod-autoscaler"))
			metricsClient := metrics.NewHeapsterMetricsClient(
				hpaClient,
				metrics.DefaultHeapsterNamespace,
				metrics.DefaultHeapsterScheme,
				metrics.DefaultHeapsterService,
				metrics.DefaultHeapsterPort,
			)
			go podautoscaler.NewHorizontalController(hpaClient.Core(), hpaClient.Extensions(), hpaClient, metricsClient, s.HorizontalPodAutoscalerSyncPeriod.Duration).
				Run(wait.NeverStop)
		}

		if containsResource(resources, "daemonsets") {
			glog.Infof("Starting daemon set controller")
			go daemon.NewDaemonSetsControllerFromClient(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "daemon-set-controller")), s.resyncPeriod, int(s.LookupCacheSizeForDaemonSet)).
				Run(int(s.ConcurrentDaemonSetSyncs), wait.NeverStop)
		}

		if containsResource(resources, "jobs") {
			glog.Infof("Starting job controller")
			go job.NewJobControllerFromClient(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "job-controller")), s.resyncPeriod).
				Run(int(s.ConcurrentJobSyncs), wait.NeverStop)
		}

		if containsResource(resources, "deployments") {
			glog.Infof("Starting deployment controller")
			go deployment.NewDeploymentController(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "deployment-controller")), s.resyncPeriod).
				Run(int(s.ConcurrentDeploymentSyncs), wait.NeverStop)
		}

		if containsResource(resources, "replicasets") {
			glog.Infof("Starting ReplicaSet controller")
			go replicaset.NewReplicaSetControllerFromClient(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "replicaset-controller")), s.resyncPeriod, replicaset.BurstReplicas, int(s.LookupCacheSizeForRS)).
				Run(int(s.ConcurrentRSSyncs), wait.NeverStop)
		}
	}

	provisioner, err := kubecontrollermanager.NewVolumeProvisioner(cloud, s.VolumeConfiguration)
	if err != nil {
		glog.Fatalf("A Provisioner could not be created: %v, but one was expected. Provisioning will not work. This functionality is considered an early Alpha version.", err)
	}

	volumeController := persistentvolumecontroller.NewPersistentVolumeController(
		clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "persistent-volume-binder")),
		s.PVClaimBinderSyncPeriod.Duration,
		provisioner,
		kubecontrollermanager.ProbeRecyclableVolumePlugins(s.VolumeConfiguration),
		cloud,
		s.ClusterName,
		nil,
		nil,
		nil,
		s.VolumeConfiguration.EnableDynamicProvisioning,
	)
	volumeController.Run()

	var rootCA []byte

	if s.RootCAFile != "" {
		rootCA, err = ioutil.ReadFile(s.RootCAFile)
		if err != nil {
			return fmt.Errorf("error reading root-ca-file at %s: %v", s.RootCAFile, err)
		}
		if _, err := crypto.CertsFromPEM(rootCA); err != nil {
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
			go serviceaccountcontroller.NewTokensController(
				clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "tokens-controller")),
				serviceaccountcontroller.TokensControllerOptions{
					TokenGenerator: serviceaccount.JWTTokenGenerator(privateKey),
					RootCA:         rootCA,
				},
			).Run(int(s.ConcurrentSATokenSyncs), wait.NeverStop)
		}
	}

	serviceaccountcontroller.NewServiceAccountsController(
		clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "service-account-controller")),
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
	).Run()

	select {}
}

func (s *CMServer) createEndpointController(client *clientset.Clientset) kmendpoint.EndpointController {
	if s.UseHostPortEndpoints {
		glog.V(2).Infof("Creating hostIP:hostPort endpoint controller")
		return kmendpoint.NewEndpointController(client)
	}
	glog.V(2).Infof("Creating podIP:containerPort endpoint controller")
	stockEndpointController := endpointcontroller.NewEndpointControllerFromClient(client, s.resyncPeriod)
	return stockEndpointController
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
