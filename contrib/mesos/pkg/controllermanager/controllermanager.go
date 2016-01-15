/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_1"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/mesos"
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
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	"k8s.io/kubernetes/contrib/mesos/pkg/profile"
	kmendpoint "k8s.io/kubernetes/contrib/mesos/pkg/service"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/pflag"
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
	fs.BoolVar(&s.UseHostPortEndpoints, "host_port_endpoints", s.UseHostPortEndpoints, "Map service endpoints to hostIP:hostPort instead of podIP:containerPort. Default true.")
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
			Addr:    net.JoinHostPort(s.Address.String(), strconv.Itoa(s.Port)),
			Handler: mux,
		}
		glog.Fatal(server.ListenAndServe())
	}()

	endpoints := s.createEndpointController(clientForUserAgentOrDie(*kubeconfig, "endpoint-controller"))
	go endpoints.Run(s.ConcurrentEndpointSyncs, util.NeverStop)

	go replicationcontroller.NewReplicationManager(clientForUserAgentOrDie(*kubeconfig, "replication-controller"), s.resyncPeriod, replicationcontroller.BurstReplicas).
		Run(s.ConcurrentRCSyncs, util.NeverStop)

	if s.TerminatedPodGCThreshold > 0 {
		go gc.New(clientForUserAgentOrDie(*kubeconfig, "garbage-collector"), s.resyncPeriod, s.TerminatedPodGCThreshold).
			Run(util.NeverStop)
	}

	//TODO(jdef) should eventually support more cloud providers here
	if s.CloudProvider != mesos.ProviderName {
		glog.Fatalf("Only provider %v is supported, you specified %v", mesos.ProviderName, s.CloudProvider)
	}
	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}

	nodeController := nodecontroller.NewNodeController(cloud, clientForUserAgentOrDie(*kubeconfig, "node-controller"),
		s.PodEvictionTimeout, util.NewTokenBucketRateLimiter(s.DeletingPodsQps, s.DeletingPodsBurst),
		util.NewTokenBucketRateLimiter(s.DeletingPodsQps, s.DeletingPodsBurst),
		s.NodeMonitorGracePeriod, s.NodeStartupGracePeriod, s.NodeMonitorPeriod, (*net.IPNet)(&s.ClusterCIDR), s.AllocateNodeCIDRs)
	nodeController.Run(s.NodeSyncPeriod)

	nodeStatusUpdaterController := node.NewStatusUpdater(clientForUserAgentOrDie(*kubeconfig, "node-status-controller"), s.NodeMonitorPeriod, time.Now)
	if err := nodeStatusUpdaterController.Run(util.NeverStop); err != nil {
		glog.Fatalf("Failed to start node status update controller: %v", err)
	}

	serviceController := servicecontroller.New(cloud, clientForUserAgentOrDie(*kubeconfig, "service-controller"), s.ClusterName)
	if err := serviceController.Run(s.ServiceSyncPeriod, s.NodeSyncPeriod); err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	}

	if s.AllocateNodeCIDRs {
		routes, ok := cloud.Routes()
		if !ok {
			glog.Fatal("Cloud provider must support routes if allocate-node-cidrs is set")
		}
		routeController := routecontroller.New(routes, clientForUserAgentOrDie(*kubeconfig, "route-controller"), s.ClusterName, (*net.IPNet)(&s.ClusterCIDR))
		routeController.Run(s.NodeSyncPeriod)
	}

	go resourcequotacontroller.NewResourceQuotaController(
		clientForUserAgentOrDie(*kubeconfig, "resource-quota-controller"), controller.StaticResyncPeriodFunc(s.ResourceQuotaSyncPeriod)).Run(s.ConcurrentResourceQuotaSyncs, util.NeverStop)

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

	namespaceController := namespacecontroller.NewNamespaceController(clientForUserAgentOrDie(*kubeconfig, "namespace-controller"), &unversioned.APIVersions{}, s.NamespaceSyncPeriod)
	namespaceController.Run()

	groupVersion := "extensions/v1beta1"
	resources, found := resourceMap[groupVersion]
	// TODO(k8s): this needs to be dynamic so users don't have to restart their controller manager if they change the apiserver
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
			go daemon.NewDaemonSetsController(clientForUserAgentOrDie(*kubeconfig, "daemon-set-controller"), s.resyncPeriod).
				Run(s.ConcurrentDSCSyncs, util.NeverStop)
		}

		if containsResource(resources, "jobs") {
			glog.Infof("Starting job controller")
			go job.NewJobController(clientForUserAgentOrDie(*kubeconfig, "job-controller"), s.resyncPeriod).
				Run(s.ConcurrentJobSyncs, util.NeverStop)
		}

		if containsResource(resources, "deployments") {
			glog.Infof("Starting deployment controller")
			go deployment.NewDeploymentController(clientsetForUserAgentOrDie(*kubeconfig, "deployment-controller"), s.resyncPeriod).
				Run(s.ConcurrentDeploymentSyncs, util.NeverStop)
		}
	}

	volumePlugins := kubecontrollermanager.ProbeRecyclableVolumePlugins(s.VolumeConfigFlags)
	provisioner, err := kubecontrollermanager.NewVolumeProvisioner(cloud, s.VolumeConfigFlags)
	if err != nil {
		glog.Fatal("A Provisioner could not be created, but one was expected. Provisioning will not work. This functionality is considered an early Alpha version.")
	}

	pvclaimBinder := persistentvolumecontroller.NewPersistentVolumeClaimBinder(clientForUserAgentOrDie(*kubeconfig, "persistent-volume-binder"), s.PVClaimBinderSyncPeriod)
	pvclaimBinder.Run()

	pvRecycler, err := persistentvolumecontroller.NewPersistentVolumeRecycler(clientForUserAgentOrDie(*kubeconfig, "persistent-volume-recycler"), s.PVClaimBinderSyncPeriod, kubecontrollermanager.ProbeRecyclableVolumePlugins(s.VolumeConfigFlags), cloud)
	if err != nil {
		glog.Fatalf("Failed to start persistent volume recycler: %+v", err)
	}
	pvRecycler.Run()

	if provisioner != nil {
		pvController, err := persistentvolumecontroller.NewPersistentVolumeProvisionerController(persistentvolumecontroller.NewControllerClient(clientForUserAgentOrDie(*kubeconfig, "persistent-volume-controller")), s.PVClaimBinderSyncPeriod, volumePlugins, provisioner, cloud)
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
			serviceaccountcontroller.NewTokensController(
				clientForUserAgentOrDie(*kubeconfig, "tokens-controller"),
				serviceaccountcontroller.TokensControllerOptions{
					TokenGenerator: serviceaccount.JWTTokenGenerator(privateKey),
					RootCA:         rootCA,
				},
			).Run()
		}
	}

	serviceaccountcontroller.NewServiceAccountsController(
		clientForUserAgentOrDie(*kubeconfig, "service-account-controller"),
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
	).Run()

	select {}
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

// clientsetForUserAgentOrDie is going to replace clientForUserAgentOrDie
func clientsetForUserAgentOrDie(config client.Config, userAgent string) release_1_1.Interface {
	fullUserAgent := client.DefaultKubernetesUserAgent() + "/" + userAgent
	config.UserAgent = fullUserAgent
	kubeClient, err := release_1_1.NewForConfig(&config)
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}
	return kubeClient
}

func (s *CMServer) createEndpointController(client *client.Client) kmendpoint.EndpointController {
	if s.UseHostPortEndpoints {
		glog.V(2).Infof("Creating hostIP:hostPort endpoint controller")
		return kmendpoint.NewEndpointController(client)
	}
	glog.V(2).Infof("Creating podIP:containerPort endpoint controller")
	stockEndpointController := endpointcontroller.NewEndpointController(client, s.resyncPeriod)
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
