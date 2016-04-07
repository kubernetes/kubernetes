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
	"os"
	"reflect"
	"strconv"
	"time"

	"k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/leaderelection"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controller/deployment"
	endpointcontroller "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
	"k8s.io/kubernetes/pkg/controller/gc"
	"k8s.io/kubernetes/pkg/controller/job"
	namespacecontroller "k8s.io/kubernetes/pkg/controller/namespace"
	nodecontroller "k8s.io/kubernetes/pkg/controller/node"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/persistentvolume"
	petset "k8s.io/kubernetes/pkg/controller/petset"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	replicaset "k8s.io/kubernetes/pkg/controller/replicaset"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	routecontroller "k8s.io/kubernetes/pkg/controller/route"
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/healthz"
	quotainstall "k8s.io/kubernetes/pkg/quota/install"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/util/crypto"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

const (
	// Jitter used when starting controller managers
	ControllerStartJitter = 1.0
)

// NewControllerManagerCommand creates a *cobra.Command object with default parameters
func NewControllerManagerCommand() *cobra.Command {
	s := options.NewCMServer()
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

func ResyncPeriod(s *options.CMServer) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(s.MinResyncPeriod.Nanoseconds()) * factor)
	}
}

// Run runs the CMServer.  This should never exit.
func Run(s *options.CMServer) error {
	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(s.KubeControllerManagerConfiguration)
	} else {
		glog.Errorf("unable to register configz: %s", err)
	}
	kubeconfig, err := clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
	if err != nil {
		return err
	}

	kubeconfig.ContentConfig.ContentType = s.ContentType
	// Override kubeconfig qps/burst settings from flags
	kubeconfig.QPS = s.KubeAPIQPS
	kubeconfig.Burst = int(s.KubeAPIBurst)

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
			Addr:    net.JoinHostPort(s.Address, strconv.Itoa(int(s.Port))),
			Handler: mux,
		}
		glog.Fatal(server.ListenAndServe())
	}()

	run := func(stop <-chan struct{}) {
		err := StartControllers(s, kubeClient, kubeconfig, stop)
		glog.Fatalf("error running controllers: %v", err)
		panic("unreachable")
	}

	if !s.LeaderElection.LeaderElect {
		run(nil)
		panic("unreachable")
	}

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "controller-manager"})

	id, err := os.Hostname()
	if err != nil {
		return err
	}

	leaderelection.RunOrDie(leaderelection.LeaderElectionConfig{
		EndpointsMeta: api.ObjectMeta{
			Namespace: "kube-system",
			Name:      "kube-controller-manager",
		},
		Client:        kubeClient,
		Identity:      id,
		EventRecorder: recorder,
		LeaseDuration: s.LeaderElection.LeaseDuration.Duration,
		RenewDeadline: s.LeaderElection.RenewDeadline.Duration,
		RetryPeriod:   s.LeaderElection.RetryPeriod.Duration,
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: run,
			OnStoppedLeading: func() {
				glog.Fatalf("leaderelection lost")
			},
		},
	})
	panic("unreachable")
}

func StartControllers(s *options.CMServer, kubeClient *client.Client, kubeconfig *restclient.Config, stop <-chan struct{}) error {
	podInformer := informers.CreateSharedIndexPodInformer(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "pod-informer")), ResyncPeriod(s)(), cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	informers := map[reflect.Type]framework.SharedIndexInformer{}
	informers[reflect.TypeOf(&api.Pod{})] = podInformer

	go endpointcontroller.NewEndpointController(podInformer, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "endpoint-controller"))).
		Run(int(s.ConcurrentEndpointSyncs), wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	go replicationcontroller.NewReplicationManager(
		podInformer,
		clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "replication-controller")),
		ResyncPeriod(s),
		replicationcontroller.BurstReplicas,
		int(s.LookupCacheSizeForRC),
	).Run(int(s.ConcurrentRCSyncs), wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	if s.TerminatedPodGCThreshold > 0 {
		go gc.New(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "garbage-collector")), ResyncPeriod(s), int(s.TerminatedPodGCThreshold)).
			Run(wait.NeverStop)
		time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
	}

	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}

	// this cidr has been validated already
	_, clusterCIDR, _ := net.ParseCIDR(s.ClusterCIDR)
	nodeController := nodecontroller.NewNodeController(cloud, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "node-controller")),
		s.PodEvictionTimeout.Duration, flowcontrol.NewTokenBucketRateLimiter(s.DeletingPodsQps, int(s.DeletingPodsBurst)),
		flowcontrol.NewTokenBucketRateLimiter(s.DeletingPodsQps, int(s.DeletingPodsBurst)),
		s.NodeMonitorGracePeriod.Duration, s.NodeStartupGracePeriod.Duration, s.NodeMonitorPeriod.Duration, clusterCIDR, s.AllocateNodeCIDRs)
	nodeController.Run(s.NodeSyncPeriod.Duration)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	serviceController := servicecontroller.New(cloud, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "service-controller")), s.ClusterName)
	if err := serviceController.Run(s.ServiceSyncPeriod.Duration, s.NodeSyncPeriod.Duration); err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	}
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	if s.AllocateNodeCIDRs {
		if cloud == nil {
			glog.Warning("allocate-node-cidrs is set, but no cloud provider specified. Will not manage routes.")
		} else if routes, ok := cloud.Routes(); !ok {
			glog.Warning("allocate-node-cidrs is set, but cloud provider does not support routes. Will not manage routes.")
		} else {
			routeController := routecontroller.New(routes, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "route-controller")), s.ClusterName, clusterCIDR)
			routeController.Run(s.NodeSyncPeriod.Duration)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	} else {
		glog.Infof("allocate-node-cidrs set to %v, node controller not creating routes", s.AllocateNodeCIDRs)
	}

	resourceQuotaControllerClient := clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "resourcequota-controller"))
	resourceQuotaRegistry := quotainstall.NewRegistry(resourceQuotaControllerClient)
	groupKindsToReplenish := []unversioned.GroupKind{
		api.Kind("Pod"),
		api.Kind("Service"),
		api.Kind("ReplicationController"),
		api.Kind("PersistentVolumeClaim"),
		api.Kind("Secret"),
		api.Kind("ConfigMap"),
	}
	resourceQuotaControllerOptions := &resourcequotacontroller.ResourceQuotaControllerOptions{
		KubeClient:                resourceQuotaControllerClient,
		ResyncPeriod:              controller.StaticResyncPeriodFunc(s.ResourceQuotaSyncPeriod.Duration),
		Registry:                  resourceQuotaRegistry,
		ControllerFactory:         resourcequotacontroller.NewReplenishmentControllerFactory(podInformer, resourceQuotaControllerClient),
		ReplenishmentResyncPeriod: ResyncPeriod(s),
		GroupKindsToReplenish:     groupKindsToReplenish,
	}
	go resourcequotacontroller.NewResourceQuotaController(resourceQuotaControllerOptions).Run(int(s.ConcurrentResourceQuotaSyncs), wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

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
	groupVersionResources, err := namespacecontroller.ServerPreferredNamespacedGroupVersionResources(namespaceKubeClient.Discovery())
	if err != nil {
		glog.Fatalf("Failed to get supported resources from server: %v", err)
	}
	namespaceController := namespacecontroller.NewNamespaceController(namespaceKubeClient, namespaceClientPool, groupVersionResources, s.NamespaceSyncPeriod.Duration, api.FinalizerKubernetes)
	go namespaceController.Run(int(s.ConcurrentNamespaceSyncs), wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	groupVersion := "extensions/v1beta1"
	resources, found := resourceMap[groupVersion]
	// TODO: this needs to be dynamic so users don't have to restart their controller manager if they change the apiserver
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
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}

		if containsResource(resources, "daemonsets") {
			glog.Infof("Starting daemon set controller")
			go daemon.NewDaemonSetsController(podInformer, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "daemon-set-controller")), ResyncPeriod(s), int(s.LookupCacheSizeForDaemonSet)).
				Run(int(s.ConcurrentDaemonSetSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}

		if containsResource(resources, "jobs") {
			glog.Infof("Starting job controller")
			go job.NewJobController(podInformer, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "job-controller"))).
				Run(int(s.ConcurrentJobSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}

		if containsResource(resources, "deployments") {
			glog.Infof("Starting deployment controller")
			go deployment.NewDeploymentController(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "deployment-controller")), ResyncPeriod(s)).
				Run(int(s.ConcurrentDeploymentSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}

		if containsResource(resources, "replicasets") {
			glog.Infof("Starting ReplicaSet controller")
			go replicaset.NewReplicaSetController(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "replicaset-controller")), ResyncPeriod(s), replicaset.BurstReplicas, int(s.LookupCacheSizeForRS)).
				Run(int(s.ConcurrentRSSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	groupVersion = "apps/v1alpha1"
	resources, found = resourceMap[groupVersion]
	glog.Infof("Attempting to start petset, full resource map %+v", resourceMap)
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "petsets") {
			glog.Infof("Starting PetSet controller")
			resyncPeriod := ResyncPeriod(s)()
			go petset.NewPetSetController(
				podInformer,
				// TODO: Switch to using clientset
				kubeClient,
				resyncPeriod,
			).Run(1, wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	volumePlugins := ProbeRecyclableVolumePlugins(s.VolumeConfiguration)
	provisioner, err := NewVolumeProvisioner(cloud, s.VolumeConfiguration)
	if err != nil {
		glog.Fatal("A Provisioner could not be created, but one was expected. Provisioning will not work. This functionality is considered an early Alpha version.")
	}

	pvclaimBinder := persistentvolumecontroller.NewPersistentVolumeClaimBinder(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "persistent-volume-binder")), s.PVClaimBinderSyncPeriod.Duration)
	pvclaimBinder.Run()
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	pvRecycler, err := persistentvolumecontroller.NewPersistentVolumeRecycler(
		clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "persistent-volume-recycler")),
		s.PVClaimBinderSyncPeriod.Duration,
		int(s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MaximumRetry),
		ProbeRecyclableVolumePlugins(s.VolumeConfiguration),
		cloud,
	)
	if err != nil {
		glog.Fatalf("Failed to start persistent volume recycler: %+v", err)
	}
	pvRecycler.Run()
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	if provisioner != nil {
		pvController, err := persistentvolumecontroller.NewPersistentVolumeProvisionerController(persistentvolumecontroller.NewControllerClient(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "persistent-volume-provisioner"))), s.PVClaimBinderSyncPeriod.Duration, s.ClusterName, volumePlugins, provisioner, cloud)
		if err != nil {
			glog.Fatalf("Failed to start persistent volume provisioner controller: %+v", err)
		}
		pvController.Run()
		time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
	}

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
			serviceaccountcontroller.NewTokensController(
				clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "tokens-controller")),
				serviceaccountcontroller.TokensControllerOptions{
					TokenGenerator: serviceaccount.JWTTokenGenerator(privateKey),
					RootCA:         rootCA,
				},
			).Run()
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	serviceaccountcontroller.NewServiceAccountsController(
		clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "service-account-controller")),
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
	).Run()
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	// run the shared informers
	for _, informer := range informers {
		go informer.Run(wait.NeverStop)
	}

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
