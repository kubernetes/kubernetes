/*
Copyright 2014 The Kubernetes Authors.

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
	"strconv"
	"time"

	"k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/leaderelection"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	certcontroller "k8s.io/kubernetes/pkg/controller/certificates"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controller/deployment"
	"k8s.io/kubernetes/pkg/controller/disruption"
	endpointcontroller "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
	"k8s.io/kubernetes/pkg/controller/job"
	namespacecontroller "k8s.io/kubernetes/pkg/controller/namespace"
	nodecontroller "k8s.io/kubernetes/pkg/controller/node"
	petset "k8s.io/kubernetes/pkg/controller/petset"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/pkg/controller/podgc"
	replicaset "k8s.io/kubernetes/pkg/controller/replicaset"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	routecontroller "k8s.io/kubernetes/pkg/controller/route"
	"k8s.io/kubernetes/pkg/controller/scheduledjob"
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/pkg/healthz"
	quotainstall "k8s.io/kubernetes/pkg/quota/install"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/util/crypto"
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
		configz.InstallHandler(mux)
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
	sharedInformers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "shared-informers")), ResyncPeriod(s)())

	go endpointcontroller.NewEndpointController(sharedInformers.Pods().Informer(), clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "endpoint-controller"))).
		Run(int(s.ConcurrentEndpointSyncs), wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	go replicationcontroller.NewReplicationManager(
		sharedInformers.Pods().Informer(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "replication-controller")),
		ResyncPeriod(s),
		replicationcontroller.BurstReplicas,
		int(s.LookupCacheSizeForRC),
		s.EnableGarbageCollector,
	).Run(int(s.ConcurrentRCSyncs), wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	if s.TerminatedPodGCThreshold > 0 {
		go podgc.New(clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "pod-garbage-collector")), ResyncPeriod(s), int(s.TerminatedPodGCThreshold)).
			Run(wait.NeverStop)
		time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
	}

	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}

	_, clusterCIDR, err := net.ParseCIDR(s.ClusterCIDR)
	if err != nil {
		glog.Warningf("Unsuccessful parsing of cluster CIDR %v: %v", s.ClusterCIDR, err)
	}
	_, serviceCIDR, err := net.ParseCIDR(s.ServiceCIDR)
	if err != nil {
		glog.Warningf("Unsuccessful parsing of service CIDR %v: %v", s.ServiceCIDR, err)
	}
	nodeController, err := nodecontroller.NewNodeController(sharedInformers.Pods().Informer(), cloud, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "node-controller")),
		s.PodEvictionTimeout.Duration, s.DeletingPodsQps, s.NodeMonitorGracePeriod.Duration,
		s.NodeStartupGracePeriod.Duration, s.NodeMonitorPeriod.Duration, clusterCIDR, serviceCIDR,
		int(s.NodeCIDRMaskSize), s.AllocateNodeCIDRs)
	if err != nil {
		glog.Fatalf("Failed to initialize nodecontroller: %v", err)
	}
	nodeController.Run(s.NodeSyncPeriod.Duration)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	serviceController, err := servicecontroller.New(cloud, clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "service-controller")), s.ClusterName)
	if err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	} else {
		serviceController.Run(int(s.ConcurrentServiceSyncs))
	}
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

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
		ControllerFactory:         resourcequotacontroller.NewReplenishmentControllerFactory(sharedInformers.Pods().Informer(), resourceQuotaControllerClient),
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
	groupVersionResources, err := namespaceKubeClient.Discovery().ServerPreferredNamespacedResources()
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
			go daemon.NewDaemonSetsController(sharedInformers.Pods().Informer(), clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "daemon-set-controller")), ResyncPeriod(s), int(s.LookupCacheSizeForDaemonSet)).
				Run(int(s.ConcurrentDaemonSetSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}

		if containsResource(resources, "jobs") {
			glog.Infof("Starting job controller")
			go job.NewJobController(sharedInformers.Pods().Informer(), clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "job-controller"))).
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
			go replicaset.NewReplicaSetController(sharedInformers.Pods().Informer(), clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "replicaset-controller")), ResyncPeriod(s), replicaset.BurstReplicas, int(s.LookupCacheSizeForRS), s.EnableGarbageCollector).
				Run(int(s.ConcurrentRSSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	groupVersion = "policy/v1alpha1"
	resources, found = resourceMap[groupVersion]
	glog.Infof("Attempting to start disruption controller, full resource map %+v", resourceMap)
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "poddisruptionbudgets") {
			glog.Infof("Starting disruption controller")
			go disruption.NewDisruptionController(sharedInformers.Pods().Informer(), kubeClient).Run(wait.NeverStop)
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
				sharedInformers.Pods().Informer(),
				// TODO: Switch to using clientset
				kubeClient,
				resyncPeriod,
			).Run(1, wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	groupVersion = "batch/v2alpha1"
	resources, found = resourceMap[groupVersion]
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "scheduledjobs") {
			glog.Infof("Starting scheduledjob controller")
			// TODO: this is a temp fix for allowing kubeClient list v2alpha1 sj, should switch to using clientset
			kubeconfig.ContentConfig.GroupVersion = &unversioned.GroupVersion{Group: batch.GroupName, Version: "v2alpha1"}
			kubeClient, err := client.New(kubeconfig)
			if err != nil {
				glog.Fatalf("Invalid API configuration: %v", err)
			}
			go scheduledjob.NewScheduledJobController(kubeClient).
				Run(wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	} else {
		glog.Infof("Not starting %s apis", groupVersion)
	}

	provisioner, err := NewVolumeProvisioner(cloud, s.VolumeConfiguration)
	if err != nil {
		glog.Fatalf("A Provisioner could not be created: %v, but one was expected. Provisioning will not work. This functionality is considered an early Alpha version.", err)
	}

	volumeController := persistentvolumecontroller.NewPersistentVolumeController(
		clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "persistent-volume-binder")),
		s.PVClaimBinderSyncPeriod.Duration,
		provisioner,
		ProbeRecyclableVolumePlugins(s.VolumeConfiguration),
		cloud,
		s.ClusterName,
		nil, nil, nil,
		s.VolumeConfiguration.EnableDynamicProvisioning,
	)
	volumeController.Run()
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	attachDetachController, attachDetachControllerErr :=
		attachdetach.NewAttachDetachController(
			clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "attachdetach-controller")),
			sharedInformers.Pods().Informer(),
			sharedInformers.Nodes().Informer(),
			sharedInformers.PersistentVolumeClaims().Informer(),
			sharedInformers.PersistentVolumes().Informer(),
			cloud,
			ProbeAttachableVolumePlugins(s.VolumeConfiguration))
	if attachDetachControllerErr != nil {
		glog.Fatalf("Failed to start attach/detach controller: %v", attachDetachControllerErr)
	}
	go attachDetachController.Run(wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	groupVersion = "certificates/v1alpha1"
	resources, found = resourceMap[groupVersion]
	glog.Infof("Attempting to start certificates, full resource map %+v", resourceMap)
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "certificatesigningrequests") {
			glog.Infof("Starting certificate request controller")
			resyncPeriod := ResyncPeriod(s)()
			certController, err := certcontroller.NewCertificateController(
				clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "certificate-controller")),
				resyncPeriod,
				s.ClusterSigningCertFile,
				s.ClusterSigningKeyFile,
			)
			if err != nil {
				glog.Errorf("Failed to start certificate controller: %v", err)
			} else {
				go certController.Run(1, wait.NeverStop)
			}
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
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
			go serviceaccountcontroller.NewTokensController(
				clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "tokens-controller")),
				serviceaccountcontroller.TokensControllerOptions{
					TokenGenerator: serviceaccount.JWTTokenGenerator(privateKey),
					RootCA:         rootCA,
				},
			).Run(int(s.ConcurrentSATokenSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	serviceaccountcontroller.NewServiceAccountsController(
		clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "service-account-controller")),
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
	).Run()
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	if s.EnableGarbageCollector {
		gcClientset := clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "generic-garbage-collector"))
		groupVersionResources, err := gcClientset.Discovery().ServerPreferredResources()
		if err != nil {
			glog.Fatalf("Failed to get supported resources from server: %v", err)
		}
		config := restclient.AddUserAgent(kubeconfig, "generic-garbage-collector")
		config.ContentConfig.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: metaonly.NewMetadataCodecFactory()}
		metaOnlyClientPool := dynamic.NewClientPool(config, dynamic.LegacyAPIPathResolverFunc)
		config.ContentConfig.NegotiatedSerializer = nil
		clientPool := dynamic.NewClientPool(config, dynamic.LegacyAPIPathResolverFunc)
		garbageCollector, err := garbagecollector.NewGarbageCollector(metaOnlyClientPool, clientPool, groupVersionResources)
		if err != nil {
			glog.Errorf("Failed to start the generic garbage collector: %v", err)
		} else {
			workers := int(s.ConcurrentGCSyncs)
			go garbageCollector.Run(workers, wait.NeverStop)
		}
	}

	sharedInformers.Start(stop)

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
