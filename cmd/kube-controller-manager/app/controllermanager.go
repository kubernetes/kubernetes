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
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/batch"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/leaderelection"
	"k8s.io/kubernetes/pkg/client/leaderelection/resourcelock"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	certcontroller "k8s.io/kubernetes/pkg/controller/certificates"
	"k8s.io/kubernetes/pkg/controller/cronjob"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controller/deployment"
	"k8s.io/kubernetes/pkg/controller/disruption"
	endpointcontroller "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
	"k8s.io/kubernetes/pkg/controller/informers"
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
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/pkg/healthz"
	quotainstall "k8s.io/kubernetes/pkg/quota/install"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/serviceaccount"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	"k8s.io/kubernetes/pkg/util/configz"
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
	kubeClient, err := clientset.NewForConfig(restclient.AddUserAgent(kubeconfig, "controller-manager"))
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}
	leaderElectionClient := clientset.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "leader-election"))

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

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "controller-manager"})

	run := func(stop <-chan struct{}) {
		rootClientBuilder := controller.SimpleControllerClientBuilder{
			ClientConfig: kubeconfig,
		}
		var clientBuilder controller.ControllerClientBuilder
		if len(s.ServiceAccountKeyFile) > 0 && s.UseServiceAccountCredentials {
			clientBuilder = controller.SAControllerClientBuilder{
				ClientConfig: restclient.AnonymousClientConfig(kubeconfig),
				CoreClient:   kubeClient.Core(),
				Namespace:    "kube-system",
			}
		} else {
			clientBuilder = rootClientBuilder
		}

		err := StartControllers(s, kubeconfig, rootClientBuilder, clientBuilder, stop, recorder)
		glog.Fatalf("error running controllers: %v", err)
		panic("unreachable")
	}

	if !s.LeaderElection.LeaderElect {
		run(nil)
		panic("unreachable")
	}

	id, err := os.Hostname()
	if err != nil {
		return err
	}

	// TODO: enable other lock types
	rl := resourcelock.EndpointsLock{
		EndpointsMeta: api.ObjectMeta{
			Namespace: "kube-system",
			Name:      "kube-controller-manager",
		},
		Client: leaderElectionClient,
		LockConfig: resourcelock.ResourceLockConfig{
			Identity:      id,
			EventRecorder: recorder,
		},
	}

	leaderelection.RunOrDie(leaderelection.LeaderElectionConfig{
		Lock:          &rl,
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

func StartControllers(s *options.CMServer, kubeconfig *restclient.Config, rootClientBuilder, clientBuilder controller.ControllerClientBuilder, stop <-chan struct{}, recorder record.EventRecorder) error {
	client := func(serviceAccountName string) clientset.Interface {
		return rootClientBuilder.ClientOrDie(serviceAccountName)
	}
	discoveryClient := client("controller-discovery").Discovery()
	sharedInformers := informers.NewSharedInformerFactory(client("shared-informers"), ResyncPeriod(s)())

	// always start the SA token controller first using a full-power client, since it needs to mint tokens for the rest
	if len(s.ServiceAccountKeyFile) > 0 {
		privateKey, err := serviceaccount.ReadPrivateKey(s.ServiceAccountKeyFile)
		if err != nil {
			return fmt.Errorf("Error reading key for service account token controller: %v", err)
		} else {
			var rootCA []byte
			if s.RootCAFile != "" {
				rootCA, err = ioutil.ReadFile(s.RootCAFile)
				if err != nil {
					return fmt.Errorf("error reading root-ca-file at %s: %v", s.RootCAFile, err)
				}
				if _, err := certutil.ParseCertsPEM(rootCA); err != nil {
					return fmt.Errorf("error parsing root-ca-file at %s: %v", s.RootCAFile, err)
				}
			} else {
				rootCA = kubeconfig.CAData
			}

			go serviceaccountcontroller.NewTokensController(
				rootClientBuilder.ClientOrDie("tokens-controller"),
				serviceaccountcontroller.TokensControllerOptions{
					TokenGenerator: serviceaccount.JWTTokenGenerator(privateKey),
					RootCA:         rootCA,
				},
			).Run(int(s.ConcurrentSATokenSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	go endpointcontroller.NewEndpointController(sharedInformers.Pods().Informer(), client("endpoint-controller")).
		Run(int(s.ConcurrentEndpointSyncs), wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	go replicationcontroller.NewReplicationManager(
		sharedInformers.Pods().Informer(),
		clientBuilder.ClientOrDie("replication-controller"),
		ResyncPeriod(s),
		replicationcontroller.BurstReplicas,
		int(s.LookupCacheSizeForRC),
		s.EnableGarbageCollector,
	).Run(int(s.ConcurrentRCSyncs), wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	go podgc.NewPodGC(client("pod-garbage-collector"), sharedInformers.Pods().Informer(),
		int(s.TerminatedPodGCThreshold)).Run(wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

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
	nodeController, err := nodecontroller.NewNodeController(
		sharedInformers.Pods(), sharedInformers.Nodes(), sharedInformers.DaemonSets(),
		cloud, client("node-controller"),
		s.PodEvictionTimeout.Duration, s.NodeEvictionRate, s.SecondaryNodeEvictionRate, s.LargeClusterSizeThreshold, s.UnhealthyZoneThreshold, s.NodeMonitorGracePeriod.Duration,
		s.NodeStartupGracePeriod.Duration, s.NodeMonitorPeriod.Duration, clusterCIDR, serviceCIDR,
		int(s.NodeCIDRMaskSize), s.AllocateNodeCIDRs)
	if err != nil {
		glog.Fatalf("Failed to initialize nodecontroller: %v", err)
	}
	nodeController.Run()
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	serviceController, err := servicecontroller.New(cloud, client("service-controller"), s.ClusterName)
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
			routeController := routecontroller.New(routes, client("route-controller"), s.ClusterName, clusterCIDR)
			routeController.Run(s.RouteReconciliationPeriod.Duration)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	} else {
		glog.Infof("Will not configure cloud provider routes for allocate-node-cidrs: %v, configure-cloud-routes: %v.", s.AllocateNodeCIDRs, s.ConfigureCloudRoutes)
	}

	resourceQuotaControllerClient := client("resourcequota-controller")
	resourceQuotaRegistry := quotainstall.NewRegistry(resourceQuotaControllerClient, sharedInformers)
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
		ControllerFactory:         resourcequotacontroller.NewReplenishmentControllerFactory(sharedInformers, resourceQuotaControllerClient),
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

	resourceMap, err := discoveryClient.ServerResources()
	if err != nil {
		glog.Fatalf("Failed to get supported resources from server: %v", err)
	}

	// TODO: should use a dynamic RESTMapper built from the discovery results.
	restMapper := registered.RESTMapper()

	// Find the list of namespaced resources via discovery that the namespace controller must manage
	namespaceKubeClient := client("namespace-controller")
	namespaceClientPool := dynamic.NewClientPool(restclient.AddUserAgent(kubeconfig, "namespace-controller"), restMapper, dynamic.LegacyAPIPathResolverFunc)
	// TODO: consider using a list-watch + cache here rather than polling
	var gvrFn func() ([]unversioned.GroupVersionResource, error)
	rsrcs, err := namespaceKubeClient.Discovery().ServerResources()
	if err != nil {
		glog.Fatalf("Failed to get group version resources: %v", err)
	}
	for _, rsrcList := range rsrcs {
		for ix := range rsrcList.APIResources {
			rsrc := &rsrcList.APIResources[ix]
			if rsrc.Kind == "ThirdPartyResource" {
				gvrFn = namespaceKubeClient.Discovery().ServerPreferredNamespacedResources
			}
		}
	}
	if gvrFn == nil {
		gvr, err := namespaceKubeClient.Discovery().ServerPreferredNamespacedResources()
		if err != nil {
			glog.Fatalf("Failed to get resources: %v", err)
		}
		gvrFn = func() ([]unversioned.GroupVersionResource, error) {
			return gvr, nil
		}
	}
	namespaceController := namespacecontroller.NewNamespaceController(namespaceKubeClient, namespaceClientPool, gvrFn, s.NamespaceSyncPeriod.Duration, api.FinalizerKubernetes)
	go namespaceController.Run(int(s.ConcurrentNamespaceSyncs), wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	groupVersion := "extensions/v1beta1"
	resources, found := resourceMap[groupVersion]
	// TODO: this needs to be dynamic so users don't have to restart their controller manager if they change the apiserver
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)

		if containsResource(resources, "daemonsets") {
			glog.Infof("Starting daemon set controller")
			go daemon.NewDaemonSetsController(sharedInformers.DaemonSets(), sharedInformers.Pods(), sharedInformers.Nodes(), client("daemon-set-controller"), int(s.LookupCacheSizeForDaemonSet)).
				Run(int(s.ConcurrentDaemonSetSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}

		if containsResource(resources, "jobs") {
			glog.Infof("Starting job controller")
			go job.NewJobController(sharedInformers.Pods().Informer(), sharedInformers.Jobs(), client("job-controller")).
				Run(int(s.ConcurrentJobSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}

		if containsResource(resources, "deployments") {
			glog.Infof("Starting deployment controller")
			go deployment.NewDeploymentController(sharedInformers.Deployments(), sharedInformers.ReplicaSets(), sharedInformers.Pods(), client("deployment-controller")).
				Run(int(s.ConcurrentDeploymentSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}

		if containsResource(resources, "replicasets") {
			glog.Infof("Starting ReplicaSet controller")
			go replicaset.NewReplicaSetController(sharedInformers.ReplicaSets(), sharedInformers.Pods(), client("replicaset-controller"), replicaset.BurstReplicas, int(s.LookupCacheSizeForRS), s.EnableGarbageCollector).
				Run(int(s.ConcurrentRSSyncs), wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	groupVersion = "autoscaling/v1"
	resources, found = resourceMap[groupVersion]
	glog.Infof("Attempting to start horizontal pod autoscaler controller, full resource map %+v", resourceMap)
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "horizontalpodautoscalers") {
			glog.Infof("Starting horizontal pod controller.")
			hpaClient := client("horizontal-pod-autoscaler")
			metricsClient := metrics.NewHeapsterMetricsClient(
				hpaClient,
				metrics.DefaultHeapsterNamespace,
				metrics.DefaultHeapsterScheme,
				metrics.DefaultHeapsterService,
				metrics.DefaultHeapsterPort,
			)
			replicaCalc := podautoscaler.NewReplicaCalculator(metricsClient, hpaClient.Core())
			go podautoscaler.NewHorizontalController(hpaClient.Core(), hpaClient.Extensions(), hpaClient.Autoscaling(), replicaCalc, s.HorizontalPodAutoscalerSyncPeriod.Duration).
				Run(wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	groupVersion = "policy/v1beta1"
	resources, found = resourceMap[groupVersion]
	glog.Infof("Attempting to start disruption controller, full resource map %+v", resourceMap)
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "poddisruptionbudgets") {
			glog.Infof("Starting disruption controller")
			go disruption.NewDisruptionController(sharedInformers.Pods().Informer(), client("disruption-controller")).Run(wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	groupVersion = "apps/v1beta1"
	resources, found = resourceMap[groupVersion]
	glog.Infof("Attempting to start statefulset, full resource map %+v", resourceMap)
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "statefulsets") {
			glog.Infof("Starting StatefulSet controller")
			resyncPeriod := ResyncPeriod(s)()
			go petset.NewStatefulSetController(
				sharedInformers.Pods().Informer(),
				client("statefulset-controller"),
				resyncPeriod,
			).Run(1, wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	groupVersion = "batch/v2alpha1"
	resources, found = resourceMap[groupVersion]
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "cronjobs") {
			glog.Infof("Starting cronjob controller")
			// // TODO: this is a temp fix for allowing kubeClient list v2alpha1 sj, should switch to using clientset
			kubeconfig.ContentConfig.GroupVersion = &unversioned.GroupVersion{Group: batch.GroupName, Version: "v2alpha1"}
			go cronjob.NewCronJobController(client("cronjob-controller")).
				Run(wait.NeverStop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	} else {
		glog.Infof("Not starting %s apis", groupVersion)
	}

	alphaProvisioner, err := NewAlphaVolumeProvisioner(cloud, s.VolumeConfiguration)
	if err != nil {
		glog.Fatalf("An backward-compatible provisioner could not be created: %v, but one was expected. Provisioning will not work. This functionality is considered an early Alpha version.", err)
	}
	params := persistentvolumecontroller.ControllerParameters{
		KubeClient:                client("persistent-volume-binder"),
		SyncPeriod:                s.PVClaimBinderSyncPeriod.Duration,
		AlphaProvisioner:          alphaProvisioner,
		VolumePlugins:             ProbeControllerVolumePlugins(cloud, s.VolumeConfiguration),
		Cloud:                     cloud,
		ClusterName:               s.ClusterName,
		EnableDynamicProvisioning: s.VolumeConfiguration.EnableDynamicProvisioning,
	}
	volumeController := persistentvolumecontroller.NewController(params)
	volumeController.Run(wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	attachDetachController, attachDetachControllerErr :=
		attachdetach.NewAttachDetachController(
			client("attachdetach-controller"),
			sharedInformers.Pods().Informer(),
			sharedInformers.Nodes().Informer(),
			sharedInformers.PersistentVolumeClaims().Informer(),
			sharedInformers.PersistentVolumes().Informer(),
			cloud,
			ProbeAttachableVolumePlugins(s.VolumeConfiguration),
			recorder)
	if attachDetachControllerErr != nil {
		glog.Fatalf("Failed to start attach/detach controller: %v", attachDetachControllerErr)
	}
	go attachDetachController.Run(wait.NeverStop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	groupVersion = "certificates.k8s.io/v1alpha1"
	resources, found = resourceMap[groupVersion]
	glog.Infof("Attempting to start certificates, full resource map %+v", resourceMap)
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "certificatesigningrequests") {
			glog.Infof("Starting certificate request controller")
			resyncPeriod := ResyncPeriod(s)()
			certController, err := certcontroller.NewCertificateController(
				client("certificate-controller"),
				resyncPeriod,
				s.ClusterSigningCertFile,
				s.ClusterSigningKeyFile,
				s.ApproveAllKubeletCSRsForGroup,
			)
			if err != nil {
				glog.Errorf("Failed to start certificate controller: %v", err)
			} else {
				go certController.Run(1, wait.NeverStop)
			}
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	go serviceaccountcontroller.NewServiceAccountsController(
		sharedInformers.ServiceAccounts(), sharedInformers.Namespaces(),
		client("service-account-controller"),
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
	).Run(1, stop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	if s.EnableGarbageCollector {
		gcClientset := client("generic-garbage-collector")
		groupVersionResources, err := gcClientset.Discovery().ServerPreferredResources()
		if err != nil {
			glog.Fatalf("Failed to get supported resources from server: %v", err)
		}

		config := restclient.AddUserAgent(kubeconfig, "generic-garbage-collector")
		config.ContentConfig.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: metaonly.NewMetadataCodecFactory()}
		metaOnlyClientPool := dynamic.NewClientPool(config, restMapper, dynamic.LegacyAPIPathResolverFunc)
		config.ContentConfig = dynamic.ContentConfig()
		clientPool := dynamic.NewClientPool(config, restMapper, dynamic.LegacyAPIPathResolverFunc)
		garbageCollector, err := garbagecollector.NewGarbageCollector(metaOnlyClientPool, clientPool, restMapper, groupVersionResources)
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
