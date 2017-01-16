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

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/healthz"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/leaderelection"
	"k8s.io/kubernetes/pkg/client/leaderelection/resourcelock"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	nodecontroller "k8s.io/kubernetes/pkg/controller/node"
	routecontroller "k8s.io/kubernetes/pkg/controller/route"
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/pkg/serviceaccount"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	"k8s.io/kubernetes/pkg/util/configz"

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
	s.AddFlags(pflag.CommandLine, KnownControllers(), ControllersDisabledByDefault.List())
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
	if err := s.Validate(KnownControllers(), ControllersDisabledByDefault.List()); err != nil {
		return err
	}

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
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.Core().Events("")})
	recorder := eventBroadcaster.NewRecorder(v1.EventSource{Component: "controller-manager"})

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

		err := StartControllers(newControllerInitializers(), s, rootClientBuilder, clientBuilder, stop)
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
		EndpointsMeta: v1.ObjectMeta{
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

type ControllerContext struct {
	// ClientBuilder will provide a client for this controller to use
	ClientBuilder controller.ControllerClientBuilder

	// InformerFactory gives access to informers for the controller
	InformerFactory informers.SharedInformerFactory

	// Options provides access to init options for a given controller
	Options options.CMServer

	// AvailableResources is a map listing currently available resources
	AvailableResources map[schema.GroupVersionResource]bool

	// Stop is the stop channel
	Stop <-chan struct{}
}

func (c ControllerContext) IsControllerEnabled(name string) bool {
	return IsControllerEnabled(name, ControllersDisabledByDefault, c.Options.Controllers...)
}

func IsControllerEnabled(name string, disabledByDefaultControllers sets.String, controllers ...string) bool {
	hasStar := false
	for _, controller := range controllers {
		if controller == name {
			return true
		}
		if controller == "-"+name {
			return false
		}
		if controller == "*" {
			hasStar = true
		}
	}
	// if we get here, there was no explicit choice
	if !hasStar {
		// nothing on by default
		return false
	}
	if disabledByDefaultControllers.Has(name) {
		return false
	}

	return true
}

// InitFunc is used to launch a particular controller.  It may run additional "should I activate checks".
// Any error returned will cause the controller process to `Fatal`
// The bool indicates whether the controller was enabled.
type InitFunc func(ctx ControllerContext) (bool, error)

func KnownControllers() []string {
	return sets.StringKeySet(newControllerInitializers()).List()
}

var ControllersDisabledByDefault = sets.NewString()

func newControllerInitializers() map[string]InitFunc {
	controllers := map[string]InitFunc{}
	controllers["endpoint"] = startEndpointController
	controllers["replicationcontroller"] = startReplicationController
	controllers["podgc"] = startPodGCController
	controllers["resourcequota"] = startResourceQuotaController
	controllers["namespace"] = startNamespaceController
	controllers["serviceaccount"] = startServiceAccountController
	controllers["garbagecollector"] = startGarbageCollectorController
	controllers["daemonset"] = startDaemonSetController
	controllers["job"] = startJobController
	controllers["deployment"] = startDeploymentController
	controllers["replicaset"] = startReplicaSetController
	controllers["horizontalpodautoscaling"] = startHPAController
	controllers["disruption"] = startDisruptionController
	controllers["statefuleset"] = startStatefulSetController
	controllers["cronjob"] = startCronJobController
	controllers["certificatesigningrequests"] = startCSRController

	return controllers
}

// TODO: In general, any controller checking this needs to be dynamic so
//  users don't have to restart their controller manager if they change the apiserver.
func getAvailableResources(clientBuilder controller.ControllerClientBuilder) (map[schema.GroupVersionResource]bool, error) {
	var discoveryClient discovery.DiscoveryInterface

	// If apiserver is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and controller manager at the same time.
	err := wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		client, err := clientBuilder.Client("controller-discovery")
		if err != nil {
			glog.Errorf("Failed to get api versions from server: %v", err)
			return false, nil
		}

		discoveryClient = client.Discovery()
		return true, nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get api versions from server: %v", err)
	}

	resourceMap, err := discoveryClient.ServerResources()
	if err != nil {
		return nil, fmt.Errorf("failed to get supported resources from server: %v", err)
	}

	allResources := map[schema.GroupVersionResource]bool{}
	for _, apiResourceList := range resourceMap {
		version, err := schema.ParseGroupVersion(apiResourceList.GroupVersion)
		if err != nil {
			return nil, err
		}
		for _, apiResource := range apiResourceList.APIResources {
			allResources[version.WithResource(apiResource.Name)] = true
		}
	}

	return allResources, nil
}

func StartControllers(controllers map[string]InitFunc, s *options.CMServer, rootClientBuilder, clientBuilder controller.ControllerClientBuilder, stop <-chan struct{}) error {
	sharedInformers := informers.NewSharedInformerFactory(rootClientBuilder.ClientOrDie("shared-informers"), nil, ResyncPeriod(s)())

	// always start the SA token controller first using a full-power client, since it needs to mint tokens for the rest
	if len(s.ServiceAccountKeyFile) > 0 {
		privateKey, err := serviceaccount.ReadPrivateKey(s.ServiceAccountKeyFile)
		if err != nil {
			return fmt.Errorf("error reading key for service account token controller: %v", err)
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
				rootCA = rootClientBuilder.ConfigOrDie("tokens-controller").CAData
			}

			go serviceaccountcontroller.NewTokensController(
				rootClientBuilder.ClientOrDie("tokens-controller"),
				serviceaccountcontroller.TokensControllerOptions{
					TokenGenerator: serviceaccount.JWTTokenGenerator(privateKey),
					RootCA:         rootCA,
				},
			).Run(int(s.ConcurrentSATokenSyncs), stop)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	}

	availableResources, err := getAvailableResources(clientBuilder)
	if err != nil {
		return err
	}

	ctx := ControllerContext{
		ClientBuilder:      clientBuilder,
		InformerFactory:    sharedInformers,
		Options:            *s,
		AvailableResources: availableResources,
		Stop:               stop,
	}

	for controllerName, initFn := range controllers {
		if !ctx.IsControllerEnabled(controllerName) {
			glog.Warningf("%q is disabled", controllerName)
			continue
		}

		time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

		glog.V(1).Infof("Starting %q", controllerName)
		started, err := initFn(ctx)
		if err != nil {
			glog.Errorf("Error starting %q", controllerName)
			return err
		}
		if !started {
			glog.Warningf("Skipping %q", controllerName)
			continue
		}
		glog.Infof("Started %q", controllerName)
	}

	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		return fmt.Errorf("cloud provider could not be initialized: %v", err)
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
		cloud, clientBuilder.ClientOrDie("node-controller"),
		s.PodEvictionTimeout.Duration, s.NodeEvictionRate, s.SecondaryNodeEvictionRate, s.LargeClusterSizeThreshold, s.UnhealthyZoneThreshold, s.NodeMonitorGracePeriod.Duration,
		s.NodeStartupGracePeriod.Duration, s.NodeMonitorPeriod.Duration, clusterCIDR, serviceCIDR,
		int(s.NodeCIDRMaskSize), s.AllocateNodeCIDRs)
	if err != nil {
		return fmt.Errorf("failed to initialize nodecontroller: %v", err)
	}
	nodeController.Run()
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	serviceController, err := servicecontroller.New(cloud, clientBuilder.ClientOrDie("service-controller"), s.ClusterName)
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
			routeController := routecontroller.New(routes, clientBuilder.ClientOrDie("route-controller"), s.ClusterName, clusterCIDR)
			routeController.Run(s.RouteReconciliationPeriod.Duration)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	} else {
		glog.Infof("Will not configure cloud provider routes for allocate-node-cidrs: %v, configure-cloud-routes: %v.", s.AllocateNodeCIDRs, s.ConfigureCloudRoutes)
	}

	alphaProvisioner, err := NewAlphaVolumeProvisioner(cloud, s.VolumeConfiguration)
	if err != nil {
		return fmt.Errorf("an backward-compatible provisioner could not be created: %v, but one was expected. Provisioning will not work. This functionality is considered an early Alpha version.", err)
	}
	params := persistentvolumecontroller.ControllerParameters{
		KubeClient:                clientBuilder.ClientOrDie("persistent-volume-binder"),
		SyncPeriod:                s.PVClaimBinderSyncPeriod.Duration,
		AlphaProvisioner:          alphaProvisioner,
		VolumePlugins:             ProbeControllerVolumePlugins(cloud, s.VolumeConfiguration),
		Cloud:                     cloud,
		ClusterName:               s.ClusterName,
		EnableDynamicProvisioning: s.VolumeConfiguration.EnableDynamicProvisioning,
	}
	volumeController := persistentvolumecontroller.NewController(params)
	go volumeController.Run(stop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	if s.ReconcilerSyncLoopPeriod.Duration < time.Second {
		return fmt.Errorf("Duration time must be greater than one second as set via command line option reconcile-sync-loop-period.")
	}

	attachDetachController, attachDetachControllerErr :=
		attachdetach.NewAttachDetachController(
			clientBuilder.ClientOrDie("attachdetach-controller"),
			sharedInformers.Pods().Informer(),
			sharedInformers.Nodes().Informer(),
			sharedInformers.PersistentVolumeClaims().Informer(),
			sharedInformers.PersistentVolumes().Informer(),
			cloud,
			ProbeAttachableVolumePlugins(s.VolumeConfiguration),
			s.DisableAttachDetachReconcilerSync,
			s.ReconcilerSyncLoopPeriod.Duration,
		)
	if attachDetachControllerErr != nil {
		return fmt.Errorf("failed to start attach/detach controller: %v", attachDetachControllerErr)
	}
	go attachDetachController.Run(stop)
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	sharedInformers.Start(stop)

	select {}
}
