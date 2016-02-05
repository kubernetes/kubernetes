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
	"strconv"
	"time"

	"k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/leaderelection"
	"k8s.io/kubernetes/pkg/client/record"
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
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
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

func StartControllers(s *options.CMServer, kubeClient *client.Client, kubeconfig *client.Config, stop <-chan struct{}) error {
	go endpointcontroller.NewEndpointController(clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "endpoint-controller")), ResyncPeriod(s)).
		Run(s.EndpointControllerOptions.ConcurrentEndpointSyncs, util.NeverStop)

	go replicationcontroller.NewReplicationManager(
		clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "replication-controller")),
		ResyncPeriod(s),
		replicationcontroller.BurstReplicas,
	).Run(s.ReplicationControllerOptions.ConcurrentRCSyncs, util.NeverStop)

	if s.GarbageCollectorOptions.TerminatedPodGCThreshold > 0 {
		go gc.New(clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "garbage-collector")), ResyncPeriod(s), s.GarbageCollectorOptions.TerminatedPodGCThreshold).
			Run(util.NeverStop)
	}

	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}

	nodeController := nodecontroller.NewNodeController(cloud, clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "node-controller")),
		s.NodeControllerOptions.PodEvictionTimeout, util.NewTokenBucketRateLimiter(s.NodeControllerOptions.DeletingPodsQps, s.NodeControllerOptions.DeletingPodsBurst),
		util.NewTokenBucketRateLimiter(s.NodeControllerOptions.DeletingPodsQps, s.NodeControllerOptions.DeletingPodsBurst),
		s.NodeControllerOptions.NodeMonitorGracePeriod, s.NodeControllerOptions.NodeStartupGracePeriod, s.NodeControllerOptions.NodeMonitorPeriod,
		&s.NodeControllerOptions.ClusterCIDR, s.NodeControllerOptions.AllocateNodeCIDRs,
	)
	nodeController.Run(s.NodeSyncPeriod)

	serviceController := servicecontroller.New(cloud, clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "service-controller")), s.ClusterName)
	if err := serviceController.Run(s.ServiceControllerOptions.ServiceSyncPeriod, s.NodeSyncPeriod); err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	}

	// TODO: Figure out what the relation between route controller and node controller should be.
	if s.NodeControllerOptions.AllocateNodeCIDRs {
		if cloud == nil {
			glog.Warning("allocate-node-cidrs is set, but no cloud provider specified. Will not manage routes.")
		} else if routes, ok := cloud.Routes(); !ok {
			glog.Warning("allocate-node-cidrs is set, but cloud provider does not support routes. Will not manage routes.")
		} else {
			routeController := routecontroller.New(
				routes,
				clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "route-controller")),
				s.ClusterName,
				&s.NodeControllerOptions.ClusterCIDR,
			)
			routeController.Run(s.NodeSyncPeriod)
		}
	} else {
		glog.Infof("allocate-node-cidrs set to %v, node controller not creating routes", s.NodeControllerOptions.AllocateNodeCIDRs)
	}

	go resourcequotacontroller.NewResourceQuotaController(
		clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "resourcequota-controller")),
		controller.StaticResyncPeriodFunc(s.ResourceQuotaControllerOptions.ResourceQuotaSyncPeriod)).Run(s.ResourceQuotaControllerOptions.ConcurrentResourceQuotaSyncs, util.NeverStop)

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

	namespacecontroller.NewNamespaceController(
		clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "namespace-controller")),
		versions,
		s.NamespaceControllerOptions.NamespaceSyncPeriod).Run()

	groupVersion := "extensions/v1beta1"
	resources, found := resourceMap[groupVersion]
	// TODO: this needs to be dynamic so users don't have to restart their controller manager if they change the apiserver
	if containsVersion(versions, groupVersion) && found {
		glog.Infof("Starting %s apis", groupVersion)
		if containsResource(resources, "horizontalpodautoscalers") {
			glog.Infof("Starting horizontal pod controller.")
			hpaClient := clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "horizontal-pod-autoscaler"))
			metricsClient := metrics.NewHeapsterMetricsClient(
				hpaClient,
				metrics.DefaultHeapsterNamespace,
				metrics.DefaultHeapsterScheme,
				metrics.DefaultHeapsterService,
				metrics.DefaultHeapsterPort,
			)
			// TODO: rename controller/podautoscaler into controller/horizontal
			podautoscaler.NewHorizontalController(hpaClient.Core(), hpaClient.Extensions(), hpaClient, metricsClient).
				Run(s.PodAutoscalerOptions.HorizontalPodAutoscalerSyncPeriod)
		}

		if containsResource(resources, "daemonsets") {
			glog.Infof("Starting daemon set controller")
			go daemon.NewDaemonSetsController(clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "daemon-set-controller")), ResyncPeriod(s)).
				Run(s.DaemonControllerOptions.ConcurrentDSCSyncs, util.NeverStop)
		}

		if containsResource(resources, "jobs") {
			glog.Infof("Starting job controller")
			go job.NewJobController(clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "job-controller")), ResyncPeriod(s)).
				Run(s.JobControllerOptions.ConcurrentJobSyncs, util.NeverStop)
		}

		if containsResource(resources, "deployments") {
			glog.Infof("Starting deployment controller")
			go deployment.NewDeploymentController(clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "deployment-controller")), ResyncPeriod(s)).
				Run(s.DeploymentControllerOptions.ConcurrentDeploymentSyncs, util.NeverStop)
		}
	}

	volumePlugins := ProbeRecyclableVolumePlugins(s.PersistentVolumeControllerOptions.VolumeConfigFlags)
	provisioner, err := NewVolumeProvisioner(cloud, s.PersistentVolumeControllerOptions.VolumeConfigFlags)
	if err != nil {
		glog.Fatal("A Provisioner could not be created, but one was expected. Provisioning will not work. This functionality is considered an early Alpha version.")
	}

	pvclaimBinder := persistentvolumecontroller.NewPersistentVolumeClaimBinder(
		clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "persistent-volume-binder")),
		s.PersistentVolumeControllerOptions.PVClaimBinderSyncPeriod,
	)
	pvclaimBinder.Run()

	pvRecycler, err := persistentvolumecontroller.NewPersistentVolumeRecycler(
		clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "persistent-volume-recycler")),
		s.PersistentVolumeControllerOptions.PVClaimBinderSyncPeriod,
		s.PersistentVolumeControllerOptions.VolumeConfigFlags.PersistentVolumeRecyclerMaximumRetry,
		ProbeRecyclableVolumePlugins(s.PersistentVolumeControllerOptions.VolumeConfigFlags),
		cloud,
	)
	if err != nil {
		glog.Fatalf("Failed to start persistent volume recycler: %+v", err)
	}
	pvRecycler.Run()

	if provisioner != nil {
		pvController, err := persistentvolumecontroller.NewPersistentVolumeProvisionerController(persistentvolumecontroller.NewControllerClient(clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "persistent-volume-provisioner"))), s.PersistentVolumeControllerOptions.PVClaimBinderSyncPeriod, volumePlugins, provisioner, cloud)
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

	if len(s.ServiceAccountControllerOptions.ServiceAccountKeyFile) > 0 {
		privateKey, err := serviceaccount.ReadPrivateKey(s.ServiceAccountControllerOptions.ServiceAccountKeyFile)
		if err != nil {
			glog.Errorf("Error reading key for service account token controller: %v", err)
		} else {
			serviceaccountcontroller.NewTokensController(
				clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "tokens-controller")),
				serviceaccountcontroller.TokensControllerOptions{
					TokenGenerator: serviceaccount.JWTTokenGenerator(privateKey),
					RootCA:         rootCA,
				},
			).Run()
		}
	}

	serviceaccountcontroller.NewServiceAccountsController(
		clientset.NewForConfigOrDie(client.AddUserAgent(kubeconfig, "service-account-controller")),
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
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
