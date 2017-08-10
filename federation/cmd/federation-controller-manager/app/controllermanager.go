/*
Copyright 2016 The Kubernetes Authors.

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
// components.  This includes cluster controller

package app

import (
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	goruntime "runtime"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	utilflag "k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/tools/record"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/cmd/federation-controller-manager/app/options"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	clustercontroller "k8s.io/kubernetes/federation/pkg/federation-controller/cluster"
	ingresscontroller "k8s.io/kubernetes/federation/pkg/federation-controller/ingress"
	jobcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/job"
	servicecontroller "k8s.io/kubernetes/federation/pkg/federation-controller/service"
	servicednscontroller "k8s.io/kubernetes/federation/pkg/federation-controller/service/dns"
	synccontroller "k8s.io/kubernetes/federation/pkg/federation-controller/sync"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/version"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
)

const (
	apiserverWaitTimeout   = 2 * time.Minute
	apiserverRetryInterval = 2 * time.Second
)

// NewControllerManagerCommand creates a *cobra.Command object with default parameters
func NewControllerManagerCommand() *cobra.Command {
	s := options.NewCMServer()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "federation-controller-manager",
		Long: `The federation controller manager is a daemon that embeds
the core control loops shipped with federation. In applications of robotics and
automation, a control loop is a non-terminating loop that regulates the state of
the system. In federation, a controller is a control loop that watches the shared
state of the federation cluster through the apiserver and makes changes attempting
to move the current state towards the desired state. Examples of controllers that
ship with federation today is the cluster controller.`,
		Run: func(cmd *cobra.Command, args []string) {
		},
	}

	return cmd
}

// Run runs the CMServer.  This should never exit.
func Run(s *options.CMServer) error {
	glog.Infof("%+v", version.Get())
	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(s.ControllerManagerConfiguration)
	} else {
		glog.Errorf("unable to register configz: %s", err)
	}

	restClientCfg, err := clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
	if err != nil || restClientCfg == nil {
		glog.V(2).Infof("Couldn't build the rest client config from flags: %v", err)
		return err
	}

	// Override restClientCfg qps/burst settings from flags
	restClientCfg.QPS = s.APIServerQPS
	restClientCfg.Burst = s.APIServerBurst

	go func() {
		mux := http.NewServeMux()
		healthz.InstallHandler(mux)
		if s.EnableProfiling {
			mux.HandleFunc("/debug/pprof/", pprof.Index)
			mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
			mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
			mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
			if s.EnableContentionProfiling {
				goruntime.SetBlockProfileRate(1)
			}
		}
		mux.Handle("/metrics", prometheus.Handler())

		server := &http.Server{
			Addr:    net.JoinHostPort(s.Address, strconv.Itoa(s.Port)),
			Handler: mux,
		}
		glog.Fatal(server.ListenAndServe())
	}()

	federationClientset, err := federationclientset.NewForConfig(restclient.AddUserAgent(restClientCfg, "federation-controller-manager"))
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}

	run := func(stop <-chan struct{}) {
		err := StartControllers(s, restClientCfg, stop)
		glog.Fatalf("error running controllers: %v", err)
		panic("unreachable")
	}

	if !s.LeaderElection.LeaderElect {
		run(nil)
		// unreachable
	}

	if err := ensureFederationNamespace(federationClientset, s.FederationOnlyNamespace); err != nil {
		glog.Fatalf("Failed to ensure federation only namespace %s: %v", s.FederationOnlyNamespace, err)
	}

	leaderElectionClient := kubernetes.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, "leader-election"))
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(federationClientset))
	recorder := eventBroadcaster.NewRecorder(api.Scheme, v1.EventSource{Component: "controller-manager"})

	id, err := os.Hostname()
	if err != nil {
		return err
	}

	rl := resourcelock.ConfigMapLock{
		ConfigMapMeta: metav1.ObjectMeta{
			Namespace: s.FederationOnlyNamespace,
			Name:      "federation-controller-manager-leader-election",
			Annotations: map[string]string{
				federationapi.FederationClusterSelectorAnnotation: federationapi.FederationOnlyClusterSelector,
			}},
		Client: leaderElectionClient.CoreV1(),
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

func StartControllers(s *options.CMServer, restClientCfg *restclient.Config, stopChan <-chan struct{}) error {
	minimizeLatency := false

	discoveryClient := discovery.NewDiscoveryClientForConfigOrDie(restClientCfg)
	serverResources, err := discoveryClient.ServerResources()
	if err != nil {
		glog.Fatalf("Could not find resources from API Server: %v", err)
	}

	clustercontroller.StartClusterController(restClientCfg, stopChan, s.ClusterMonitorPeriod.Duration)

	if controllerEnabled(s.Controllers, serverResources, servicecontroller.ControllerName, servicecontroller.RequiredResources, true) {
		if controllerEnabled(s.Controllers, serverResources, servicednscontroller.ControllerName, servicecontroller.RequiredResources, true) {
			serviceDNScontrollerClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, servicednscontroller.UserAgentName))
			serviceDNSController, err := servicednscontroller.NewServiceDNSController(serviceDNScontrollerClientset, s.DnsProvider, s.DnsConfigFile, s.FederationName, s.ServiceDnsSuffix, s.ZoneName, s.ZoneID)
			if err != nil {
				glog.Fatalf("Failed to start service dns controller: %v", err)
			} else {
				go serviceDNSController.DNSControllerRun(s.ConcurrentServiceSyncs, wait.NeverStop)
			}
		}

		glog.V(3).Infof("Loading client config for service controller %q", servicecontroller.UserAgentName)
		scClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, servicecontroller.UserAgentName))
		serviceController := servicecontroller.New(scClientset)
		go serviceController.Run(s.ConcurrentServiceSyncs, stopChan)
	}

	adapterSpecificArgs := make(map[string]interface{})
	adapterSpecificArgs[federatedtypes.HpaKind] = &s.HpaScaleForbiddenWindow
	for kind, federatedType := range federatedtypes.FederatedTypes() {
		if controllerEnabled(s.Controllers, serverResources, federatedType.ControllerName, federatedType.RequiredResources, true) {
			synccontroller.StartFederationSyncController(kind, federatedType.AdapterFactory, restClientCfg, stopChan, minimizeLatency, adapterSpecificArgs)
		}
	}

	if controllerEnabled(s.Controllers, serverResources, jobcontroller.ControllerName, jobcontroller.RequiredResources, true) {
		glog.V(3).Infof("Loading client config for job controller %q", jobcontroller.UserAgentName)
		jobClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, jobcontroller.UserAgentName))
		jobController := jobcontroller.NewJobController(jobClientset)
		glog.V(3).Infof("Running job controller")
		go jobController.Run(s.ConcurrentJobSyncs, wait.NeverStop)
	}

	if controllerEnabled(s.Controllers, serverResources, ingresscontroller.ControllerName, ingresscontroller.RequiredResources, true) {
		glog.V(3).Infof("Loading client config for ingress controller %q", ingresscontroller.UserAgentName)
		ingClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, ingresscontroller.UserAgentName))
		ingressController := ingresscontroller.NewIngressController(ingClientset)
		glog.V(3).Infof("Running ingress controller")
		ingressController.Run(stopChan)
	}

	select {}
}

func controllerEnabled(controllers utilflag.ConfigurationMap, serverResources []*metav1.APIResourceList, controller string, requiredResources []schema.GroupVersionResource, defaultValue bool) bool {
	controllerConfig, ok := controllers[controller]
	if ok {
		if controllerConfig == "false" {
			glog.Infof("%s controller disabled by config", controller)
			return false
		}
		if controllerConfig == "true" {
			if !hasRequiredResources(serverResources, requiredResources) {
				glog.Fatalf("%s controller enabled explicitly but API Server does not have required resources", controller)
				panic("unreachable")
			}
			return true
		}
	} else if defaultValue {
		if !hasRequiredResources(serverResources, requiredResources) {
			glog.Warningf("%s controller disabled because API Server does not have required resources", controller)
			return false
		}
	}
	return defaultValue
}

func hasRequiredResources(serverResources []*metav1.APIResourceList, requiredResources []schema.GroupVersionResource) bool {
	for _, resource := range requiredResources {
		found := false
		for _, serverResource := range serverResources {
			if serverResource.GroupVersion == resource.GroupVersion().String() {
				for _, apiResource := range serverResource.APIResources {
					if apiResource.Name == resource.Resource {
						found = true
						break
					}
				}
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func ensureFederationNamespace(clientset *federationclientset.Clientset, namespace string) error {
	ns := v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: namespace,
			Annotations: map[string]string{
				federationapi.FederationClusterSelectorAnnotation: federationapi.FederationOnlyClusterSelector,
			},
		},
	}
	// Probably this is the first operation by controller manager on api server. So retry the operation
	// until timeout to handle scenario where api server is not yet ready.
	err := wait.PollImmediate(apiserverRetryInterval, apiserverWaitTimeout, func() (bool, error) {
		var err error
		_, err = clientset.CoreV1().Namespaces().Get(namespace, metav1.GetOptions{})
		if err != nil {
			if !errors.IsNotFound(err) {
				glog.V(2).Infof("Failed to get namespace %s: %v", namespace, err)
				return false, nil
			}
			_, err := clientset.CoreV1().Namespaces().Create(&ns)
			if err != nil {
				glog.V(2).Infof("Failed to create namespace %s: %v", namespace, err)
				return false, nil
			}
		}
		return true, nil
	})
	return err
}
