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
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"strconv"

	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5"
	"k8s.io/kubernetes/federation/cmd/federation-controller-manager/app/options"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	clustercontroller "k8s.io/kubernetes/federation/pkg/federation-controller/cluster"
	configmapcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/configmap"
	daemonset "k8s.io/kubernetes/federation/pkg/federation-controller/daemonset"
	deploymentcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/deployment"
	ingresscontroller "k8s.io/kubernetes/federation/pkg/federation-controller/ingress"
	namespacecontroller "k8s.io/kubernetes/federation/pkg/federation-controller/namespace"
	replicasetcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/replicaset"
	secretcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/secret"
	servicecontroller "k8s.io/kubernetes/federation/pkg/federation-controller/service"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/config"
)

const (
	// "federation-apiserver-kubeconfig" was the old name we used to
	// store Federation API server kubeconfig secret. We are
	// deprecating it in favor of `--kubeconfig` flag but giving people
	// time to migrate.
	// TODO(madhusudancs): this name is deprecated in 1.5 and should be
	// removed in 1.6. Remove it in 1.6.
	DeprecatedKubeconfigSecretName = "federation-apiserver-kubeconfig"
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

	// If s.Kubeconfig flag is empty, try with the deprecated name in 1.5.
	// TODO(madhusudancs): Remove this in 1.6.
	var restClientCfg *restclient.Config
	var err error
	if len(s.Kubeconfig) <= 0 {
		restClientCfg, err = restClientConfigFromSecret(s.Master)
		if err != nil {
			return err
		}
	} else {
		// Create the config to talk to federation-apiserver.
		restClientCfg, err = clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
		if err != nil || restClientCfg == nil {
			// Retry with the deprecated name in 1.5.
			// TODO(madhusudancs): Remove this in 1.6.
			glog.V(2).Infof("Couldn't build the rest client config from flags: %v", err)
			glog.V(2).Infof("Trying with deprecated secret: %s", DeprecatedKubeconfigSecretName)
			restClientCfg, err = restClientConfigFromSecret(s.Master)
			if err != nil {
				return err
			}
		}
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
		}
		mux.Handle("/metrics", prometheus.Handler())

		server := &http.Server{
			Addr:    net.JoinHostPort(s.Address, strconv.Itoa(s.Port)),
			Handler: mux,
		}
		glog.Fatal(server.ListenAndServe())
	}()

	run := func() {
		err := StartControllers(s, restClientCfg)
		glog.Fatalf("error running controllers: %v", err)
		panic("unreachable")
	}
	run()
	panic("unreachable")
}

func StartControllers(s *options.CMServer, restClientCfg *restclient.Config) error {
	glog.Infof("Loading client config for cluster controller %q", "cluster-controller")
	ccClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, "cluster-controller"))
	glog.Infof("Running cluster controller")
	go clustercontroller.NewclusterController(ccClientset, s.ClusterMonitorPeriod.Duration).Run()
	dns, err := dnsprovider.InitDnsProvider(s.DnsProvider, s.DnsConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}

	discoveryClient := discovery.NewDiscoveryClientForConfigOrDie(restClientCfg)
	serverResources, err := discoveryClient.ServerResources()
	if err != nil {
		glog.Fatalf("Could not find resources from API Server: %v", err)
	}

	glog.Infof("Loading client config for namespace controller %q", "namespace-controller")
	nsClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, "namespace-controller"))
	namespaceController := namespacecontroller.NewNamespaceController(nsClientset)
	glog.Infof("Running namespace controller")
	namespaceController.Run(wait.NeverStop)

	secretcontrollerClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, "secret-controller"))
	secretcontroller := secretcontroller.NewSecretController(secretcontrollerClientset)
	secretcontroller.Run(wait.NeverStop)

	configmapcontrollerClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, "configmap-controller"))
	configmapcontroller := configmapcontroller.NewConfigMapController(configmapcontrollerClientset)
	configmapcontroller.Run(wait.NeverStop)

	daemonsetcontrollerClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, "daemonset-controller"))
	daemonsetcontroller := daemonset.NewDaemonSetController(daemonsetcontrollerClientset)
	daemonsetcontroller.Run(wait.NeverStop)

	replicaSetClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, replicasetcontroller.UserAgentName))
	replicaSetController := replicasetcontroller.NewReplicaSetController(replicaSetClientset)
	go replicaSetController.Run(s.ConcurrentReplicaSetSyncs, wait.NeverStop)

	deploymentClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, deploymentcontroller.UserAgentName))
	deploymentController := deploymentcontroller.NewDeploymentController(deploymentClientset)
	// TODO: rename s.ConcurentReplicaSetSyncs
	go deploymentController.Run(s.ConcurrentReplicaSetSyncs, wait.NeverStop)

	if controllerEnabled(s.Controllers, serverResources, ingresscontroller.ControllerName, ingresscontroller.RequiredResources, true) {
		glog.Infof("Loading client config for ingress controller %q", "ingress-controller")
		ingClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, "ingress-controller"))
		ingressController := ingresscontroller.NewIngressController(ingClientset)
		glog.Infof("Running ingress controller")
		ingressController.Run(wait.NeverStop)
	}

	glog.Infof("Loading client config for service controller %q", servicecontroller.UserAgentName)
	scClientset := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, servicecontroller.UserAgentName))
	servicecontroller := servicecontroller.New(scClientset, dns, s.FederationName, s.ServiceDnsSuffix, s.ZoneName, s.ZoneID)
	glog.Infof("Running service controller")
	if err := servicecontroller.Run(s.ConcurrentServiceSyncs, wait.NeverStop); err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	}

	select {}
}

// TODO(madhusudancs): Remove this in 1.6. This is only temporary to give an
// upgrade path in 1.4/1.5.
func restClientConfigFromSecret(master string) (*restclient.Config, error) {
	kubeconfigGetter := util.KubeconfigGetterForSecret(DeprecatedKubeconfigSecretName)
	restClientCfg, err := clientcmd.BuildConfigFromKubeconfigGetter(master, kubeconfigGetter)
	if err != nil {
		return nil, fmt.Errorf("failed to find the Federation API server kubeconfig, tried the --kubeconfig flag and the deprecated secret %s: %v", DeprecatedKubeconfigSecretName, err)
	}
	return restClientCfg, nil
}

func controllerEnabled(controllers config.ConfigurationMap, serverResources []*metav1.APIResourceList, controller string, requiredResources []schema.GroupVersionResource, defaultValue bool) bool {
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
