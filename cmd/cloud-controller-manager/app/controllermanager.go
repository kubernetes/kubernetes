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

package app

import (
	"fmt"
	"math/rand"
	"net"
	"os"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	cloudcontrollerconfig "k8s.io/kubernetes/cmd/cloud-controller-manager/app/config"
	"k8s.io/kubernetes/cmd/cloud-controller-manager/app/options"
	genericcontrollermanager "k8s.io/kubernetes/cmd/controller-manager/app"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	cloudcontrollers "k8s.io/kubernetes/pkg/controller/cloud"
	routecontroller "k8s.io/kubernetes/pkg/controller/route"
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
	"k8s.io/kubernetes/pkg/util/configz"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
	"k8s.io/kubernetes/pkg/version/verflag"
)

const (
	// ControllerStartJitter is the jitter value used when starting controller managers.
	ControllerStartJitter = 1.0
)

// NewCloudControllerManagerCommand creates a *cobra.Command object with default parameters
func NewCloudControllerManagerCommand() *cobra.Command {
	s, err := options.NewCloudControllerManagerOptions()
	if err != nil {
		glog.Fatalf("unable to initialize command options: %v", err)
	}

	cmd := &cobra.Command{
		Use: "cloud-controller-manager",
		Long: `The Cloud controller manager is a daemon that embeds
the cloud specific control loops shipped with Kubernetes.`,
		Run: func(cmd *cobra.Command, args []string) {
			verflag.PrintAndExitIfRequested()
			utilflag.PrintFlags(cmd.Flags())

			c, err := s.Config()
			if err != nil {
				fmt.Fprintf(os.Stderr, "%v\n", err)
				os.Exit(1)
			}

			if err := Run(c.Complete()); err != nil {
				fmt.Fprintf(os.Stderr, "%v\n", err)
				os.Exit(1)
			}

		},
	}
	s.AddFlags(cmd.Flags())

	return cmd
}

// resyncPeriod computes the time interval a shared informer waits before resyncing with the api server
func resyncPeriod(c *cloudcontrollerconfig.CompletedConfig) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(c.ComponentConfig.GenericComponent.MinResyncPeriod.Nanoseconds()) * factor)
	}
}

// Run runs the ExternalCMServer.  This should never exit.
func Run(c *cloudcontrollerconfig.CompletedConfig) error {
	cloud, err := cloudprovider.InitCloudProvider(c.ComponentConfig.CloudProvider.Name, c.ComponentConfig.CloudProvider.CloudConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}
	if cloud == nil {
		glog.Fatalf("cloud provider is nil")
	}

	if cloud.HasClusterID() == false {
		if c.ComponentConfig.KubeCloudShared.AllowUntaggedCloud == true {
			glog.Warning("detected a cluster without a ClusterID.  A ClusterID will be required in the future.  Please tag your cluster to avoid any future issues")
		} else {
			glog.Fatalf("no ClusterID found.  A ClusterID is required for the cloud provider to function properly.  This check can be bypassed by setting the allow-untagged-cloud option")
		}
	}

	// setup /configz endpoint
	if cz, err := configz.New("componentconfig"); err == nil {
		cz.Set(c.ComponentConfig)
	} else {
		glog.Errorf("unable to register configz: %c", err)
	}

	// Setup any healthz checks we will want to use.
	var checks []healthz.HealthzChecker
	var electionChecker *leaderelection.HealthzAdaptor
	if c.ComponentConfig.GenericComponent.LeaderElection.LeaderElect {
		electionChecker = leaderelection.NewLeaderHealthzAdaptor(time.Second * 20)
		checks = append(checks, electionChecker)
	}

	// Start the controller manager HTTP server
	stopCh := make(chan struct{})
	if c.SecureServing != nil {
		handler := genericcontrollermanager.NewBaseHandler(&c.ComponentConfig.Debugging, checks...)
		handler = genericcontrollermanager.BuildHandlerChain(handler, &c.Authorization, &c.Authentication)
		if err := c.SecureServing.Serve(handler, 0, stopCh); err != nil {
			return err
		}
	}
	if c.InsecureServing != nil {
		handler := genericcontrollermanager.NewBaseHandler(&c.ComponentConfig.Debugging, checks...)
		handler = genericcontrollermanager.BuildHandlerChain(handler, &c.Authorization, &c.Authentication)
		if err := c.InsecureServing.Serve(handler, 0, stopCh); err != nil {
			return err
		}
	}

	run := func(stop <-chan struct{}) {
		rootClientBuilder := controller.SimpleControllerClientBuilder{
			ClientConfig: c.Kubeconfig,
		}
		var clientBuilder controller.ControllerClientBuilder
		if c.ComponentConfig.KubeCloudShared.UseServiceAccountCredentials {
			clientBuilder = controller.SAControllerClientBuilder{
				ClientConfig:         restclient.AnonymousClientConfig(c.Kubeconfig),
				CoreClient:           c.Client.CoreV1(),
				AuthenticationClient: c.Client.AuthenticationV1(),
				Namespace:            "kube-system",
			}
		} else {
			clientBuilder = rootClientBuilder
		}

		if err := startControllers(c, rootClientBuilder, clientBuilder, stop, cloud); err != nil {
			glog.Fatalf("error running controllers: %v", err)
		}
	}

	if !c.ComponentConfig.GenericComponent.LeaderElection.LeaderElect {
		run(nil)
		panic("unreachable")
	}

	// Identity used to distinguish between multiple cloud controller manager instances
	id, err := os.Hostname()
	if err != nil {
		return err
	}
	// add a uniquifier so that two processes on the same host don't accidentally both become active
	id = id + "_" + string(uuid.NewUUID())

	// Lock required for leader election
	rl, err := resourcelock.New(c.ComponentConfig.GenericComponent.LeaderElection.ResourceLock,
		"kube-system",
		"cloud-controller-manager",
		c.LeaderElectionClient.CoreV1(),
		resourcelock.ResourceLockConfig{
			Identity:      id,
			EventRecorder: c.EventRecorder,
		})
	if err != nil {
		glog.Fatalf("error creating lock: %v", err)
	}

	// Try and become the leader and start cloud controller manager loops
	leaderelection.RunOrDie(leaderelection.LeaderElectionConfig{
		Lock:          rl,
		LeaseDuration: c.ComponentConfig.GenericComponent.LeaderElection.LeaseDuration.Duration,
		RenewDeadline: c.ComponentConfig.GenericComponent.LeaderElection.RenewDeadline.Duration,
		RetryPeriod:   c.ComponentConfig.GenericComponent.LeaderElection.RetryPeriod.Duration,
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: run,
			OnStoppedLeading: func() {
				glog.Fatalf("leaderelection lost")
			},
		},
		WatchDog: electionChecker,
		Name:     "cloud-controller-manager",
	})
	panic("unreachable")
}

// startControllers starts the cloud specific controller loops.
func startControllers(c *cloudcontrollerconfig.CompletedConfig, rootClientBuilder, clientBuilder controller.ControllerClientBuilder, stop <-chan struct{}, cloud cloudprovider.Interface) error {
	// Function to build the kube client object
	client := func(serviceAccountName string) kubernetes.Interface {
		return clientBuilder.ClientOrDie(serviceAccountName)
	}
	if cloud != nil {
		// Initialize the cloud provider with a reference to the clientBuilder
		cloud.Initialize(clientBuilder)
	}

	// TODO: move this setup into Config
	versionedClient := rootClientBuilder.ClientOrDie("shared-informers")
	sharedInformers := informers.NewSharedInformerFactory(versionedClient, resyncPeriod(c)())

	// Start the CloudNodeController
	nodeController := cloudcontrollers.NewCloudNodeController(
		sharedInformers.Core().V1().Nodes(),
		client("cloud-node-controller"), cloud,
		c.ComponentConfig.KubeCloudShared.NodeMonitorPeriod.Duration,
		c.ComponentConfig.NodeStatusUpdateFrequency.Duration)

	nodeController.Run(stop)
	time.Sleep(wait.Jitter(c.ComponentConfig.GenericComponent.ControllerStartInterval.Duration, ControllerStartJitter))

	// Start the PersistentVolumeLabelController
	pvlController := cloudcontrollers.NewPersistentVolumeLabelController(client("pvl-controller"), cloud)
	go pvlController.Run(5, stop)
	time.Sleep(wait.Jitter(c.ComponentConfig.GenericComponent.ControllerStartInterval.Duration, ControllerStartJitter))

	// Start the service controller
	serviceController, err := servicecontroller.New(
		cloud,
		client("service-controller"),
		sharedInformers.Core().V1().Services(),
		sharedInformers.Core().V1().Nodes(),
		c.ComponentConfig.KubeCloudShared.ClusterName,
	)
	if err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	} else {
		go serviceController.Run(stop, int(c.ComponentConfig.ServiceController.ConcurrentServiceSyncs))
		time.Sleep(wait.Jitter(c.ComponentConfig.GenericComponent.ControllerStartInterval.Duration, ControllerStartJitter))
	}

	// If CIDRs should be allocated for pods and set on the CloudProvider, then start the route controller
	if c.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs && c.ComponentConfig.KubeCloudShared.ConfigureCloudRoutes {
		if routes, ok := cloud.Routes(); !ok {
			glog.Warning("configure-cloud-routes is set, but cloud provider does not support routes. Will not configure cloud provider routes.")
		} else {
			var clusterCIDR *net.IPNet
			if len(strings.TrimSpace(c.ComponentConfig.KubeCloudShared.ClusterCIDR)) != 0 {
				_, clusterCIDR, err = net.ParseCIDR(c.ComponentConfig.KubeCloudShared.ClusterCIDR)
				if err != nil {
					glog.Warningf("Unsuccessful parsing of cluster CIDR %v: %v", c.ComponentConfig.KubeCloudShared.ClusterCIDR, err)
				}
			}

			routeController := routecontroller.New(routes, client("route-controller"), sharedInformers.Core().V1().Nodes(), c.ComponentConfig.KubeCloudShared.ClusterName, clusterCIDR)
			go routeController.Run(stop, c.ComponentConfig.KubeCloudShared.RouteReconciliationPeriod.Duration)
			time.Sleep(wait.Jitter(c.ComponentConfig.GenericComponent.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	} else {
		glog.Infof("Will not configure cloud provider routes for allocate-node-cidrs: %v, configure-cloud-routes: %v.", c.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs, c.ComponentConfig.KubeCloudShared.ConfigureCloudRoutes)
	}

	// If apiserver is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and controller manager at the same time.
	err = genericcontrollermanager.WaitForAPIServer(versionedClient, 10*time.Second)
	if err != nil {
		glog.Fatalf("Failed to wait for apiserver being healthy: %v", err)
	}

	sharedInformers.Start(stop)

	select {}
}
