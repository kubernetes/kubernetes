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
	"context"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	cacheddiscovery "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/metadatainformer"
	"k8s.io/client-go/restmapper"
	cloudprovider "k8s.io/cloud-provider"
	cloudcontrollerconfig "k8s.io/cloud-provider/app/config"
	"k8s.io/cloud-provider/options"
	"k8s.io/controller-manager/pkg/clientbuilder"
	"k8s.io/controller-manager/pkg/informerfactory"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/component-base/configz"
	"k8s.io/component-base/term"
	"k8s.io/component-base/version"
	"k8s.io/component-base/version/verflag"
	genericcontrollermanager "k8s.io/controller-manager/app"
	"k8s.io/klog/v2"
)

const (
	// ControllerStartJitter is the jitter value used when starting controller managers.
	ControllerStartJitter = 1.0
	// ConfigzName is the name used for register cloud-controller manager /configz, same with GroupName.
	ConfigzName = "cloudcontrollermanager.config.k8s.io"
)

// NewCloudControllerManagerCommand creates a *cobra.Command object with default parameters
func NewCloudControllerManagerCommand(s *options.CloudControllerManagerOptions, c *cloudcontrollerconfig.Config, controllerInitializers map[string]InitFunc) *cobra.Command {

	cmd := &cobra.Command{
		Use: "cloud-controller-manager",
		Long: `The Cloud controller manager is a daemon that embeds
the cloud specific control loops shipped with Kubernetes.`,
		Run: func(cmd *cobra.Command, args []string) {
			verflag.PrintAndExitIfRequested()
			cliflag.PrintFlags(cmd.Flags())

			if err := Run(c.Complete(), controllerInitializers, wait.NeverStop); err != nil {
				fmt.Fprintf(os.Stderr, "%v\n", err)
				os.Exit(1)
			}

		},
		Args: func(cmd *cobra.Command, args []string) error {
			for _, arg := range args {
				if len(arg) > 0 {
					return fmt.Errorf("%q does not take any arguments, got %q", cmd.CommandPath(), args)
				}
			}
			return nil
		},
	}

	fs := cmd.Flags()
	namedFlagSets := s.Flags(KnownControllers(controllerInitializers), ControllersDisabledByDefault.List())
	verflag.AddFlags(namedFlagSets.FlagSet("global"))
	globalflag.AddGlobalFlags(namedFlagSets.FlagSet("global"), cmd.Name())

	if flag.CommandLine.Lookup("cloud-provider-gce-lb-src-cidrs") != nil {
		// hoist this flag from the global flagset to preserve the commandline until
		// the gce cloudprovider is removed.
		globalflag.Register(namedFlagSets.FlagSet("generic"), "cloud-provider-gce-lb-src-cidrs")
	}
	if flag.CommandLine.Lookup("cloud-provider-gce-l7lb-src-cidrs") != nil {
		globalflag.Register(namedFlagSets.FlagSet("generic"), "cloud-provider-gce-l7lb-src-cidrs")
	}
	for _, f := range namedFlagSets.FlagSets {
		fs.AddFlagSet(f)
	}
	usageFmt := "Usage:\n  %s\n"
	cols, _, _ := term.TerminalSize(cmd.OutOrStdout())
	cmd.SetUsageFunc(func(cmd *cobra.Command) error {
		fmt.Fprintf(cmd.OutOrStderr(), usageFmt, cmd.UseLine())
		cliflag.PrintSections(cmd.OutOrStderr(), namedFlagSets, cols)
		return nil
	})
	cmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
		fmt.Fprintf(cmd.OutOrStdout(), "%s\n\n"+usageFmt, cmd.Long, cmd.UseLine())
		cliflag.PrintSections(cmd.OutOrStdout(), namedFlagSets, cols)
	})

	return cmd
}

// Run runs the ExternalCMServer.  This should never exit.
func Run(c *cloudcontrollerconfig.CompletedConfig, controllerInitializers map[string]InitFunc, stopCh <-chan struct{}) error {
	// To help debugging, immediately log version
	klog.Infof("Version: %+v", version.Get())

	// setup /configz endpoint
	if cz, err := configz.New(ConfigzName); err == nil {
		cz.Set(c.ComponentConfig)
	} else {
		klog.Errorf("unable to register configz: %v", err)
	}

	// Setup any health checks we will want to use.
	var checks []healthz.HealthChecker
	var electionChecker *leaderelection.HealthzAdaptor
	if c.ComponentConfig.Generic.LeaderElection.LeaderElect {
		electionChecker = leaderelection.NewLeaderHealthzAdaptor(time.Second * 20)
		checks = append(checks, electionChecker)
	}

	// Start the controller manager HTTP server
	if c.SecureServing != nil {
		unsecuredMux := genericcontrollermanager.NewBaseHandler(&c.ComponentConfig.Generic.Debugging, checks...)
		handler := genericcontrollermanager.BuildHandlerChain(unsecuredMux, &c.Authorization, &c.Authentication)
		// TODO: handle stoppedCh returned by c.SecureServing.Serve
		if _, err := c.SecureServing.Serve(handler, 0, stopCh); err != nil {
			return err
		}
	}
	if c.InsecureServing != nil {
		unsecuredMux := genericcontrollermanager.NewBaseHandler(&c.ComponentConfig.Generic.Debugging, checks...)
		insecureSuperuserAuthn := server.AuthenticationInfo{Authenticator: &server.InsecureSuperuser{}}
		handler := genericcontrollermanager.BuildHandlerChain(unsecuredMux, nil, &insecureSuperuserAuthn)
		if err := c.InsecureServing.Serve(handler, 0, stopCh); err != nil {
			return err
		}
	}

	run := func(ctx context.Context) {
		clientBuilder := clientbuilder.SimpleControllerClientBuilder{
			ClientConfig: c.Kubeconfig,
		}
		controllerContext, err := CreateControllerContext(c, clientBuilder, ctx.Done())
		if err != nil {
			klog.Fatalf("error building controller context: %v", err)
		}
		if err := startControllers(controllerContext, c, ctx.Done(), controllerInitializers); err != nil {
			klog.Fatalf("error running controllers: %v", err)
		}
	}

	if !c.ComponentConfig.Generic.LeaderElection.LeaderElect {
		run(context.TODO())
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
	rl, err := resourcelock.New(c.ComponentConfig.Generic.LeaderElection.ResourceLock,
		c.ComponentConfig.Generic.LeaderElection.ResourceNamespace,
		c.ComponentConfig.Generic.LeaderElection.ResourceName,
		c.LeaderElectionClient.CoreV1(),
		c.LeaderElectionClient.CoordinationV1(),
		resourcelock.ResourceLockConfig{
			Identity:      id,
			EventRecorder: c.EventRecorder,
		})
	if err != nil {
		klog.Fatalf("error creating lock: %v", err)
	}

	// Try and become the leader and start cloud controller manager loops
	leaderelection.RunOrDie(context.TODO(), leaderelection.LeaderElectionConfig{
		Lock:          rl,
		LeaseDuration: c.ComponentConfig.Generic.LeaderElection.LeaseDuration.Duration,
		RenewDeadline: c.ComponentConfig.Generic.LeaderElection.RenewDeadline.Duration,
		RetryPeriod:   c.ComponentConfig.Generic.LeaderElection.RetryPeriod.Duration,
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: run,
			OnStoppedLeading: func() {
				klog.Fatalf("leaderelection lost")
			},
		},
		WatchDog: electionChecker,
		Name:     "cloud-controller-manager",
	})
	panic("unreachable")
}

// startControllers starts the cloud specific controller loops.
func startControllers(ctx genericcontrollermanager.ControllerContext, c *cloudcontrollerconfig.CompletedConfig, stopCh <-chan struct{}, controllers map[string]InitFunc) error {
	for controllerName, initFn := range controllers {
		if !genericcontrollermanager.IsControllerEnabled(controllerName, ControllersDisabledByDefault, c.ComponentConfig.Generic.Controllers) {
			klog.Warningf("%q is disabled", controllerName)
			continue
		}

		klog.V(1).Infof("Starting %q", controllerName)
		_, started, err := initFn(ctx)
		if err != nil {
			klog.Errorf("Error starting %q", controllerName)
			return err
		}
		if !started {
			klog.Warningf("Skipping %q", controllerName)
			continue
		}
		klog.Infof("Started %q", controllerName)

		time.Sleep(wait.Jitter(c.ComponentConfig.Generic.ControllerStartInterval.Duration, ControllerStartJitter))
	}

	// If apiserver is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and controller manager at the same time.
	if err := genericcontrollermanager.WaitForAPIServer(c.VersionedClient, 10*time.Second); err != nil {
		klog.Fatalf("Failed to wait for apiserver being healthy: %v", err)
	}

	c.SharedInformers.Start(stopCh)

	select {}
}

// InitFunc is used to launch a particular controller.  It may run additional "should I activate checks".
// Any error returned will cause the controller process to `Fatal`
// The bool indicates whether the controller was enabled.
type InitFunc func(ctx genericcontrollermanager.ControllerContext) (debuggingHandler http.Handler, enabled bool, err error)

// KnownControllers indicate the default controller we are known.
func KnownControllers(controllerInitializers map[string]InitFunc) []string {
	ret := sets.StringKeySet(controllerInitializers)
	return ret.List()
}

// ControllersDisabledByDefault is the controller disabled default when starting cloud-controller managers.
var ControllersDisabledByDefault = sets.NewString()

// DefaultControllerInitializers is a private map of named controller groups (you can start more than one in an init func)
// paired to their InitFunc.  This allows for structured downstream composition and subdivision.
func DefaultControllerInitializers(completedConfig *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface) map[string]InitFunc {
	controllers := map[string]InitFunc{}
	controllers["cloud-node"] = StartCloudNodeControllerWrapper(completedConfig, cloud)
	controllers["cloud-node-lifecycle"] = startCloudNodeLifecycleControllerWrapper(completedConfig, cloud)
	controllers["service"] = startServiceControllerWrapper(completedConfig, cloud)
	controllers["route"] = startRouteControllerWrapper(completedConfig, cloud)
	return controllers
}

// StartCloudNodeControllerWrapper is used to take cloud cofig as input and start cloud node controller
func StartCloudNodeControllerWrapper(completedConfig *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface) func(ctx genericcontrollermanager.ControllerContext) (http.Handler, bool, error) {
	return func(ctx genericcontrollermanager.ControllerContext) (http.Handler, bool, error) {
		return startCloudNodeController(completedConfig, cloud, ctx.Stop)
	}
}

// startCloudNodeLifecycleControllerWrapper is used to take cloud cofig as input and start cloud node lifecycle controller
func startCloudNodeLifecycleControllerWrapper(completedConfig *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface) func(ctx genericcontrollermanager.ControllerContext) (http.Handler, bool, error) {
	return func(ctx genericcontrollermanager.ControllerContext) (http.Handler, bool, error) {
		return startCloudNodeLifecycleController(completedConfig, cloud, ctx.Stop)
	}
}

// startServiceControllerWrapper is used to take cloud cofig as input and start service controller
func startServiceControllerWrapper(completedConfig *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface) func(ctx genericcontrollermanager.ControllerContext) (http.Handler, bool, error) {
	return func(ctx genericcontrollermanager.ControllerContext) (http.Handler, bool, error) {
		return startServiceController(completedConfig, cloud, ctx.Stop)
	}
}

// startRouteControllerWrapper is used to take cloud cofig as input and start route controller
func startRouteControllerWrapper(completedConfig *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface) func(ctx genericcontrollermanager.ControllerContext) (http.Handler, bool, error) {
	return func(ctx genericcontrollermanager.ControllerContext) (http.Handler, bool, error) {
		return startRouteController(completedConfig, cloud, ctx.Stop)
	}
}

// CreateControllerContext creates a context struct containing references to resources needed by the
// controllers such as the cloud provider and clientBuilder. rootClientBuilder is only used for
// the shared-informers client and token controller.
func CreateControllerContext(s *cloudcontrollerconfig.CompletedConfig, clientBuilder clientbuilder.ControllerClientBuilder, stop <-chan struct{}) (genericcontrollermanager.ControllerContext, error) {
	versionedClient := clientBuilder.ClientOrDie("shared-informers")
	sharedInformers := informers.NewSharedInformerFactory(versionedClient, ResyncPeriod(s)())

	metadataClient := metadata.NewForConfigOrDie(clientBuilder.ConfigOrDie("metadata-informers"))
	metadataInformers := metadatainformer.NewSharedInformerFactory(metadataClient, ResyncPeriod(s)())

	// If apiserver is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and controller manager at the same time.
	if err := genericcontrollermanager.WaitForAPIServer(versionedClient, 10*time.Second); err != nil {
		return genericcontrollermanager.ControllerContext{}, fmt.Errorf("failed to wait for apiserver being healthy: %v", err)
	}

	// Use a discovery client capable of being refreshed.
	discoveryClient := clientBuilder.ClientOrDie("controller-discovery")
	cachedClient := cacheddiscovery.NewMemCacheClient(discoveryClient.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedClient)
	go wait.Until(func() {
		restMapper.Reset()
	}, 30*time.Second, stop)

	availableResources, err := GetAvailableResources(clientBuilder)
	if err != nil {
		return genericcontrollermanager.ControllerContext{}, err
	}

	ctx := genericcontrollermanager.ControllerContext{
		ClientBuilder:                   clientBuilder,
		InformerFactory:                 sharedInformers,
		ObjectOrMetadataInformerFactory: informerfactory.NewInformerFactory(sharedInformers, metadataInformers),
		RESTMapper:                      restMapper,
		AvailableResources:              availableResources,
		Stop:                            stop,
		InformersStarted:                make(chan struct{}),
		ResyncPeriod:                    ResyncPeriod(s),
	}
	return ctx, nil
}

// GetAvailableResources gets the map which contains all available resources of the apiserver
// TODO: In general, any controller checking this needs to be dynamic so
// users don't have to restart their controller manager if they change the apiserver.
// Until we get there, the structure here needs to be exposed for the construction of a proper ControllerContext.
func GetAvailableResources(clientBuilder clientbuilder.ControllerClientBuilder) (map[schema.GroupVersionResource]bool, error) {
	client := clientBuilder.ClientOrDie("controller-discovery")
	discoveryClient := client.Discovery()
	_, resourceMap, err := discoveryClient.ServerGroupsAndResources()
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to get all supported resources from server: %v", err))
	}
	if len(resourceMap) == 0 {
		return nil, fmt.Errorf("unable to get any supported resources from server")
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

// ResyncPeriod returns a function which generates a duration each time it is
// invoked; this is so that multiple controllers don't get into lock-step and all
// hammer the apiserver with list requests simultaneously.
func ResyncPeriod(c *cloudcontrollerconfig.CompletedConfig) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(c.ComponentConfig.Generic.MinResyncPeriod.Nanoseconds()) * factor)
	}
}
