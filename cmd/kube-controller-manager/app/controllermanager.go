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
package app

import (
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/blang/semver/v4"
	"github.com/spf13/cobra"
	coordinationv1 "k8s.io/api/coordination/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/apiserver/pkg/server/statusz"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/metadatainformer"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	certutil "k8s.io/client-go/util/cert"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	basecompatibility "k8s.io/component-base/compatibility"
	"k8s.io/component-base/configz"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	metricsfeatures "k8s.io/component-base/metrics/features"
	controllersmetrics "k8s.io/component-base/metrics/prometheus/controllers"
	"k8s.io/component-base/metrics/prometheus/slis"
	"k8s.io/component-base/term"
	utilversion "k8s.io/component-base/version"
	"k8s.io/component-base/version/verflag"
	zpagesfeatures "k8s.io/component-base/zpages/features"
	"k8s.io/component-base/zpages/flagz"
	genericcontrollermanager "k8s.io/controller-manager/app"
	"k8s.io/controller-manager/controller"
	"k8s.io/controller-manager/pkg/clientbuilder"
	controllerhealthz "k8s.io/controller-manager/pkg/healthz"
	"k8s.io/controller-manager/pkg/informerfactory"
	"k8s.io/controller-manager/pkg/leadermigration"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	garbagecollector "k8s.io/kubernetes/pkg/controller/garbagecollector"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

func init() {
	utilruntime.Must(logsapi.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
	utilruntime.Must(metricsfeatures.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
}

const (
	// ControllerStartJitter is the Jitter used when starting controller managers
	ControllerStartJitter = 1.0
	// ConfigzName is the name used for register kube-controller manager /configz, same with GroupName.
	ConfigzName = "kubecontrollermanager.config.k8s.io"
	// kubeControllerManager defines variable used internally when referring to cloud-controller-manager component
	kubeControllerManager = "kube-controller-manager"
)

// NewControllerManagerCommand creates a *cobra.Command object with default parameters
func NewControllerManagerCommand() *cobra.Command {
	s, err := options.NewKubeControllerManagerOptions()
	if err != nil {
		klog.Background().Error(err, "Unable to initialize command options")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	cmd := &cobra.Command{
		Use: kubeControllerManager,
		Long: `The Kubernetes controller manager is a daemon that embeds
the core control loops shipped with Kubernetes. In applications of robotics and
automation, a control loop is a non-terminating loop that regulates the state of
the system. In Kubernetes, a controller is a control loop that watches the shared
state of the cluster through the apiserver and makes changes attempting to move the
current state towards the desired state. Examples of controllers that ship with
Kubernetes today are the replication controller, endpoints controller, namespace
controller, and serviceaccounts controller.`,
		PersistentPreRunE: func(*cobra.Command, []string) error {
			// silence client-go warnings.
			// kube-controller-manager generically watches APIs (including deprecated ones),
			// and CI ensures it works properly against matching kube-apiserver versions.
			restclient.SetDefaultWarningHandler(restclient.NoWarnings{})
			// makes sure feature gates are set before RunE.
			return s.ComponentGlobalsRegistry.Set()
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			verflag.PrintAndExitIfRequested()

			// Activate logging as soon as possible, after that
			// show flags with the final logging configuration.
			if err := logsapi.ValidateAndApply(s.Logs, utilfeature.DefaultFeatureGate); err != nil {
				return err
			}
			cliflag.PrintFlags(cmd.Flags())

			ctx := context.Background()
			c, err := s.Config(ctx, KnownControllers(), ControllersDisabledByDefault(), ControllerAliases())
			if err != nil {
				return err
			}

			// add feature enablement metrics
			fg := s.ComponentGlobalsRegistry.FeatureGateFor(basecompatibility.DefaultKubeComponent)
			fg.(featuregate.MutableFeatureGate).AddMetrics()
			// add component version metrics
			s.ComponentGlobalsRegistry.AddMetrics()
			return Run(ctx, c.Complete())
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
	namedFlagSets := s.Flags(KnownControllers(), ControllersDisabledByDefault(), ControllerAliases())
	s.ParsedFlags = &namedFlagSets
	verflag.AddFlags(namedFlagSets.FlagSet("global"))
	globalflag.AddGlobalFlags(namedFlagSets.FlagSet("global"), cmd.Name(), logs.SkipLoggingConfigurationFlags())
	for _, f := range namedFlagSets.FlagSets {
		fs.AddFlagSet(f)
	}

	cols, _, _ := term.TerminalSize(cmd.OutOrStdout())
	cliflag.SetUsageAndHelpFunc(cmd, namedFlagSets, cols)

	return cmd
}

// ResyncPeriod returns a function which generates a duration each time it is
// invoked; this is because that multiple controllers don't get into lock-step.
func ResyncPeriod(c *config.CompletedConfig) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(c.ComponentConfig.Generic.MinResyncPeriod.Nanoseconds()) * factor)
	}
}

// Run runs the KubeControllerManagerOptions.
func Run(ctx context.Context, c *config.CompletedConfig) error {
	logger := klog.FromContext(ctx)
	stopCh := ctx.Done()

	// To help debugging, immediately log version
	logger.Info("Starting", "version", utilversion.Get())

	logger.Info("Golang settings", "GOGC", os.Getenv("GOGC"), "GOMAXPROCS", os.Getenv("GOMAXPROCS"), "GOTRACEBACK", os.Getenv("GOTRACEBACK"))

	// Start events processing pipeline.
	c.EventBroadcaster.StartStructuredLogging(0)
	c.EventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.Client.CoreV1().Events("")})
	defer c.EventBroadcaster.Shutdown()

	if cfgz, err := configz.New(ConfigzName); err == nil {
		cfgz.Set(c.ComponentConfig)
	} else {
		logger.Error(err, "Unable to register configz")
	}

	// Setup any healthz checks we will want to use.
	var checks []healthz.HealthChecker
	var electionChecker *leaderelection.HealthzAdaptor
	if c.ComponentConfig.Generic.LeaderElection.LeaderElect {
		electionChecker = leaderelection.NewLeaderHealthzAdaptor(time.Second * 20)
		checks = append(checks, electionChecker)
	}
	healthzHandler := controllerhealthz.NewMutableHealthzHandler(checks...)

	// Start the controller manager HTTP server
	// unsecuredMux is the handler for these controller *after* authn/authz filters have been applied
	var unsecuredMux *mux.PathRecorderMux
	if c.SecureServing != nil {
		unsecuredMux = genericcontrollermanager.NewBaseHandler(&c.ComponentConfig.Generic.Debugging, healthzHandler)
		slis.SLIMetricsWithReset{}.Install(unsecuredMux)
		if utilfeature.DefaultFeatureGate.Enabled(zpagesfeatures.ComponentFlagz) {
			if c.Flagz != nil {
				flagz.Install(unsecuredMux, kubeControllerManager, c.Flagz)
			}
		}

		if utilfeature.DefaultFeatureGate.Enabled(zpagesfeatures.ComponentStatusz) {
			statusz.Install(
				unsecuredMux,
				statusz.NewRegistry(
					kubeControllerManager,
					c.ComponentGlobalsRegistry.EffectiveVersionFor(basecompatibility.DefaultKubeComponent),
					statusz.WithListedPaths(unsecuredMux.ListedPaths()),
				),
			)
		}

		handler := genericcontrollermanager.BuildHandlerChain(unsecuredMux, &c.Authorization, &c.Authentication)
		// TODO: handle stoppedCh and listenerStoppedCh returned by c.SecureServing.Serve
		if _, _, err := c.SecureServing.Serve(handler, 0, stopCh); err != nil {
			return err
		}
	}

	clientBuilder, rootClientBuilder := createClientBuilders(c)

	saTokenControllerDescriptor := newServiceAccountTokenControllerDescriptor(rootClientBuilder)

	run := func(ctx context.Context, controllerDescriptors map[string]*ControllerDescriptor) {
		controllerContext, err := CreateControllerContext(ctx, c, rootClientBuilder, clientBuilder)
		if err != nil {
			logger.Error(err, "Error building controller context")
			klog.FlushAndExit(klog.ExitFlushTimeout, 1)
		}

		// Prepare all controllers in advance.
		controllers, err := BuildControllers(ctx, controllerContext, controllerDescriptors, unsecuredMux, healthzHandler)
		if err != nil {
			logger.Error(err, "Error building controllers")
			klog.FlushAndExit(klog.ExitFlushTimeout, 1)
		}

		// Start the informers.
		stopCh := ctx.Done()
		controllerContext.InformerFactory.Start(stopCh)
		controllerContext.ObjectOrMetadataInformerFactory.Start(stopCh)
		close(controllerContext.InformersStarted)

		// Actually start the controllers.
		if len(controllers) > 0 {
			if !RunControllers(ctx, controllerContext, controllers, ControllerStartJitter, c.ControllerShutdownTimeout) {
				klog.FlushAndExit(klog.ExitFlushTimeout, 1)
			}
		} else {
			<-ctx.Done()
		}
	}

	// No leader election, run directly
	if !c.ComponentConfig.Generic.LeaderElection.LeaderElect {
		controllerDescriptors := NewControllerDescriptors()
		controllerDescriptors[names.ServiceAccountTokenController] = saTokenControllerDescriptor
		run(ctx, controllerDescriptors)
		return nil
	}

	id, err := os.Hostname()
	if err != nil {
		return err
	}

	// add a uniquifier so that two processes on the same host don't accidentally both become active
	id = id + "_" + string(uuid.NewUUID())

	// leaderMigrator will be non-nil if and only if Leader Migration is enabled.
	var leaderMigrator *leadermigration.LeaderMigrator = nil

	// If leader migration is enabled, create the LeaderMigrator and prepare for migration
	if leadermigration.Enabled(&c.ComponentConfig.Generic) {
		logger.Info("starting leader migration")

		leaderMigrator = leadermigration.NewLeaderMigrator(&c.ComponentConfig.Generic.LeaderMigration,
			kubeControllerManager)

		// startSATokenControllerInit is the original constructor.
		saTokenControllerInit := saTokenControllerDescriptor.GetControllerConstructor()

		// Wrap saTokenControllerDescriptor to signal readiness for migration after starting the controller.
		saTokenControllerDescriptor.constructor = func(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
			ctrl, err := saTokenControllerInit(ctx, controllerContext, controllerName)
			if err != nil {
				return nil, err
			}

			// This wrapping is not exactly flawless as RunControllers uses type casting,
			// which is now not possible for the wrapped controller.
			// This fortunately doesn't matter for this particular controller.
			return newControllerLoop(func(ctx context.Context) {
				close(leaderMigrator.MigrationReady)
				ctrl.Run(ctx)
			}, controllerName), nil
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.CoordinatedLeaderElection) {
		binaryVersion, err := semver.ParseTolerant(c.ComponentGlobalsRegistry.EffectiveVersionFor(basecompatibility.DefaultKubeComponent).BinaryVersion().String())
		if err != nil {
			return err
		}
		emulationVersion, err := semver.ParseTolerant(c.ComponentGlobalsRegistry.EffectiveVersionFor(basecompatibility.DefaultKubeComponent).EmulationVersion().String())
		if err != nil {
			return err
		}

		// Start lease candidate controller for coordinated leader election
		leaseCandidate, waitForSync, err := leaderelection.NewCandidate(
			c.Client,
			"kube-system",
			id,
			kubeControllerManager,
			binaryVersion.FinalizeVersion(),
			emulationVersion.FinalizeVersion(),
			coordinationv1.OldestEmulationVersion,
		)
		if err != nil {
			return err
		}
		healthzHandler.AddHealthChecker(healthz.NewInformerSyncHealthz(waitForSync))

		go leaseCandidate.Run(ctx)
	}

	// Start the main lock
	go leaderElectAndRun(ctx, c, id, electionChecker,
		c.ComponentConfig.Generic.LeaderElection.ResourceLock,
		c.ComponentConfig.Generic.LeaderElection.ResourceName,
		leaderelection.LeaderCallbacks{
			OnStartedLeading: func(ctx context.Context) {
				controllerDescriptors := NewControllerDescriptors()
				if leaderMigrator != nil {
					// If leader migration is enabled, we should start only non-migrated controllers
					//  for the main lock.
					controllerDescriptors = filteredControllerDescriptors(controllerDescriptors, leaderMigrator.FilterFunc, leadermigration.ControllerNonMigrated)
					logger.Info("leader migration: starting main controllers.")
				}
				controllerDescriptors[names.ServiceAccountTokenController] = saTokenControllerDescriptor
				run(ctx, controllerDescriptors)
			},
			OnStoppedLeading: func() {
				logger.Error(nil, "leaderelection lost")
				klog.FlushAndExit(klog.ExitFlushTimeout, 1)
			},
		})

	// If Leader Migration is enabled, proceed to attempt the migration lock.
	if leaderMigrator != nil {
		// Wait for Service Account Token Controller to start before acquiring the migration lock.
		// At this point, the main lock must have already been acquired, or the KCM process already exited.
		// We wait for the main lock before acquiring the migration lock to prevent the situation
		//  where KCM instance A holds the main lock while KCM instance B holds the migration lock.
		<-leaderMigrator.MigrationReady

		// Start the migration lock.
		go leaderElectAndRun(ctx, c, id, electionChecker,
			c.ComponentConfig.Generic.LeaderMigration.ResourceLock,
			c.ComponentConfig.Generic.LeaderMigration.LeaderName,
			leaderelection.LeaderCallbacks{
				OnStartedLeading: func(ctx context.Context) {
					logger.Info("leader migration: starting migrated controllers.")
					controllerDescriptors := NewControllerDescriptors()
					controllerDescriptors = filteredControllerDescriptors(controllerDescriptors, leaderMigrator.FilterFunc, leadermigration.ControllerMigrated)
					// DO NOT start saTokenController under migration lock
					delete(controllerDescriptors, names.ServiceAccountTokenController)
					run(ctx, controllerDescriptors)
				},
				OnStoppedLeading: func() {
					logger.Error(nil, "migration leaderelection lost")
					klog.FlushAndExit(klog.ExitFlushTimeout, 1)
				},
			})
	}

	<-stopCh
	return nil
}

// ControllerContext defines the context object for controller
type ControllerContext struct {
	// ClientBuilder will provide a client for this controller to use
	ClientBuilder clientbuilder.ControllerClientBuilder

	// InformerFactory gives access to informers for the controller.
	InformerFactory informers.SharedInformerFactory

	// ObjectOrMetadataInformerFactory gives access to informers for typed resources
	// and dynamic resources by their metadata. All generic controllers currently use
	// object metadata - if a future controller needs access to the full object this
	// would become GenericInformerFactory and take a dynamic client.
	ObjectOrMetadataInformerFactory informerfactory.InformerFactory

	// ComponentConfig provides access to init options for a given controller
	ComponentConfig kubectrlmgrconfig.KubeControllerManagerConfiguration

	// DeferredDiscoveryRESTMapper is a RESTMapper that will defer
	// initialization of the RESTMapper until the first mapping is
	// requested.
	RESTMapper *restmapper.DeferredDiscoveryRESTMapper

	// InformersStarted is closed after all of the controllers have been initialized and are running.  After this point it is safe,
	// for an individual controller to start the shared informers. Before it is closed, they should not.
	InformersStarted chan struct{}

	// ResyncPeriod generates a duration each time it is invoked; this is so that
	// multiple controllers don't get into lock-step and all hammer the apiserver
	// with list requests simultaneously.
	ResyncPeriod func() time.Duration

	// ControllerManagerMetrics provides a proxy to set controller manager specific metrics.
	ControllerManagerMetrics *controllersmetrics.ControllerManagerMetrics

	// GraphBuilder gives an access to dependencyGraphBuilder which keeps tracks of resources in the cluster
	GraphBuilder *garbagecollector.GraphBuilder
}

// IsControllerEnabled checks if the context's controllers enabled or not
func (c ControllerContext) IsControllerEnabled(controllerDescriptor *ControllerDescriptor) bool {
	controllersDisabledByDefault := sets.NewString()
	if controllerDescriptor.IsDisabledByDefault() {
		controllersDisabledByDefault.Insert(controllerDescriptor.Name())
	}
	return genericcontrollermanager.IsControllerEnabled(controllerDescriptor.Name(), controllersDisabledByDefault, c.ComponentConfig.Generic.Controllers)
}

// NewClientConfig is a shortcut for ClientBuilder.Config. It wraps the error with an additional message.
func (c ControllerContext) NewClientConfig(name string) (*restclient.Config, error) {
	config, err := c.ClientBuilder.Config(name)
	if err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes client config for %q: %w", name, err)
	}
	return config, nil
}

// NewClient is a shortcut for ClientBuilder.Client. It wraps the error with an additional message.
func (c ControllerContext) NewClient(name string) (kubernetes.Interface, error) {
	client, err := c.ClientBuilder.Client(name)
	if err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes client for %q: %w", name, err)
	}
	return client, nil
}

// CreateControllerContext creates a context struct containing references to resources needed by the
// controllers such as the cloud provider and clientBuilder. rootClientBuilder is only used for
// the shared-informers client and token controller.
func CreateControllerContext(ctx context.Context, s *config.CompletedConfig, rootClientBuilder, clientBuilder clientbuilder.ControllerClientBuilder) (ControllerContext, error) {
	// Informer transform to trim ManagedFields for memory efficiency.
	trim := func(obj interface{}) (interface{}, error) {
		if accessor, err := meta.Accessor(obj); err == nil {
			if accessor.GetManagedFields() != nil {
				accessor.SetManagedFields(nil)
			}
		}
		return obj, nil
	}

	versionedClient, err := rootClientBuilder.Client("shared-informers")
	if err != nil {
		return ControllerContext{}, fmt.Errorf("failed to create Kubernetes client for %q: %w", "shared-informers", err)
	}

	sharedInformers := informers.NewSharedInformerFactoryWithOptions(versionedClient, ResyncPeriod(s)(), informers.WithTransform(trim))

	metadataConfig, err := rootClientBuilder.Config("metadata-informers")
	if err != nil {
		return ControllerContext{}, fmt.Errorf("failed to create metadata client config: %w", err)
	}

	metadataClient, err := metadata.NewForConfig(metadataConfig)
	if err != nil {
		return ControllerContext{}, fmt.Errorf("failed to create metadata client: %w", err)
	}

	metadataInformers := metadatainformer.NewSharedInformerFactoryWithOptions(metadataClient, ResyncPeriod(s)(), metadatainformer.WithTransform(trim))

	// If apiserver is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and controller manager at the same time.
	if err := genericcontrollermanager.WaitForAPIServer(versionedClient, 10*time.Second); err != nil {
		return ControllerContext{}, fmt.Errorf("failed to wait for apiserver being healthy: %w", err)
	}

	// Use a discovery client capable of being refreshed.
	discoveryClient, err := rootClientBuilder.DiscoveryClient("controller-discovery")
	if err != nil {
		return ControllerContext{}, fmt.Errorf("failed to create discovery client: %w", err)
	}

	cachedClient := cacheddiscovery.NewMemCacheClient(discoveryClient)
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedClient)
	go wait.Until(func() {
		restMapper.Reset()
	}, 30*time.Second, ctx.Done())

	controllerContext := ControllerContext{
		ClientBuilder:                   clientBuilder,
		InformerFactory:                 sharedInformers,
		ObjectOrMetadataInformerFactory: informerfactory.NewInformerFactory(sharedInformers, metadataInformers),
		ComponentConfig:                 s.ComponentConfig,
		RESTMapper:                      restMapper,
		InformersStarted:                make(chan struct{}),
		ResyncPeriod:                    ResyncPeriod(s),
		ControllerManagerMetrics:        controllersmetrics.NewControllerManagerMetrics(kubeControllerManager),
	}

	if controllerContext.ComponentConfig.GarbageCollectorController.EnableGarbageCollector &&
		controllerContext.IsControllerEnabled(NewControllerDescriptors()[names.GarbageCollectorController]) {
		ignoredResources := make(map[schema.GroupResource]struct{})
		for _, r := range controllerContext.ComponentConfig.GarbageCollectorController.GCIgnoredResources {
			ignoredResources[schema.GroupResource{Group: r.Group, Resource: r.Resource}] = struct{}{}
		}

		controllerContext.GraphBuilder = garbagecollector.NewDependencyGraphBuilder(
			ctx,
			metadataClient,
			controllerContext.RESTMapper,
			ignoredResources,
			controllerContext.ObjectOrMetadataInformerFactory,
			controllerContext.InformersStarted,
		)
	}

	controllersmetrics.Register()
	return controllerContext, nil
}

// HealthCheckAdder is an interface to represent a healthz handler.
// The extra level of indirection is useful for testing.
type HealthCheckAdder interface {
	AddHealthChecker(checks ...healthz.HealthChecker)
}

// BuildControllers builds all controllers in the given descriptor map. Disabled controllers are obviously skipped.
//
// A health check is registered for each controller using the controller name. The default check always passes.
// If the controller implements controller.HealthCheckable, though, the given check is used.
// The controller can also implement controller.Debuggable, in which case the debug handler is registered with the given mux.
func BuildControllers(ctx context.Context, controllerCtx ControllerContext, controllerDescriptors map[string]*ControllerDescriptor,
	unsecuredMux *mux.PathRecorderMux, healthzHandler HealthCheckAdder) ([]Controller, error) {
	logger := klog.FromContext(ctx)
	var (
		controllers []Controller
		checks      []healthz.HealthChecker
	)
	buildController := func(controllerDesc *ControllerDescriptor) error {
		controllerName := controllerDesc.Name()
		ctrl, err := controllerDesc.BuildController(ctx, controllerCtx)
		if err != nil {
			logger.Error(err, "Error initializing a controller", "controller", controllerName)
			return err
		}
		if ctrl == nil {
			logger.Info("Warning: skipping controller", "controller", controllerName)
			return nil
		}

		check := controllerhealthz.NamedPingChecker(controllerName)
		// check if the controller supports and requests a debugHandler,
		// and it needs the unsecuredMux to mount the handler onto.
		if debuggable, ok := ctrl.(controller.Debuggable); ok && unsecuredMux != nil {
			if debugHandler := debuggable.DebuggingHandler(); debugHandler != nil {
				basePath := "/debug/controllers/" + controllerName
				unsecuredMux.UnlistedHandle(basePath, http.StripPrefix(basePath, debugHandler))
				unsecuredMux.UnlistedHandlePrefix(basePath+"/", http.StripPrefix(basePath, debugHandler))
			}
		}
		if healthCheckable, ok := ctrl.(controller.HealthCheckable); ok {
			if realCheck := healthCheckable.HealthChecker(); realCheck != nil {
				check = controllerhealthz.NamedHealthChecker(controllerName, realCheck)
			}
		}

		controllers = append(controllers, ctrl)
		checks = append(checks, check)
		return nil
	}

	// Always start the SA token controller first using a full-power client, since it needs to mint tokens for the rest
	// If this fails, just return here and fail since other controllers won't be able to get credentials.
	if serviceAccountTokenControllerDescriptor, ok := controllerDescriptors[names.ServiceAccountTokenController]; ok {
		if err := buildController(serviceAccountTokenControllerDescriptor); err != nil {
			return nil, err
		}
	}

	// Each controller is passed a context where the logger has the name of
	// the controller set through WithName. That name then becomes the prefix of
	// all log messages emitted by that controller.
	//
	// In StartController, an explicit "controller" key is used instead, for two reasons:
	// - while contextual logging is alpha, klog.LoggerWithName is still a no-op,
	//   so we cannot rely on it yet to add the name
	// - it allows distinguishing between log entries emitted by the controller
	//   and those emitted for it - this is a bit debatable and could be revised.
	for _, controllerDesc := range controllerDescriptors {
		if controllerDesc.RequiresSpecialHandling() {
			continue
		}

		if !controllerCtx.IsControllerEnabled(controllerDesc) {
			logger.Info("Warning: controller is disabled", "controller", controllerDesc.Name())
			continue
		}

		if err := buildController(controllerDesc); err != nil {
			return nil, err
		}
	}

	// Register the checks.
	if len(checks) > 0 {
		healthzHandler.AddHealthChecker(checks...)
	}
	return controllers, nil
}

// RunControllers runs all controllers concurrently and blocks until the context is cancelled and all controllers are terminated.
//
// Once the context is cancelled, RunControllers waits for shutdownTimeout for all controllers to terminate.
// When the timeout is reached, the function unblocks and returns false.
// Zero shutdown timeout means that there is no timeout.
func RunControllers(ctx context.Context, controllerCtx ControllerContext, controllers []Controller,
	controllerStartJitterMaxFactor float64, shutdownTimeout time.Duration) bool {
	logger := klog.FromContext(ctx)

	// We gather running controllers names for logging purposes.
	// When the context is cancelled, the controllers still running are logged periodically.
	runningControllers := sets.New[string]()
	var runningControllersLock sync.Mutex

	loggingCtx, cancelLoggingCtx := context.WithCancel(context.Background())
	defer cancelLoggingCtx()
	go func() {
		// Only start logging when terminating.
		select {
		case <-ctx.Done():
		case <-loggingCtx.Done():
			return
		}

		// Regularly print the controllers that still haven't returned.
		logPeriod := shutdownTimeout / 3
		if logPeriod == 0 {
			logPeriod = 5 * time.Second
		}
		ticker := time.NewTicker(logPeriod)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				runningControllersLock.Lock()
				running := sets.List(runningControllers)
				runningControllersLock.Unlock()

				logger.Info("Still waiting for some controllers to terminate...", "runningControllers", running)

			case <-loggingCtx.Done():
				return
			}
		}
	}()

	terminatedCh := make(chan struct{})
	go func() {
		defer close(terminatedCh)
		var wg sync.WaitGroup
		wg.Add(len(controllers))
		for _, controller := range controllers {
			go func() {
				defer wg.Done()

				// It would be better to unblock and return on context cancelled here,
				// but that makes tests more flaky regarding timing.
				time.Sleep(wait.Jitter(controllerCtx.ComponentConfig.Generic.ControllerStartInterval.Duration, controllerStartJitterMaxFactor))

				logger.V(1).Info("Controller starting...", "controller", controller.Name())

				runningControllersLock.Lock()
				runningControllers.Insert(controller.Name())
				runningControllersLock.Unlock()

				defer func() {
					logger.V(1).Info("Controller terminated", "controller", controller.Name())

					runningControllersLock.Lock()
					runningControllers.Delete(controller.Name())
					runningControllersLock.Unlock()
				}()
				controller.Run(ctx)
			}()
		}
		wg.Wait()
		logger.Info("All controllers terminated")
	}()

	// Wait for a signal to terminate.
	select {
	case <-ctx.Done():
	case <-terminatedCh:
		return true
	}

	// Wait for the shutdown timeout.
	var shutdownCh <-chan time.Time
	if shutdownTimeout > 0 {
		shutdownCh = time.After(shutdownTimeout)
	}
	select {
	case <-terminatedCh:
		return true
	case <-shutdownCh:
		runningControllersLock.Lock()
		running := sets.List(runningControllers)
		runningControllersLock.Unlock()
		logger.Info("Controller shutdown timeout reached", "timeout", shutdownTimeout, "runningControllers", running)
		return false
	}
}

func readCA(file string) ([]byte, error) {
	rootCA, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	if _, err := certutil.ParseCertsPEM(rootCA); err != nil {
		return nil, err
	}

	return rootCA, err
}

// createClientBuilders creates clientBuilder and rootClientBuilder from the given configuration
func createClientBuilders(c *config.CompletedConfig) (clientBuilder clientbuilder.ControllerClientBuilder, rootClientBuilder clientbuilder.ControllerClientBuilder) {

	rootClientBuilder = clientbuilder.SimpleControllerClientBuilder{
		ClientConfig: c.Kubeconfig,
	}
	if c.ComponentConfig.KubeCloudShared.UseServiceAccountCredentials {

		clientBuilder = clientbuilder.NewDynamicClientBuilder(
			restclient.AnonymousClientConfig(c.Kubeconfig),
			c.Client.CoreV1(),
			metav1.NamespaceSystem)
	} else {
		clientBuilder = rootClientBuilder
	}
	return
}

// leaderElectAndRun runs the leader election, and runs the callbacks once the leader lease is acquired.
// TODO: extract this function into staging/controller-manager
func leaderElectAndRun(ctx context.Context, c *config.CompletedConfig, lockIdentity string, electionChecker *leaderelection.HealthzAdaptor, resourceLock string, leaseName string, callbacks leaderelection.LeaderCallbacks) {
	logger := klog.FromContext(ctx)
	rl, err := resourcelock.NewFromKubeconfig(resourceLock,
		c.ComponentConfig.Generic.LeaderElection.ResourceNamespace,
		leaseName,
		resourcelock.ResourceLockConfig{
			Identity:      lockIdentity,
			EventRecorder: c.EventRecorder,
		},
		c.Kubeconfig,
		c.ComponentConfig.Generic.LeaderElection.RenewDeadline.Duration)
	if err != nil {
		logger.Error(err, "Error creating lock")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	leaderelection.RunOrDie(ctx, leaderelection.LeaderElectionConfig{
		Lock:          rl,
		LeaseDuration: c.ComponentConfig.Generic.LeaderElection.LeaseDuration.Duration,
		RenewDeadline: c.ComponentConfig.Generic.LeaderElection.RenewDeadline.Duration,
		RetryPeriod:   c.ComponentConfig.Generic.LeaderElection.RetryPeriod.Duration,
		Callbacks:     callbacks,
		WatchDog:      electionChecker,
		Name:          leaseName,
		Coordinated:   utilfeature.DefaultFeatureGate.Enabled(kubefeatures.CoordinatedLeaderElection),
	})

	panic("unreachable")
}

// filteredControllerDescriptors returns all controllerDescriptors after filtering through filterFunc.
func filteredControllerDescriptors(controllerDescriptors map[string]*ControllerDescriptor, filterFunc leadermigration.FilterFunc, expected leadermigration.FilterResult) map[string]*ControllerDescriptor {
	resultControllers := make(map[string]*ControllerDescriptor)
	for name, controllerDesc := range controllerDescriptors {
		if filterFunc(name) == expected {
			resultControllers[name] = controllerDesc
		}
	}
	return resultControllers
}
