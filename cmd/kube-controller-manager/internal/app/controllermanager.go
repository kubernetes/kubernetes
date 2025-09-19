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
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/blang/semver/v4"
	coordinationv1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/mux"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	certutil "k8s.io/client-go/util/cert"
	basecompatibility "k8s.io/component-base/compatibility"
	"k8s.io/component-base/configz"
	logsapi "k8s.io/component-base/logs/api/v1"
	metricsfeatures "k8s.io/component-base/metrics/features"
	"k8s.io/component-base/metrics/prometheus/slis"
	utilversion "k8s.io/component-base/version"
	zpagesfeatures "k8s.io/component-base/zpages/features"
	"k8s.io/component-base/zpages/flagz"
	"k8s.io/component-base/zpages/statusz"
	genericcontrollermanager "k8s.io/controller-manager/app"
	kcontroller "k8s.io/controller-manager/controller"
	"k8s.io/controller-manager/pkg/clientbuilder"
	controllerhealthz "k8s.io/controller-manager/pkg/healthz"
	"k8s.io/controller-manager/pkg/leadermigration"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	kubefeatures "k8s.io/kubernetes/pkg/features"

	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller"
	controllerrun "k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller/run"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
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
				kubeControllerManager,
				statusz.NewRegistry(
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

	run := func(ctx context.Context, controllerDescriptors map[string]*controller.Descriptor) {
		controllerContext, err := controller.CreateControllerContext(ctx, c, rootClientBuilder, clientBuilder)
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
		saTokenControllerInit := saTokenControllerDescriptor.Constructor

		// Wrap saTokenControllerDescriptor to signal readiness for migration after starting the controller.
		saTokenControllerDescriptor.Constructor = func(ctx context.Context, controllerContext controller.Context, controllerName string) (controller.Controller, error) {
			ctrl, err := saTokenControllerInit(ctx, controllerContext, controllerName)
			if err != nil {
				return nil, err
			}

			// This wrapping is not exactly flawless as RunControllers uses type casting,
			// which is now not possible for the wrapped controller.
			// This fortunately doesn't matter for this particular controller.
			return controllerrun.NewControllerLoop(func(ctx context.Context) {
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
func BuildControllers(ctx context.Context, controllerCtx controller.Context, controllerDescriptors map[string]*controller.Descriptor,
	unsecuredMux *mux.PathRecorderMux, healthzHandler HealthCheckAdder) ([]controller.Controller, error) {
	logger := klog.FromContext(ctx)
	var (
		controllers []controller.Controller
		checks      []healthz.HealthChecker
	)
	buildController := func(controllerDesc *controller.Descriptor) error {
		controllerName := controllerDesc.Name
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
		if debuggable, ok := ctrl.(kcontroller.Debuggable); ok && unsecuredMux != nil {
			if debugHandler := debuggable.DebuggingHandler(); debugHandler != nil {
				basePath := "/debug/controllers/" + controllerName
				unsecuredMux.UnlistedHandle(basePath, http.StripPrefix(basePath, debugHandler))
				unsecuredMux.UnlistedHandlePrefix(basePath+"/", http.StripPrefix(basePath, debugHandler))
			}
		}
		if healthCheckable, ok := ctrl.(kcontroller.HealthCheckable); ok {
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
		if controllerDesc.RequiresSpecialHandling {
			continue
		}

		if !controllerCtx.IsControllerEnabled(controllerDesc) {
			logger.Info("Warning: controller is disabled", "controller", controllerDesc.Name)
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
func RunControllers(ctx context.Context, controllerCtx controller.Context, controllers []controller.Controller,
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
func filteredControllerDescriptors(controllerDescriptors map[string]*controller.Descriptor, filterFunc leadermigration.FilterFunc, expected leadermigration.FilterResult) map[string]*controller.Descriptor {
	resultControllers := make(map[string]*controller.Descriptor)
	for name, controllerDesc := range controllerDescriptors {
		if filterFunc(name) == expected {
			resultControllers[name] = controllerDesc
		}
	}
	return resultControllers
}
