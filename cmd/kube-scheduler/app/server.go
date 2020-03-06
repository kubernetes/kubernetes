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

// Package app implements a Server object for running the scheduler.
package app

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	goruntime "runtime"

	"github.com/spf13/cobra"

	"k8s.io/api/core/v1"
	eventsv1beta1 "k8s.io/api/events/v1beta1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapifilters "k8s.io/apiserver/pkg/endpoints/filters"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfilters "k8s.io/apiserver/pkg/server/filters"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/apiserver/pkg/server/routes"
	"k8s.io/apiserver/pkg/util/term"
	"k8s.io/client-go/kubernetes/scheme"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/record"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/component-base/logs"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/version"
	"k8s.io/component-base/version/verflag"
	"k8s.io/klog"
	schedulerserverconfig "k8s.io/kubernetes/cmd/kube-scheduler/app/config"
	"k8s.io/kubernetes/cmd/kube-scheduler/app/options"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/scheduler"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"k8s.io/kubernetes/pkg/util/configz"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
)

// Option configures a framework.Registry.
type Option func(framework.Registry) error

// NewSchedulerCommand creates a *cobra.Command object with default parameters and registryOptions
func NewSchedulerCommand(registryOptions ...Option) *cobra.Command {
	opts, err := options.NewOptions()
	if err != nil {
		klog.Fatalf("unable to initialize command options: %v", err)
	}

	cmd := &cobra.Command{
		Use: "kube-scheduler",
		Long: `The Kubernetes scheduler is a policy-rich, topology-aware,
workload-specific function that significantly impacts availability, performance,
and capacity. The scheduler needs to take into account individual and collective
resource requirements, quality of service requirements, hardware/software/policy
constraints, affinity and anti-affinity specifications, data locality, inter-workload
interference, deadlines, and so on. Workload-specific requirements will be exposed
through the API as necessary. See [scheduling](https://kubernetes.io/docs/concepts/scheduling/)
for more information about scheduling and the kube-scheduler component.`,
		Run: func(cmd *cobra.Command, args []string) {
			if err := runCommand(cmd, args, opts, registryOptions...); err != nil {
				fmt.Fprintf(os.Stderr, "%v\n", err)
				os.Exit(1)
			}
		},
	}
	fs := cmd.Flags()
	namedFlagSets := opts.Flags()
	verflag.AddFlags(namedFlagSets.FlagSet("global"))
	globalflag.AddGlobalFlags(namedFlagSets.FlagSet("global"), cmd.Name())
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
	cmd.MarkFlagFilename("config", "yaml", "yml", "json")

	return cmd
}

// runCommand runs the scheduler.
func runCommand(cmd *cobra.Command, args []string, opts *options.Options, registryOptions ...Option) error {
	verflag.PrintAndExitIfRequested()
	utilflag.PrintFlags(cmd.Flags())

	if len(args) != 0 {
		fmt.Fprint(os.Stderr, "arguments are not supported\n")
	}

	if errs := opts.Validate(); len(errs) > 0 {
		return utilerrors.NewAggregate(errs)
	}

	if len(opts.WriteConfigTo) > 0 {
		c := &schedulerserverconfig.Config{}
		if err := opts.ApplyTo(c); err != nil {
			return err
		}
		if err := options.WriteConfigFile(opts.WriteConfigTo, &c.ComponentConfig); err != nil {
			return err
		}
		klog.Infof("Wrote configuration to: %s\n", opts.WriteConfigTo)
		return nil
	}

	c, err := opts.Config()
	if err != nil {
		return err
	}

	// Get the completed config
	cc := c.Complete()

	// Configz registration.
	if cz, err := configz.New("componentconfig"); err == nil {
		cz.Set(cc.ComponentConfig)
	} else {
		return fmt.Errorf("unable to register configz: %s", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	return Run(ctx, cc, registryOptions...)
}

// Run executes the scheduler based on the given configuration. It only returns on error or when context is done.
func Run(ctx context.Context, cc schedulerserverconfig.CompletedConfig, outOfTreeRegistryOptions ...Option) error {
	// To help debugging, immediately log version
	klog.V(1).Infof("Starting Kubernetes Scheduler version %+v", version.Get())

	outOfTreeRegistry := make(framework.Registry)
	for _, option := range outOfTreeRegistryOptions {
		if err := option(outOfTreeRegistry); err != nil {
			return err
		}
	}

	recorderFactory := getRecorderFactory(&cc)
	// Create the scheduler.
	sched, err := scheduler.New(cc.Client,
		cc.InformerFactory,
		cc.PodInformer,
		recorderFactory,
		ctx.Done(),
		scheduler.WithProfiles(cc.ComponentConfig.Profiles...),
		scheduler.WithAlgorithmSource(cc.ComponentConfig.AlgorithmSource),
		scheduler.WithPreemptionDisabled(cc.ComponentConfig.DisablePreemption),
		scheduler.WithPercentageOfNodesToScore(cc.ComponentConfig.PercentageOfNodesToScore),
		scheduler.WithBindTimeoutSeconds(cc.ComponentConfig.BindTimeoutSeconds),
		scheduler.WithFrameworkOutOfTreeRegistry(outOfTreeRegistry),
		scheduler.WithPodMaxBackoffSeconds(cc.ComponentConfig.PodMaxBackoffSeconds),
		scheduler.WithPodInitialBackoffSeconds(cc.ComponentConfig.PodInitialBackoffSeconds),
		scheduler.WithExtenders(cc.ComponentConfig.Extenders...),
	)
	if err != nil {
		return err
	}

	// Prepare the event broadcaster.
	if cc.Broadcaster != nil && cc.EventClient != nil {
		cc.Broadcaster.StartRecordingToSink(ctx.Done())
	}
	if cc.CoreBroadcaster != nil && cc.CoreEventClient != nil {
		cc.CoreBroadcaster.StartRecordingToSink(&corev1.EventSinkImpl{Interface: cc.CoreEventClient.Events("")})
	}
	// Setup healthz checks.
	var checks []healthz.HealthChecker
	if cc.ComponentConfig.LeaderElection.LeaderElect {
		checks = append(checks, cc.LeaderElection.WatchDog)
	}

	// Start up the healthz server.
	if cc.InsecureServing != nil {
		separateMetrics := cc.InsecureMetricsServing != nil
		handler := buildHandlerChain(newHealthzHandler(&cc.ComponentConfig, separateMetrics, checks...), nil, nil)
		if err := cc.InsecureServing.Serve(handler, 0, ctx.Done()); err != nil {
			return fmt.Errorf("failed to start healthz server: %v", err)
		}
	}
	if cc.InsecureMetricsServing != nil {
		handler := buildHandlerChain(newMetricsHandler(&cc.ComponentConfig), nil, nil)
		if err := cc.InsecureMetricsServing.Serve(handler, 0, ctx.Done()); err != nil {
			return fmt.Errorf("failed to start metrics server: %v", err)
		}
	}
	if cc.SecureServing != nil {
		handler := buildHandlerChain(newHealthzHandler(&cc.ComponentConfig, false, checks...), cc.Authentication.Authenticator, cc.Authorization.Authorizer)
		// TODO: handle stoppedCh returned by c.SecureServing.Serve
		if _, err := cc.SecureServing.Serve(handler, 0, ctx.Done()); err != nil {
			// fail early for secure handlers, removing the old error loop from above
			return fmt.Errorf("failed to start secure server: %v", err)
		}
	}

	// Start all informers.
	go cc.PodInformer.Informer().Run(ctx.Done())
	cc.InformerFactory.Start(ctx.Done())

	// Wait for all caches to sync before scheduling.
	cc.InformerFactory.WaitForCacheSync(ctx.Done())

	// If leader election is enabled, runCommand via LeaderElector until done and exit.
	if cc.LeaderElection != nil {
		cc.LeaderElection.Callbacks = leaderelection.LeaderCallbacks{
			OnStartedLeading: sched.Run,
			OnStoppedLeading: func() {
				klog.Fatalf("leaderelection lost")
			},
		}
		leaderElector, err := leaderelection.NewLeaderElector(*cc.LeaderElection)
		if err != nil {
			return fmt.Errorf("couldn't create leader elector: %v", err)
		}

		leaderElector.Run(ctx)

		return fmt.Errorf("lost lease")
	}

	// Leader election is disabled, so runCommand inline until done.
	sched.Run(ctx)
	return fmt.Errorf("finished without leader elect")
}

// buildHandlerChain wraps the given handler with the standard filters.
func buildHandlerChain(handler http.Handler, authn authenticator.Request, authz authorizer.Authorizer) http.Handler {
	requestInfoResolver := &apirequest.RequestInfoFactory{}
	failedHandler := genericapifilters.Unauthorized(legacyscheme.Codecs, false)

	handler = genericapifilters.WithAuthorization(handler, authz, legacyscheme.Codecs)
	handler = genericapifilters.WithAuthentication(handler, authn, failedHandler, nil)
	handler = genericapifilters.WithRequestInfo(handler, requestInfoResolver)
	handler = genericapifilters.WithCacheControl(handler)
	handler = genericfilters.WithPanicRecovery(handler)

	return handler
}

func installMetricHandler(pathRecorderMux *mux.PathRecorderMux) {
	configz.InstallHandler(pathRecorderMux)
	//lint:ignore SA1019 See the Metrics Stability Migration KEP
	defaultMetricsHandler := legacyregistry.Handler().ServeHTTP
	pathRecorderMux.HandleFunc("/metrics", func(w http.ResponseWriter, req *http.Request) {
		if req.Method == "DELETE" {
			metrics.Reset()
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			w.Header().Set("X-Content-Type-Options", "nosniff")
			io.WriteString(w, "metrics reset\n")
			return
		}
		defaultMetricsHandler(w, req)
	})
}

// newMetricsHandler builds a metrics server from the config.
func newMetricsHandler(config *kubeschedulerconfig.KubeSchedulerConfiguration) http.Handler {
	pathRecorderMux := mux.NewPathRecorderMux("kube-scheduler")
	installMetricHandler(pathRecorderMux)
	if config.EnableProfiling {
		routes.Profiling{}.Install(pathRecorderMux)
		if config.EnableContentionProfiling {
			goruntime.SetBlockProfileRate(1)
		}
		routes.DebugFlags{}.Install(pathRecorderMux, "v", routes.StringFlagPutHandler(logs.GlogSetter))
	}
	return pathRecorderMux
}

// newHealthzHandler creates a healthz server from the config, and will also
// embed the metrics handler if the healthz and metrics address configurations
// are the same.
func newHealthzHandler(config *kubeschedulerconfig.KubeSchedulerConfiguration, separateMetrics bool, checks ...healthz.HealthChecker) http.Handler {
	pathRecorderMux := mux.NewPathRecorderMux("kube-scheduler")
	healthz.InstallHandler(pathRecorderMux, checks...)
	if !separateMetrics {
		installMetricHandler(pathRecorderMux)
	}
	if config.EnableProfiling {
		routes.Profiling{}.Install(pathRecorderMux)
		if config.EnableContentionProfiling {
			goruntime.SetBlockProfileRate(1)
		}
		routes.DebugFlags{}.Install(pathRecorderMux, "v", routes.StringFlagPutHandler(logs.GlogSetter))
	}
	return pathRecorderMux
}

func getRecorderFactory(cc *schedulerserverconfig.CompletedConfig) profile.RecorderFactory {
	if _, err := cc.Client.Discovery().ServerResourcesForGroupVersion(eventsv1beta1.SchemeGroupVersion.String()); err == nil {
		cc.Broadcaster = events.NewBroadcaster(&events.EventSinkImpl{Interface: cc.EventClient.Events("")})
		return profile.NewRecorderFactory(cc.Broadcaster)
	}
	return func(name string) events.EventRecorder {
		r := cc.CoreBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: name})
		return record.NewEventRecorderAdapter(r)
	}
}

// WithPlugin creates an Option based on plugin name and factory. Please don't remove this function: it is used to register out-of-tree plugins,
// hence there are no references to it from the kubernetes scheduler code base.
func WithPlugin(name string, factory framework.PluginFactory) Option {
	return func(registry framework.Registry) error {
		return registry.Register(name, factory)
	}
}
