/*
Copyright 2022 The Kubernetes Authors.

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

// Package app does all of the work necessary to configure and run a
// Kubernetes app process.
package app

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"os/signal"
	"path"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/component-base/metrics"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/term"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/dynamic-resource-allocation/leaderelection"
	"k8s.io/klog/v2"
)

// NewCommand creates a *cobra.Command object with default parameters.
func NewCommand() *cobra.Command {
	o := logsapi.NewLoggingConfiguration()
	var clientset kubernetes.Interface
	var config *rest.Config
	ctx := context.Background()
	logger := klog.Background()

	cmd := &cobra.Command{
		Use:  "cdi-test-driver",
		Long: "cdi-test-driver implements a resource driver controller and kubelet plugin.",
	}
	sharedFlagSets := cliflag.NamedFlagSets{}
	fs := sharedFlagSets.FlagSet("logging")
	logsapi.AddFlags(o, fs)
	logs.AddFlags(fs, logs.SkipLoggingConfigurationFlags())

	fs = sharedFlagSets.FlagSet("Kubernetes client")
	kubeconfig := fs.String("kubeconfig", "", "Absolute path to the kube.config file. Either this or KUBECONFIG need to be set if the driver is being run out of cluster.")
	kubeAPIQPS := fs.Float32("kube-api-qps", 50, "QPS to use while communicating with the kubernetes apiserver.")
	kubeAPIBurst := fs.Int("kube-api-burst", 100, "Burst to use while communicating with the kubernetes apiserver.")
	workers := fs.Int("workers", 10, "Concurrency to process multiple claims")

	fs = sharedFlagSets.FlagSet("http server")
	httpEndpoint := fs.String("http-endpoint", "",
		"The TCP network address where the HTTP server for diagnostics, including pprof, metrics and (if applicable) leader election health check, will listen (example: `:8080`). The default is the empty string, which means the server is disabled.")
	metricsPath := fs.String("metrics-path", "/metrics", "The HTTP path where Prometheus metrics will be exposed, disabled if empty.")
	profilePath := fs.String("pprof-path", "", "The HTTP path where pprof profiling will be available, disabled if empty.")

	fs = sharedFlagSets.FlagSet("CDI")
	driverNameFlagName := "drivername"
	driverName := fs.String(driverNameFlagName, "test-driver.cdi.k8s.io", "Resource driver name.")
	driverNameFlag := fs.Lookup(driverNameFlagName)

	fs = sharedFlagSets.FlagSet("other")
	featureGate := featuregate.NewFeatureGate()
	utilruntime.Must(logsapi.AddFeatureGates(featureGate))
	featureGate.AddFlag(fs)

	fs = cmd.PersistentFlags()
	for _, f := range sharedFlagSets.FlagSets {
		fs.AddFlagSet(f)
	}

	mux := http.NewServeMux()

	cmd.PersistentPreRunE = func(cmd *cobra.Command, args []string) error {
		// Activate logging as soon as possible, after that
		// show flags with the final logging configuration.

		if err := logsapi.ValidateAndApply(o, featureGate); err != nil {
			return err
		}

		// get the KUBECONFIG from env if specified (useful for local/debug cluster)
		kubeconfigEnv := os.Getenv("KUBECONFIG")

		if kubeconfigEnv != "" {
			logger.Info("Found KUBECONFIG environment variable set, using that..")
			*kubeconfig = kubeconfigEnv
		}

		var err error
		if *kubeconfig == "" {
			config, err = rest.InClusterConfig()
			if err != nil {
				return fmt.Errorf("create in-cluster client configuration: %w", err)
			}
		} else {
			config, err = clientcmd.BuildConfigFromFlags("", *kubeconfig)
			if err != nil {
				return fmt.Errorf("create out-of-cluster client configuration: %w", err)
			}
		}
		config.QPS = *kubeAPIQPS
		config.Burst = *kubeAPIBurst

		clientset, err = kubernetes.NewForConfig(config)
		if err != nil {
			return fmt.Errorf("create client: %w", err)
		}

		if *httpEndpoint != "" {
			if *metricsPath != "" {
				// For workqueue and leader election metrics, set up via the anonymous imports of:
				// https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/component-base/metrics/prometheus/workqueue/metrics.go
				// https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/component-base/metrics/prometheus/clientgo/leaderelection/metrics.go
				//
				// Also to happens to include Go runtime and process metrics:
				// https://github.com/kubernetes/kubernetes/blob/9780d88cb6a4b5b067256ecb4abf56892093ee87/staging/src/k8s.io/component-base/metrics/legacyregistry/registry.go#L46-L49
				gatherer := legacyregistry.DefaultGatherer
				actualPath := path.Join("/", *metricsPath)
				logger.Info("Starting metrics", "path", actualPath)
				mux.Handle(actualPath,
					metrics.HandlerFor(gatherer, metrics.HandlerOpts{}))
			}

			if *profilePath != "" {
				actualPath := path.Join("/", *profilePath)
				logger.Info("Starting profiling", "path", actualPath)
				mux.HandleFunc(path.Join("/", *profilePath), pprof.Index)
				mux.HandleFunc(path.Join("/", *profilePath, "cmdline"), pprof.Cmdline)
				mux.HandleFunc(path.Join("/", *profilePath, "profile"), pprof.Profile)
				mux.HandleFunc(path.Join("/", *profilePath, "symbol"), pprof.Symbol)
				mux.HandleFunc(path.Join("/", *profilePath, "trace"), pprof.Trace)
			}

			listener, err := net.Listen("tcp", *httpEndpoint)
			if err != nil {
				return fmt.Errorf("listen on HTTP endpoint: %w", err)
			}

			go func() {
				logger.Info("Starting HTTP server", "endpoint", *httpEndpoint)
				err := http.Serve(listener, mux)
				if err != nil {
					logger.Error(err, "HTTP server failed")
					klog.FlushAndExit(klog.ExitFlushTimeout, 1)
				}
			}()
		}

		return nil
	}

	controller := &cobra.Command{
		Use:   "controller",
		Short: "run as resource controller",
		Long:  "cdi-test-driver controller runs as a resource driver controller.",
		Args:  cobra.ExactArgs(0),
	}
	controllerFlagSets := cliflag.NamedFlagSets{}
	fs = controllerFlagSets.FlagSet("leader election")
	enableLeaderElection := fs.Bool("leader-election", false,
		"Enables leader election. If leader election is enabled, additional RBAC rules are required.")
	leaderElectionNamespace := fs.String("leader-election-namespace", "",
		"Namespace where the leader election resource lives. Defaults to the pod namespace if not set.")
	leaderElectionLeaseDuration := fs.Duration("leader-election-lease-duration", 15*time.Second,
		"Duration, in seconds, that non-leader candidates will wait to force acquire leadership.")
	leaderElectionRenewDeadline := fs.Duration("leader-election-renew-deadline", 10*time.Second,
		"Duration, in seconds, that the acting leader will retry refreshing leadership before giving up.")
	leaderElectionRetryPeriod := fs.Duration("leader-election-retry-period", 5*time.Second,
		"Duration, in seconds, the LeaderElector clients should wait between tries of actions.")
	fs = controllerFlagSets.FlagSet("controller")
	resourceConfig := fs.String("resource-config", "", "A JSON file containing a Resources struct. Defaults are unshared, network-attached resources.")
	fs = controller.Flags()
	for _, f := range controllerFlagSets.FlagSets {
		fs.AddFlagSet(f)
	}

	controller.RunE = func(cmd *cobra.Command, args []string) error {
		resources := Resources{}
		if *resourceConfig != "" {
			file, err := os.Open(*resourceConfig)
			if err != nil {
				return fmt.Errorf("open resource config: %w", err)
			}
			decoder := json.NewDecoder(file)
			decoder.DisallowUnknownFields()
			if err := decoder.Decode(&resources); err != nil {
				return fmt.Errorf("parse resource config %q: %w", *resourceConfig, err)
			}
		}
		if resources.DriverName == "" || driverNameFlag.Changed {
			resources.DriverName = *driverName
		}

		run := func() {
			controller := NewController(clientset, resources)
			controller.Run(ctx, *workers)
		}

		if !*enableLeaderElection {
			run()
			return nil
		}

		// This must not change between releases.
		lockName := *driverName

		// Create a new clientset for leader election
		// to avoid starving it when the normal traffic
		// exceeds the QPS+burst limits.
		leClientset, err := kubernetes.NewForConfig(config)
		if err != nil {
			return fmt.Errorf("create leaderelection client: %w", err)
		}

		le := leaderelection.New(leClientset, lockName,
			func(ctx context.Context) {
				run()
			},
			leaderelection.LeaseDuration(*leaderElectionLeaseDuration),
			leaderelection.RenewDeadline(*leaderElectionRenewDeadline),
			leaderelection.RetryPeriod(*leaderElectionRetryPeriod),
			leaderelection.Namespace(*leaderElectionNamespace),
		)
		if *httpEndpoint != "" {
			le.PrepareHealthCheck(mux)
		}
		if err := le.Run(); err != nil {
			return fmt.Errorf("leader election failed: %w", err)
		}

		return nil
	}
	cmd.AddCommand(controller)

	kubeletPlugin := &cobra.Command{
		Use:   "kubelet-plugin",
		Short: "run as kubelet plugin",
		Long:  "cdi-test-driver kubelet-plugin runs as a device plugin for kubelet that supports dynamic resource allocation.",
		Args:  cobra.ExactArgs(0),
	}
	kubeletPluginFlagSets := cliflag.NamedFlagSets{}
	fs = kubeletPluginFlagSets.FlagSet("kubelet")
	pluginRegistrationPath := fs.String("plugin-registration-path", "/var/lib/kubelet/plugins_registry", "The directory where kubelet looks for plugin registration sockets, in the filesystem of the driver.")
	endpoint := fs.String("endpoint", "/var/lib/kubelet/plugins/test-driver/dra.sock", "The Unix domain socket where the driver will listen for kubelet requests, in the filesystem of the driver.")
	draAddress := fs.String("dra-address", "/var/lib/kubelet/plugins/test-driver/dra.sock", "The Unix domain socket that kubelet will connect to for dynamic resource allocation requests, in the filesystem of kubelet.")
	fs = kubeletPluginFlagSets.FlagSet("CDI")
	cdiDir := fs.String("cdi-dir", "/var/run/cdi", "directory for dynamically created CDI JSON files")
	fs = kubeletPlugin.Flags()
	for _, f := range kubeletPluginFlagSets.FlagSets {
		fs.AddFlagSet(f)
	}
	kubeletPlugin.RunE = func(cmd *cobra.Command, args []string) error {
		// Ensure that directories exist, creating them if necessary. We want
		// to know early if there is a setup problem that would prevent
		// creating those directories.
		if err := os.MkdirAll(*cdiDir, os.FileMode(0750)); err != nil {
			return fmt.Errorf("create CDI directory: %w", err)
		}
		if err := os.MkdirAll(filepath.Dir(*endpoint), 0750); err != nil {
			return fmt.Errorf("create socket directory: %w", err)
		}

		plugin, err := StartPlugin(cmd.Context(), *cdiDir, *driverName, "", FileOperations{},
			kubeletplugin.PluginSocketPath(*endpoint),
			kubeletplugin.RegistrarSocketPath(path.Join(*pluginRegistrationPath, *driverName+"-reg.sock")),
			kubeletplugin.KubeletPluginSocketPath(*draAddress),
		)
		if err != nil {
			return fmt.Errorf("start example plugin: %w", err)
		}

		// Handle graceful shutdown. We need to delete Unix domain
		// sockets.
		sigc := make(chan os.Signal, 1)
		signal.Notify(sigc, os.Interrupt, syscall.SIGTERM)
		logger.Info("Waiting for signal.")
		sig := <-sigc
		logger.Info("Received signal, shutting down.", "signal", sig)
		plugin.Stop()
		return nil
	}
	cmd.AddCommand(kubeletPlugin)

	// SetUsageAndHelpFunc takes care of flag grouping. However,
	// it doesn't support listing child commands. We add those
	// to cmd.Use.
	cols, _, _ := term.TerminalSize(cmd.OutOrStdout())
	cliflag.SetUsageAndHelpFunc(cmd, sharedFlagSets, cols)
	var children []string
	for _, child := range cmd.Commands() {
		children = append(children, child.Use)
	}
	cmd.Use += " [shared flags] " + strings.Join(children, "|")
	cliflag.SetUsageAndHelpFunc(controller, controllerFlagSets, cols)
	cliflag.SetUsageAndHelpFunc(kubeletPlugin, kubeletPluginFlagSets, cols)

	return cmd
}
