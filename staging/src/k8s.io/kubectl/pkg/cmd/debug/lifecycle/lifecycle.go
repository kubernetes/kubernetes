/*
Copyright 2025 The Kubernetes Authors.

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

package lifecycle

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/kubernetes"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	lifecycleLong = templates.LongDesc(i18n.T(`
		Diagnose pod lifecycle issues.

		Analyzes why a pod is stuck in a problematic state (Pending,
		ImagePullBackOff, CrashLoopBackOff, etc.) by examining:

		* Pod status, conditions, and container states
		* Related events from scheduler, kubelet, and controllers
		* Node capacity and conditions (if scheduled)
		* Volume binding status and PVC events
		* Container logs from previous crashes

		Provides root cause identification and actionable recommendations.`))

	lifecycleExample = templates.Examples(i18n.T(`
		# Diagnose why a pod is stuck in Pending
		kubectl debug lifecycle nginx-pod

		# Diagnose a pod in a specific namespace
		kubectl debug lifecycle -n production api-server-pod

		# Output diagnosis in JSON format
		kubectl debug lifecycle nginx-pod -o json

		# Show detailed output including events and previous logs
		kubectl debug lifecycle nginx-pod --show-details

		# Watch pod status continuously (refresh every 5 seconds)
		kubectl debug lifecycle nginx-pod --watch

		# Watch with custom interval
		kubectl debug lifecycle nginx-pod --watch --interval 10s`))
)

// LifecycleOptions holds options for the lifecycle diagnostic command
type LifecycleOptions struct {
	Namespace     string
	PodName       string
	ShowDetails   bool
	OutputFormat  string
	Watch         bool
	WatchInterval time.Duration

	restClientGetter genericclioptions.RESTClientGetter
	clientset        kubernetes.Interface

	genericiooptions.IOStreams
}

// NewLifecycleOptions returns initialized LifecycleOptions
func NewLifecycleOptions(streams genericiooptions.IOStreams) *LifecycleOptions {
	return &LifecycleOptions{
		IOStreams:     streams,
		WatchInterval: 5 * time.Second,
	}
}

// NewCmdLifecycle creates the lifecycle diagnostic command
func NewCmdLifecycle(restClientGetter genericclioptions.RESTClientGetter, streams genericiooptions.IOStreams) *cobra.Command {
	o := NewLifecycleOptions(streams)

	cmd := &cobra.Command{
		Use:                   "lifecycle POD",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Diagnose pod lifecycle issues"),
		Long:                  lifecycleLong,
		Example:               lifecycleExample,
		Args:                  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(restClientGetter, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run(cmd.Context()))
		},
	}

	cmd.Flags().BoolVar(&o.ShowDetails, "show-details", false,
		i18n.T("If true, include events, previous container logs, and all recommendations in output."))
	cmd.Flags().StringVarP(&o.OutputFormat, "output", "o", "",
		i18n.T("Output format. One of: (json, yaml)"))
	cmd.Flags().BoolVarP(&o.Watch, "watch", "w", false,
		i18n.T("If true, continuously monitor the pod and refresh the diagnosis."))
	cmd.Flags().DurationVar(&o.WatchInterval, "interval", 5*time.Second,
		i18n.T("Interval between diagnosis refreshes when using --watch."))

	return cmd
}

// Complete fills in options from command line args
func (o *LifecycleOptions) Complete(restClientGetter genericclioptions.RESTClientGetter, cmd *cobra.Command, args []string) error {
	o.PodName = args[0]
	o.restClientGetter = restClientGetter

	var err error
	o.Namespace, _, err = restClientGetter.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	clientConfig, err := restClientGetter.ToRESTConfig()
	if err != nil {
		return err
	}

	o.clientset, err = kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	return nil
}

// Validate checks options are valid
func (o *LifecycleOptions) Validate() error {
	if o.PodName == "" {
		return fmt.Errorf("pod name is required")
	}
	if o.OutputFormat != "" && o.OutputFormat != "json" && o.OutputFormat != "yaml" {
		return fmt.Errorf("output format must be one of: json, yaml")
	}
	if o.Watch && o.WatchInterval < time.Second {
		return fmt.Errorf("watch interval must be at least 1 second")
	}
	return nil
}

// Run executes the lifecycle diagnostic
func (o *LifecycleOptions) Run(ctx context.Context) error {
	if o.Watch {
		return o.runWatch(ctx)
	}
	return o.runOnce(ctx)
}

func (o *LifecycleOptions) runOnce(ctx context.Context) error {
	// 1. Collect data
	collector := NewDataCollector(o.clientset)
	data, err := collector.Collect(ctx, o.Namespace, o.PodName)
	if err != nil {
		return fmt.Errorf("failed to collect diagnostic data: %w", err)
	}

	// 2. Analyze
	analyzer := NewAnalyzer()
	result, err := analyzer.Analyze(ctx, data)
	if err != nil {
		return fmt.Errorf("failed to analyze pod: %w", err)
	}

	// 3. Output
	printer := NewPrinter(o.IOStreams, o.OutputFormat, o.ShowDetails)
	return printer.Print(result)
}

func (o *LifecycleOptions) runWatch(ctx context.Context) error {
	collector := NewDataCollector(o.clientset)
	analyzer := NewAnalyzer()
	printer := NewPrinter(o.IOStreams, o.OutputFormat, o.ShowDetails)

	ticker := time.NewTicker(o.WatchInterval)
	defer ticker.Stop()

	// Run first diagnosis immediately
	if err := o.diagnoseAndPrint(ctx, collector, analyzer, printer); err != nil {
		return err
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			// Clear screen for human-readable output
			if o.OutputFormat == "" {
				fmt.Fprint(o.Out, "\033[H\033[2J")
			}
			if err := o.diagnoseAndPrint(ctx, collector, analyzer, printer); err != nil {
				// Pod might have been deleted, print error but continue watching
				fmt.Fprintf(o.ErrOut, "Warning: %v\n", err)
			}
		}
	}
}

func (o *LifecycleOptions) diagnoseAndPrint(ctx context.Context, collector DataCollector, analyzer Analyzer, printer *Printer) error {
	data, err := collector.Collect(ctx, o.Namespace, o.PodName)
	if err != nil {
		return fmt.Errorf("failed to collect diagnostic data: %w", err)
	}

	result, err := analyzer.Analyze(ctx, data)
	if err != nil {
		return fmt.Errorf("failed to analyze pod: %w", err)
	}

	if o.OutputFormat == "" {
		fmt.Fprintf(o.Out, "[Watch mode - refreshing every %v, Ctrl+C to stop]\n\n", o.WatchInterval)
	}

	return printer.Print(result)
}
