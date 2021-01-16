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

package top

import (
	"errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/metricsutil"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	custommetrics "k8s.io/metrics/pkg/apis/custom_metrics"

	"github.com/spf13/cobra"
	"k8s.io/cli-runtime/pkg/genericclioptions"
)

type TopMetricsOptions struct {
	Printer             *metricsutil.TopCmdPrinter
	DiscoveryClient    discovery.DiscoveryInterface

	NoHeaders bool
	Kind      string

	genericclioptions.IOStreams
}


var (
	topMetricsLong = templates.LongDesc(i18n.T(`
		Display list of available metrics.

		The 'top metrics' command allows you to see the list of available metrics that can be used .
.`))

	topMetricsExample = templates.Examples(i18n.T(`
		# show all metrics available to kubectl top
		kubectl top metrics

		# show metrics available to kubectl top pods
		kubectl top metrics pods
        `))
)

func NewCmdTopMetrics(f cmdutil.Factory, o *TopMetricsOptions, streams genericclioptions.IOStreams) *cobra.Command {
	if o == nil {
		o = &TopMetricsOptions{
			IOStreams: streams,
		}
	}

	cmd := &cobra.Command{
		Use:                   "metrics [NAME]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Display list of available metrics"),
		Long:                  topMetricsLong,
		Example:               topMetricsExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunTopMetrics())
		},
		Aliases: []string{"metrics", "m"},
	}
	cmd.Flags().BoolVar(&o.NoHeaders, "no-headers", o.NoHeaders, "If present, print output without headers")
	return cmd
}

func (o *TopMetricsOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(args) == 1 {
		o.Kind = args[0]
	} else if len(args) > 1 {
		return cmdutil.UsageErrorf(cmd, "%s", cmd.Use)
	}

	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.DiscoveryClient = clientset.DiscoveryClient
	o.Printer = metricsutil.NewTopCmdPrinter(o.Out)
	return nil
}

func (o *TopMetricsOptions) Validate() error {
	return nil
}

func (o TopMetricsOptions) RunTopMetrics() error {
	var err error
	apiGroups, err := o.DiscoveryClient.ServerGroups()
	if err != nil {
		return err
	}

	apiAvailable := SupportedAPIVersionAvailable(apiGroups, custommetrics.GroupName)
	if !apiAvailable {
		return errors.New("Custom Metrics API not available")
	}

	metrics, err := o.DiscoveryClient.ServerResourcesForGroupVersion(schema.GroupVersion{
		Group:   "custom.metrics.k8s.io",
		Version: "v1beta2",
	}.String())

	if err != nil {
		return err
	}

	return o.Printer.PrintMetrics(metrics.APIResources, o.NoHeaders, o.Kind)
}
