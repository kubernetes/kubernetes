/*
Copyright 2023 The Kubernetes Authors.

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

package clusterinfo

import (
	"bytes"
	"fmt"
	"os"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	appsv1client "k8s.io/client-go/kubernetes/typed/apps/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/klog/v2"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/rawhttp"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type ClusterInfoShowFeatureGatesOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	PrintObj   printers.ResourcePrinterFunc

	All              bool
	Timeout          time.Duration
	AppsClient       appsv1client.AppsV1Interface
	CoreClient       corev1client.CoreV1Interface
	Namespace        string
	RESTClientGetter genericclioptions.RESTClientGetter
	LogsForObject    polymorphichelpers.LogsForObjectFunc

	genericclioptions.IOStreams
}

func NewCmdClusterInfoShowFeatureGates(restClientGetter genericclioptions.RESTClientGetter, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := &ClusterInfoShowFeatureGatesOptions{
		PrintFlags: genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme).WithDefaultOutput("json"),

		IOStreams: ioStreams,
	}

	cmd := &cobra.Command{
		Use:     "show-feature-gates",
		Short:   i18n.T("Show feature gates status"),
		Long:    showFeatureGatesLong,
		Example: showFeatureGatesExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(restClientGetter, cmd))
			cmdutil.CheckErr(o.Run(restClientGetter.(cmdutil.Factory)))
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().BoolVarP(&o.All, "all", "A", o.All, "If true, showFeatureGates all feature gates.  Default false, only shows enabled alpha features and disabled beta features.")
	cmdutil.AddPodRunningTimeoutFlag(cmd, defaultPodLogsTimeout)
	return cmd
}

var (
	showFeatureGatesLong = templates.LongDesc(i18n.T(`
    Show feature gates status of cluster.  By default, only shows enabled alpha features and disabled beta features. You can optionally specify a directory with --output-directory.  If you specify a directory, Kubernetes will
	Specify --all to show all feature gates.`))

	showFeatureGatesExample = templates.Examples(i18n.T(`
    # Show feature gates status that cluster admin should pay attention to stdout
	# Only shows enabled alpha features and disabled beta features
    kubectl cluster-info show-feature-gates

    # Show all feature gates status to stdout
    kubectl cluster-info show-feature-gates --all`))
)

func (o *ClusterInfoShowFeatureGatesOptions) Complete(restClientGetter genericclioptions.RESTClientGetter, cmd *cobra.Command) error {
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObj = printer.PrintObj

	config, err := restClientGetter.ToRESTConfig()
	if err != nil {
		return err
	}

	o.CoreClient, err = corev1client.NewForConfig(config)
	if err != nil {
		return err
	}

	o.AppsClient, err = appsv1client.NewForConfig(config)
	if err != nil {
		return err
	}

	o.Timeout, err = cmdutil.GetPodRunningTimeoutFlag(cmd)
	if err != nil {
		return err
	}

	o.Namespace, _, err = restClientGetter.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.RESTClientGetter = restClientGetter
	o.LogsForObject = polymorphichelpers.LogsForObjectFn

	return nil
}

type Stage string

type fgStatus struct {
	name    string
	stage   Stage
	enabled bool
}

// feature gates status can be got from the apiserver metrics
// 1. use `kubectl get --raw /metrics` to get the metrics.
// 2. filter kubernetes_feature_enabled metrics and save it into a map, and the map key is
// the feature gate name, the value is a struct contains name, stage and its status.
// The number in the metrics is the status of the feature gate: 0 for enabled, 1 for disabled.
// The metric is like `kubernetes_feature_enabled{name="ProcMountType",stage="ALPHA"} 0`
// 3. Print them accordingly.
// if All is true, keep the map and print them all
// if All is false, filter those feature gates that is in stage ALPHA and disabled, and also filter
// those feature gates that is in stage BETA and enabled, and filter all feature GA
func (o *ClusterInfoShowFeatureGatesOptions) Run(f cmdutil.Factory) error {
	restClient, err := f.RESTClient()
	if err != nil {
		return err
	}
	out := &bytes.Buffer{}
	err = rawhttp.RawGet(restClient, genericclioptions.IOStreams{In: os.Stdin, Out: out, ErrOut: os.Stderr}, "/metrics")
	if err != nil {
		return err
	}

	fgMap, err := readAllFGs(out)
	if err != nil {
		return err
	}

	o.print(fgMap)
	return nil
}

func readAllFGs(out *bytes.Buffer) (map[string]fgStatus, error) {
	fgMap := map[string]fgStatus{}
	p := expfmt.TextParser{}
	metricFamilies, err := p.TextToMetricFamilies(out)
	if err != nil {
		return nil, err
	}
	featureEnabled := metricFamilies["kubernetes_feature_enabled"]
	for _, metric := range featureEnabled.Metric {
		fgMetric := parseFGMetric(metric)
		if fgMetric != nil {
			fgMap[fgMetric.name] = *fgMetric
		}
	}
	klog.V(5).Infof("parsed %d feature gates successfully", len(fgMap))
	return fgMap, nil
}

func parseFGMetric(metric *dto.Metric) *fgStatus {
	var fgName, stageStr string
	labels := metric.Label
	for _, label := range labels {
		switch *label.Name {
		case "name":
			fgName = *label.Value
		case "stage":
			switch *label.Value {
			case "ALPHA", "BETA", "DEPRECATED":
				stageStr = *label.Value
			case "":
				stageStr = "GA"
			default:
				stageStr = ""
			}
		}
	}
	if len(fgName) == 0 {
		return nil
	}
	return &fgStatus{
		name:    fgName,
		stage:   Stage(stageStr),
		enabled: int(*metric.GetGauge().Value) == 1,
	}
}

// Generally, feature gate is disabled in ALPHA stage by default, and is enabled if it is BETA or GA.
// More details can refer to https://kubernetes.io/docs/reference/command-line-tools-reference/feature-gates/#feature-stages
func (fgs *fgStatus) isGeneral() bool {
	return (fgs.stage == "ALPHA" && !fgs.enabled) ||
		(fgs.stage == "BETA" && fgs.enabled) ||
		(fgs.stage == "GA" && fgs.enabled)
}

func (o *ClusterInfoShowFeatureGatesOptions) print(fgMap map[string]fgStatus) {
	if o.All {
		for _, fg := range fgMap {
			o.printGateStatus(fg)
		}
	} else {
		klog.V(5).Infof("parsed %d feature gates successfully", len(fgMap))
		for _, fg := range fgMap {
			if fg.isGeneral() {
				continue
			}
			o.printGateStatus(fg)
		}
	}
}

func (o *ClusterInfoShowFeatureGatesOptions) printGateStatus(fg fgStatus) {
	fmt.Fprintf(o.Out, "%s=%v (%s)\n", fg.name, fg.enabled, fg.stage)
}
