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
	"bufio"
	"bytes"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	appsv1client "k8s.io/client-go/kubernetes/typed/apps/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
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

	fgMap := map[string]fgStatus{}
	scanner := bufio.NewScanner(out)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "kubernetes_feature_enabled") {
			fields := strings.Fields(line)
			if len(fields) == 2 {
				fgName := strings.TrimPrefix(fields[0], "kubernetes_feature_enabled{name=\"")
				fgName = strings.Split(fgName, "\",stage=")[0]
				stageStr := "unknown"
				if strings.Contains(fields[0], "stage=\"ALPHA\"") {
					stageStr = "ALPHA"
				} else if strings.Contains(fields[0], "stage=\"BETA\"") {
					stageStr = "BETA"
				} else if strings.Contains(fields[0], "stage=\"\"") {
					stageStr = "GA"
				} else {
					stageStr = ""
				}
				status, err := strconv.Atoi(fields[1])
				if err != nil {
					return err
				}
				if _, ok := fgMap[fgName]; !ok {
					fgMap[fgName] = fgStatus{
						name:    fgName,
						stage:   Stage(stageStr),
						enabled: status == 1,
					}
				}

			}
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	// Output map
	if o.All {
		for _, fg := range fgMap {
			o.printGateStatus(fg)
		}
	} else {
		for _, fg := range fgMap {
			if fg.stage == "GA" ||
				(fg.stage == "ALPHA" && !fg.enabled) ||
				(fg.stage == "BETA" && fg.enabled) {
				continue
			}
			o.printGateStatus(fg)
		}
	}

	return nil
}

func (o *ClusterInfoShowFeatureGatesOptions) printGateStatus(fg fgStatus) {
	fmt.Fprintf(o.Out, "%s=%v (%s)\n", fg.name, fg.enabled, fg.stage)
}

type Stage string

type fgStatus struct {
	name    string
	stage   Stage
	enabled bool
}
