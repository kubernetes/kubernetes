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
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
)

const (
	sortByCPU    = "cpu"
	sortByMemory = "memory"
)

var (
	supportedMetricsAPIVersions = []string{
		"v1beta1",
	}
	topLong = templates.LongDesc(i18n.T(`
		Display resource (CPU/memory) usage.

		This command provides a view of recent resource consumption for nodes and pods.
		It fetches metrics from the Metrics Server, which aggregates this data from the
		kubelet on each node. The Metrics Server must be installed and running in the
		cluster for this command to work.

		The metrics shown are specifically optimized for Kubernetes autoscaling
		decisions, such as those made by the Horizontal Pod Autoscaler (HPA) and
		Vertical Pod Autoscaler (VPA). Because of this, the values may not match those
		from standard OS tools like 'top', as the metrics are designed to provide a
		stable signal for autoscalers rather than for pinpoint accuracy.

		When to use this command:

		* For on-the-fly spot-checks of resource usage (e.g. identify which pods
		  are consuming the most resources at a glance, or get a quick sense of the load
		  on your nodes)
		* Understand current resource consumption patterns
		* Validate the behavior of your HPA or VPA configurations by seeing the metrics
		  they use for scaling decisions.

		It is not intended to be a replacement for full-featured monitoring solutions.
		Its primary design goal is to provide a low-overhead signal for autoscalers,
		not to be a perfectly accurate monitoring tool. For high-accuracy reporting,
		historical analysis, dashboarding, or alerting, you should use a dedicated
		monitoring solution.`))
)

func NewCmdTop(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "top",
		Short: i18n.T("Display resource (CPU/memory) usage"),
		Long:  topLong,
		Run:   cmdutil.DefaultSubCommandRun(streams.ErrOut),
	}

	// create subcommands
	cmd.AddCommand(NewCmdTopNode(f, nil, streams))
	cmd.AddCommand(NewCmdTopPod(f, nil, streams))

	return cmd
}

func SupportedMetricsAPIVersionAvailable(discoveredAPIGroups *metav1.APIGroupList) bool {
	for _, discoveredAPIGroup := range discoveredAPIGroups.Groups {
		if discoveredAPIGroup.Name != metricsapi.GroupName {
			continue
		}
		for _, version := range discoveredAPIGroup.Versions {
			for _, supportedVersion := range supportedMetricsAPIVersions {
				if version.Version == supportedVersion {
					return true
				}
			}
		}
	}
	return false
}
