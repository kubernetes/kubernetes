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
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
)

var (
	supportedMetricsAPIVersions = []string{
		"v1beta1",
	}
	topLong = templates.LongDesc(i18n.T(`
		Display Resource (CPU/Memory/Storage) usage.

		The top command allows you to see the resource consumption for nodes or pods.

		This command requires Heapster to be correctly configured and working on the server. `))
)

func NewCmdTop(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "top",
		Short: i18n.T("Display Resource (CPU/Memory/Storage) usage."),
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
