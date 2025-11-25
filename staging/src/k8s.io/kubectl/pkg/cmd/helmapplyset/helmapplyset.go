/*
Copyright 2024 The Kubernetes Authors.

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

package helmapplyset

import (
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"

	"github.com/spf13/cobra"
)

var (
	applysetLong = templates.LongDesc(i18n.T(`
		Manage Helm releases as ApplySets.

		ApplySets provide a way to track and manage groups of Kubernetes resources together.
		This command allows you to view and manage Helm releases that have been integrated
		with the ApplySet specification.`))

	applysetExample = templates.Examples(i18n.T(`
		# List all ApplySets
		kubectl get applysets

		# List ApplySets in a specific namespace
		kubectl get applysets -n default

		# List ApplySets across all namespaces
		kubectl get applysets --all-namespaces

		# Filter ApplySets by tooling
		kubectl get applysets -l applyset.kubernetes.io/tooling=helm/v3

		# Describe a specific ApplySet
		kubectl describe applyset my-release`))
)

// NewCmdHelmApplySet creates the 'applyset' command
func NewCmdHelmApplySet(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "applyset",
		Aliases: []string{"applysets"},
		Short:   i18n.T("Manage Helm releases as ApplySets"),
		Long:    applysetLong,
		Example: applysetExample,
	}

	// Add subcommands
	cmd.AddCommand(NewCmdGetApplySets(f, streams))
	cmd.AddCommand(NewCmdDescribeApplySet(f, streams))
	cmd.AddCommand(NewCmdStatus(f, streams))

	return cmd
}
