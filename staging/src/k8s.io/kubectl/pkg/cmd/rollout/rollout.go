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

package rollout

import (
	"github.com/lithammer/dedent"
	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	rolloutLong = templates.LongDesc(i18n.T(`
		Manage the rollout of a resource.`) + rolloutValidResources)

	rolloutExample = templates.Examples(`
		# Rollback to the previous deployment
		kubectl rollout undo deployment/abc

		# Check the rollout status of a daemonset
		kubectl rollout status daemonset/foo`)

	rolloutValidResources = dedent.Dedent(`
		Valid resource types include:

		   * deployments
		   * daemonsets
		   * statefulsets
		`)
)

// NewCmdRollout returns a Command instance for 'rollout' sub command
func NewCmdRollout(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "rollout SUBCOMMAND",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Manage the rollout of a resource"),
		Long:                  rolloutLong,
		Example:               rolloutExample,
		Run:                   cmdutil.DefaultSubCommandRun(streams.Out),
	}
	// subcommands
	cmd.AddCommand(NewCmdRolloutHistory(f, streams))
	cmd.AddCommand(NewCmdRolloutPause(f, streams))
	cmd.AddCommand(NewCmdRolloutResume(f, streams))
	cmd.AddCommand(NewCmdRolloutUndo(f, streams))
	cmd.AddCommand(NewCmdRolloutStatus(f, streams))
	cmd.AddCommand(NewCmdRolloutRestart(f, streams))

	return cmd
}
