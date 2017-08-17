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
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	rollout_long = templates.LongDesc(`
		Manage the rollout of a resource.` + rollout_valid_resources)

	rollout_example = templates.Examples(`
		# Rollback to the previous deployment
		kubectl rollout undo deployment/abc
		
		# Check the rollout status of a daemonset
		kubectl rollout status daemonset/foo`)

	rollout_valid_resources = dedent.Dedent(`
		Valid resource types include:

		   * deployments
		   * daemonsets
		`)
)

func NewCmdRollout(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {

	cmd := &cobra.Command{
		Use:     "rollout SUBCOMMAND",
		Short:   i18n.T("Manage the rollout of a resource"),
		Long:    rollout_long,
		Example: rollout_example,
		Run:     cmdutil.DefaultSubCommandRun(errOut),
	}
	// subcommands
	cmd.AddCommand(NewCmdRolloutHistory(f, out))
	cmd.AddCommand(NewCmdRolloutPause(f, out))
	cmd.AddCommand(NewCmdRolloutResume(f, out))
	cmd.AddCommand(NewCmdRolloutUndo(f, out))
	cmd.AddCommand(NewCmdRolloutStatus(f, out))

	return cmd
}
