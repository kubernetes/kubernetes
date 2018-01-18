/*
Copyright 2017 The Kubernetes Authors.

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

package phases

import (
	"io"

	"github.com/spf13/cobra"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
)

// NewCmdPhase returns the cobra command for the "kubeadm phase" command (currently alpha-gated)
func NewCmdPhase(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "phase",
		Short: "Invoke subsets of kubeadm functions separately for a manual install.",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(NewCmdAddon())
	cmd.AddCommand(NewCmdBootstrapToken())
	cmd.AddCommand(NewCmdCerts())
	cmd.AddCommand(NewCmdControlplane())
	cmd.AddCommand(NewCmdEtcd())
	cmd.AddCommand(NewCmdKubeConfig(out))
	cmd.AddCommand(NewCmdMarkMaster())
	cmd.AddCommand(NewCmdPreFlight())
	cmd.AddCommand(NewCmdSelfhosting())
	cmd.AddCommand(NewCmdUploadConfig())

	return cmd
}
