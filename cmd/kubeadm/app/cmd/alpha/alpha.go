/*
Copyright 2018 The Kubernetes Authors.

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

package alpha

import (
	"io"

	"github.com/spf13/cobra"
)

// NewCmdAlpha returns "kubeadm alpha" command.
func NewCmdAlpha(in io.Reader, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "alpha",
		Short: "Kubeadm experimental sub-commands",
	}

	kubeconfigCmd := NewCmdKubeConfigUtility(out)
	deprecateCommand(`please use the same command under "kubeadm kubeconfig"`, kubeconfigCmd)
	cmd.AddCommand(kubeconfigCmd)

	return cmd
}

func deprecateCommand(msg string, cmds ...*cobra.Command) {
	for _, cmd := range cmds {
		cmd.Deprecated = msg
		childCmds := cmd.Commands()
		if len(childCmds) > 0 {
			deprecateCommand(msg, childCmds...)
		}
	}
}
