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

	cmd.AddCommand(newCmdKubeConfigUtility(out))

	const shDeprecatedMessage = "self-hosting support in kubeadm is deprecated " +
		"and will be removed in a future release"
	shCommand := newCmdSelfhosting(in)
	shCommand.Deprecated = shDeprecatedMessage
	for _, cmd := range shCommand.Commands() {
		cmd.Deprecated = shDeprecatedMessage
	}
	cmd.AddCommand(shCommand)

	certsCommand := NewCmdCertsUtility(out)
	deprecateCertsCommand(certsCommand)
	cmd.AddCommand(certsCommand)

	return cmd
}

func deprecateCertsCommand(cmds ...*cobra.Command) {
	const deprecatedMessage = "please use the same command under \"kubeadm certs\""

	for _, cmd := range cmds {
		cmd.Deprecated = deprecatedMessage
		childCmds := cmd.Commands()
		if len(childCmds) > 0 {
			deprecateCertsCommand(childCmds...)
		}
	}
}
