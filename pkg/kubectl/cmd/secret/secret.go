/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package secret

import (
	"io"

	"github.com/spf13/cobra"

	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

const (
	secretsLong = `
Store secret artifacts.

Secrets are used to store confidential information that should not be contained inside of an image.
They are commonly used to hold things like keys for authentication to other internal systems.`
)

func NewCmdSecret(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "secret SUBCOMMAND",
		Short: "Store secret artifacts.",
		Long:  secretsLong,
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	cmds.AddCommand(NewCmdMakeSecret(f, out))
	cmds.AddCommand(NewCmdMakeDockercfgSecret(f, out))

	return cmds
}
