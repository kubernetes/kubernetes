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

package cmd

import (
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	join_long = dedent.Dedent(`
		Join an entity to a group.

		See subcommands for details`)
)

// NewCmdJoin defines the `join` command that joins an entity to a group.
func NewCmdJoin(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "join",
		Short: "Join an entity to a group",
		Long:  join_long,
		Run: func(cmd *cobra.Command, args []string) {
			err := cmdutil.UsageError(cmd, "Subcommand expected: %v", args)
			cmdutil.CheckErr(err)
		},
	}

	cmd.AddCommand(NewCmdJoinFederation(f, cmdOut, NewJoinFederationConfig(clientcmd.NewDefaultPathOptions())))
	return cmd
}
