/*
Copyright 2014 The Kubernetes Authors.

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

package auth

import (
	"github.com/spf13/cobra"
	"k8s.io/cli-runtime/pkg/genericclioptions"

	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// NewCmdAuth returns an initialized Command instance for 'auth' sub command
func NewCmdAuth(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "auth",
		Short: "Inspect authorization",
		Long:  `Inspect authorization`,
		Run:   cmdutil.DefaultSubCommandRun(streams.ErrOut),
	}

	cmds.AddCommand(NewCmdCanI(f, streams))
	cmds.AddCommand(NewCmdReconcile(f, streams))

	return cmds
}
