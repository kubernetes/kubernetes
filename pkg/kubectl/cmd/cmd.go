/*
Copyright 2014 Google Inc. All rights reserved.

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

	cmdconfig "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/config"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/golang/glog"

	"github.com/spf13/cobra"
)

// NewKubectlCommand creates the `kubectl` command and its nested children.
func NewKubectlCommand(f *cmdutil.Factory, in io.Reader, out, err io.Writer) *cobra.Command {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "kubectl",
		Short: "kubectl controls the Kubernetes cluster manager",
		Long: `kubectl controls the Kubernetes cluster manager.

Find more information at https://github.com/GoogleCloudPlatform/kubernetes.`,
		Run: runHelp,
	}

	f.BindFlags(cmds.PersistentFlags())

	cmds.AddCommand(NewCmdGet(f, out))
	cmds.AddCommand(NewCmdDescribe(f, out))
	cmds.AddCommand(NewCmdCreate(f, out))
	cmds.AddCommand(NewCmdUpdate(f, out))
	cmds.AddCommand(NewCmdDelete(f, out))

	cmds.AddCommand(NewCmdNamespace(out))
	cmds.AddCommand(NewCmdLog(f, out))
	cmds.AddCommand(NewCmdRollingUpdate(f, out))
	cmds.AddCommand(NewCmdResize(f, out))

	cmds.AddCommand(NewCmdExec(f, in, out, err))
	cmds.AddCommand(NewCmdPortForward(f))
	cmds.AddCommand(NewCmdProxy(f, out))

	cmds.AddCommand(NewCmdRunContainer(f, out))
	cmds.AddCommand(NewCmdStop(f, out))
	cmds.AddCommand(NewCmdExposeService(f, out))

	cmds.AddCommand(NewCmdLabel(f, out))

	cmds.AddCommand(cmdconfig.NewCmdConfig(f, out))
	cmds.AddCommand(NewCmdClusterInfo(f, out))
	cmds.AddCommand(NewCmdApiVersions(f, out))
	cmds.AddCommand(NewCmdVersion(f, out))

	return cmds
}

func runHelp(cmd *cobra.Command, args []string) {
	cmd.Help()
}

func printDeprecationWarning(command, alias string) {
	glog.Warningf("%s is DEPRECATED and will be removed in a future version. Use %s instead.", alias, command)
}
