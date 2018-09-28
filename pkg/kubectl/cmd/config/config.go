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

package config

import (
	"fmt"
	"path"
	"strconv"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

// NewCmdConfig creates a command object for the "config" action, and adds all child commands to it.
func NewCmdConfig(f cmdutil.Factory, pathOptions *clientcmd.PathOptions, streams genericclioptions.IOStreams) *cobra.Command {
	if len(pathOptions.ExplicitFileFlag) == 0 {
		pathOptions.ExplicitFileFlag = clientcmd.RecommendedConfigPathFlag
	}

	cmd := &cobra.Command{
		Use: "config SUBCOMMAND",
		DisableFlagsInUseLine: true,
		Short: i18n.T("Modify kubeconfig files"),
		Long: templates.LongDesc(`
			Modify kubeconfig files using subcommands like "kubectl config set current-context my-context"

			The loading order follows these rules:

			1. If the --` + pathOptions.ExplicitFileFlag + ` flag is set, then only that file is loaded. The flag may only be set once and no merging takes place.
			2. If $` + pathOptions.EnvVar + ` environment variable is set, then it is used as a list of paths (normal path delimitting rules for your system). These paths are merged. When a value is modified, it is modified in the file that defines the stanza. When a value is created, it is created in the first file that exists. If no files in the chain exist, then it creates the last file in the list.
			3. Otherwise, ` + path.Join("${HOME}", pathOptions.GlobalFileSubpath) + ` is used and no merging takes place.`),
		Run: cmdutil.DefaultSubCommandRun(streams.ErrOut),
	}

	// file paths are common to all sub commands
	cmd.PersistentFlags().StringVar(&pathOptions.LoadingRules.ExplicitPath, pathOptions.ExplicitFileFlag, pathOptions.LoadingRules.ExplicitPath, "use a particular kubeconfig file")

	// TODO(juanvallejo): update all subcommands to work with genericclioptions.IOStreams
	cmd.AddCommand(NewCmdConfigView(f, streams, pathOptions))
	cmd.AddCommand(NewCmdConfigSetCluster(streams.Out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetAuthInfo(streams.Out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetContext(streams.Out, pathOptions))
	cmd.AddCommand(NewCmdConfigSet(streams.Out, pathOptions))
	cmd.AddCommand(NewCmdConfigUnset(streams.Out, pathOptions))
	cmd.AddCommand(NewCmdConfigCurrentContext(streams.Out, pathOptions))
	cmd.AddCommand(NewCmdConfigUseContext(streams.Out, pathOptions))
	cmd.AddCommand(NewCmdConfigGetContexts(streams, pathOptions))
	cmd.AddCommand(NewCmdConfigGetClusters(streams.Out, pathOptions))
	cmd.AddCommand(NewCmdConfigDeleteCluster(streams.Out, pathOptions))
	cmd.AddCommand(NewCmdConfigDeleteContext(streams.Out, streams.ErrOut, pathOptions))
	cmd.AddCommand(NewCmdConfigRenameContext(streams.Out, pathOptions))

	return cmd
}

func toBool(propertyValue string) (bool, error) {
	boolValue := false
	if len(propertyValue) != 0 {
		var err error
		boolValue, err = strconv.ParseBool(propertyValue)
		if err != nil {
			return false, err
		}
	}

	return boolValue, nil
}

func helpErrorf(cmd *cobra.Command, format string, args ...interface{}) error {
	cmd.Help()
	msg := fmt.Sprintf(format, args...)
	return fmt.Errorf("%s\n", msg)
}
