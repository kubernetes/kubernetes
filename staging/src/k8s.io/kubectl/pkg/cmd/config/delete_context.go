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

package config

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"
	"k8s.io/client-go/tools/clientcmd"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	deleteContextExample = templates.Examples(`
		# Delete the context for the minikube cluster
		kubectl config delete-context minikube`)
)

// NewCmdConfigDeleteContext returns a Command instance for 'config delete-context' sub command
func NewCmdConfigDeleteContext(out, errOut io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "delete-context NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Delete the specified context from the kubeconfig"),
		Long:                  i18n.T("Delete the specified context from the kubeconfig."),
		Example:               deleteContextExample,
		ValidArgsFunction:     completion.ContextCompletionFunc,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(runDeleteContext(out, errOut, configAccess, cmd))
		},
	}

	return cmd
}

func runDeleteContext(out, errOut io.Writer, configAccess clientcmd.ConfigAccess, cmd *cobra.Command) error {
	config, err := configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	args := cmd.Flags().Args()
	if len(args) != 1 {
		cmd.Help()
		return nil
	}

	configFile := configAccess.GetDefaultFilename()
	if configAccess.IsExplicitFile() {
		configFile = configAccess.GetExplicitFile()
	}

	name := args[0]
	_, ok := config.Contexts[name]
	if !ok {
		return fmt.Errorf("cannot delete context %s, not in %s", name, configFile)
	}

	if config.CurrentContext == name {
		fmt.Fprint(errOut, "warning: this removed your active context, use \"kubectl config use-context\" to select a different one\n")
	}

	delete(config.Contexts, name)

	if err := clientcmd.ModifyConfig(configAccess, *config, true); err != nil {
		return err
	}

	fmt.Fprintf(out, "deleted context %s from %s\n", name, configFile)

	return nil
}
