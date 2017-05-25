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

package config

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientcmd "k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

// ImportConfigsOptions contains the assignable options from the args.
type ImportConfigsOptions struct {
	configAccess clientcmd.ConfigAccess
	source       string
	target       string
}

var (
	importConfigsLong = templates.LongDesc(`
		Imports kubecontext file content

		Adds all cluster, user and context entries of a source kubeconfig
		into another kubeconfig target file. If target is not provided,
		the kubeconfig used in path is considered. In case there is a
		name conflict, i.e. an entry with a given name exists in both
		files, this entry on the target file is preserved.`)

	importConfigsExample = templates.Examples(`
		# Imports a kubeconfig file using target as default kubeconfig
		kubectl config import sourceconfigfile

		# Providing a target file
		kubectl config import sourceconfigfile --target otherfile`)
)

// NewCmdConfigImportConfigs creates a command object for the "import" action, which
// merges two kubeconfig files
func NewCmdConfigImportConfigs(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &ImportConfigsOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:     "import source [(-t|--target)=~/.kube/config]",
		Short:   i18n.T("Imports kubeconfig file content"),
		Long:    importConfigsLong,
		Example: importConfigsExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(cmd, args, out))
			cmdutil.CheckErr(options.RunImportConfigs(out))
		},
	}
	cmd.Flags().String("target", "", "the file that will receive the entries. Defaults to the regular kubeconfig file")
	return cmd
}

// Complete assigns ImportConfigsOptions from the args.
func (o *ImportConfigsOptions) Complete(cmd *cobra.Command, args []string, out io.Writer) error {

	o.target = ""
	if cmdutil.GetFlagString(cmd, "target") != "" {
		o.target = cmdutil.GetFlagString(cmd, "target")
	}

	if len(args) != 1 {
		cmd.Help()
		return fmt.Errorf("Unexpected args: %v", args)
	}
	o.source = args[0]

	return nil
}

// RunImportConfigs implements all the necessary functionality for merging two kubeconfig files.
func (o ImportConfigsOptions) RunImportConfigs(out io.Writer) error {
	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}
	source := clientcmd.GetConfigFromFileOrDie(o.source)
	if isEmptyConfig(source) {
		return fmt.Errorf("provided source file does not exist or is empty")
	}
	if o.target != "" {
		target := clientcmd.GetConfigFromFileOrDie(o.target)
		if isEmptyConfig(target) {
			return fmt.Errorf("provided target file does not exist or is empty")
		}
		fmt.Fprintf(out, "Using provided target file instead of default kubeconfig.\n")
		config = target
	}

	if len(source.Clusters) == 0 {
		fmt.Fprintf(out, "There is no cluster entry in the source file.\n")
	} else {
		for k := range source.Clusters {
			if config.Clusters[k] == nil {
				config.Clusters[k] = source.Clusters[k]
				fmt.Fprintf(out, "Copied cluster entry %q to target file.\n", k)
			} else {
				fmt.Fprintf(out, "Cluster called %q exists in both files, so it will not be copied.\n", k)
			}
		}
	}

	if len(source.Contexts) == 0 {
		fmt.Fprintf(out, "There is no context entry in the source file.\n")
	} else {
		for k := range source.Contexts {
			if config.Contexts[k] == nil {
				config.Contexts[k] = source.Contexts[k]
				fmt.Fprintf(out, "Copied context entry %q to target file.\n", k)
			} else {
				fmt.Fprintf(out, "Context called %q exists in both files, so it will not be copied.\n", k)
			}
		}
	}

	if len(source.AuthInfos) == 0 {
		fmt.Fprintf(out, "There is no user entry in the source file.\n")
	} else {
		for k := range source.AuthInfos {
			if config.AuthInfos[k] == nil {
				config.AuthInfos[k] = source.AuthInfos[k]
				fmt.Fprintf(out, "Copied User entry %q to target file.\n", k)
			} else {
				fmt.Fprintf(out, "User called %q exists in both files, so it will not be copied.\n", k)
			}
		}
	}

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	allErrs := []error{}
	return utilerrors.NewAggregate(allErrs)
}

func isEmptyConfig(config *clientcmdapi.Config) bool {
	return (len(config.Clusters) == 0) && len(config.Contexts) == 0 && len(config.AuthInfos) == 0
}
