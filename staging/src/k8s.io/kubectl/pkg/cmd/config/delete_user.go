/*
Copyright 2020 The Kubernetes Authors.

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

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	deleteUserExample = templates.Examples(`
		# Delete the minikube user
		kubectl config delete-user minikube`)
)

// DeleteUserOptions holds the data needed to run the command
type DeleteUserOptions struct {
	User string

	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

// NewCmdConfigDeleteUser returns a Command instance for 'config delete-user' sub command
func NewCmdConfigDeleteUser(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "delete-user NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Delete the specified user from the kubeconfig"),
		Long:                  i18n.T("Delete the specified user from the kubeconfig."),
		Example:               deleteUserExample,
		ValidArgsFunction:     completion.UserCompletionFunc,
		Run: func(cmd *cobra.Command, args []string) {
			o := NewDeleteUserOptions(streams, configAccess)
			cmdutil.CheckErr(o.Complete(args))
			cmdutil.CheckErr(o.RunDeleteUser())
		},
	}

	return cmd
}

// NewDeleteUserOptions creates the options for the command
func NewDeleteUserOptions(ioStreams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *DeleteUserOptions {
	return &DeleteUserOptions{
		ConfigAccess: configAccess,
		IOStreams:    ioStreams,
	}
}

// Complete sets up the command to run
func (o *DeleteUserOptions) Complete(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("unexpected args: %v", args)
	}
	o.User = args[0]
	return nil
}

// RunDeleteUser performs the command
func (o *DeleteUserOptions) RunDeleteUser() error {
	config, configFile, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	if _, ok := config.AuthInfos[o.User]; !ok {
		return fmt.Errorf("user \"%s\" does not exist in config file %s", o.User, configFile)
	}
	delete(config.AuthInfos, o.User)

	if err := clientcmd.ModifyConfig(o.ConfigAccess, *config, true); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(o.IOStreams.Out, "deleted user \"%s\" from %s\n", o.User, configFile); err != nil {
		return err
	}

	return nil
}
