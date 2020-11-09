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
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
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
	user string

	configAccess clientcmd.ConfigAccess
	config       *clientcmdapi.Config
	configFile   string

	genericclioptions.IOStreams
}

// NewDeleteUserOptions creates the options for the command
func NewDeleteUserOptions(ioStreams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *DeleteUserOptions {
	return &DeleteUserOptions{
		configAccess: configAccess,
		IOStreams:    ioStreams,
	}
}

// NewCmdConfigDeleteUser returns a Command instance for 'config delete-user' sub command
func NewCmdConfigDeleteUser(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	o := NewDeleteUserOptions(streams, configAccess)

	cmd := &cobra.Command{
		Use:                   "delete-user NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Delete the specified user from the kubeconfig"),
		Long:                  i18n.T("Delete the specified user from the kubeconfig"),
		Example:               deleteUserExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	return cmd
}

// Complete sets up the command to run
func (o *DeleteUserOptions) Complete(cmd *cobra.Command, args []string) error {
	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}
	o.config = config

	if len(args) != 1 {
		return cmdutil.UsageErrorf(cmd, "user to delete is required")
	}
	o.user = args[0]

	configFile := o.configAccess.GetDefaultFilename()
	if o.configAccess.IsExplicitFile() {
		configFile = o.configAccess.GetExplicitFile()
	}
	o.configFile = configFile

	return nil
}

// Validate ensures the command has enough info to run
func (o *DeleteUserOptions) Validate() error {
	_, ok := o.config.AuthInfos[o.user]
	if !ok {
		return fmt.Errorf("cannot delete user %s, not in %s", o.user, o.configFile)
	}

	return nil
}

// Run performs the command
func (o *DeleteUserOptions) Run() error {
	delete(o.config.AuthInfos, o.user)

	if err := clientcmd.ModifyConfig(o.configAccess, *o.config, true); err != nil {
		return err
	}

	fmt.Fprintf(o.Out, "deleted user %s from %s\n", o.user, o.configFile)

	return nil
}
