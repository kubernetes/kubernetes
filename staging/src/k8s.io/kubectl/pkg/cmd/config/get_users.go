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
	"sort"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	getUsersExample = templates.Examples(`
		# List the users kubectl knows about
		kubectl config get-users`)
)

// GetUsersOptions holds the data needed to run the command
type GetUsersOptions struct {
	configAccess clientcmd.ConfigAccess

	genericclioptions.IOStreams
}

// NewGetUsersOptions creates the options for the command
func NewGetUsersOptions(ioStreams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *GetUsersOptions {
	return &GetUsersOptions{
		configAccess: configAccess,
		IOStreams:    ioStreams,
	}
}

// NewCmdConfigGetUsers creates a command object for the "get-users" action, which
// lists all users defined in the kubeconfig.
func NewCmdConfigGetUsers(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	o := NewGetUsersOptions(streams, configAccess)

	cmd := &cobra.Command{
		Use:     "get-users",
		Short:   i18n.T("Display users defined in the kubeconfig"),
		Long:    i18n.T("Display users defined in the kubeconfig."),
		Example: getUsersExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Run())
		},
	}

	return cmd
}

// Run performs the command
func (o *GetUsersOptions) Run() error {
	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	users := make([]string, 0, len(config.AuthInfos))
	for user := range config.AuthInfos {
		users = append(users, user)
	}
	sort.Strings(users)

	fmt.Fprintf(o.Out, "NAME\n")
	for _, user := range users {
		fmt.Fprintf(o.Out, "%s\n", user)
	}

	return nil
}
