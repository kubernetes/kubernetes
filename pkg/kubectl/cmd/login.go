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

package cmd

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	restclient "k8s.io/client-go/rest"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

// NewCmdLogin creates a command that allows interactive login
func NewCmdLogin(f cmdutil.Factory, in io.Reader, out, err io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "login",
		Short: i18n.T("Interactive login"),
		Long:  i18n.T("Interactive login from command line"),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(runLogin(f))
		},
	}
	return cmd
}

func runLogin(f cmdutil.Factory) error {
	config, err := f.ClientConfig()
	if err != nil {
		return err
	}
	if config.AuthProvider == nil {
		return fmt.Errorf("no auth provider specified for current user")
	}
	provider, err := restclient.GetAuthProvider(
		config.Host, config.AuthProvider, config.AuthConfigPersister)
	if err != nil {
		return err
	}
	return provider.Login()
}
