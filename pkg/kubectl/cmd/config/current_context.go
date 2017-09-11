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
	"io"

	"github.com/spf13/cobra"

	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

type CurrentContextOptions struct {
	ConfigAccess clientcmd.ConfigAccess
}

var (
	current_context_long = templates.LongDesc(`
		Displays the current-context`)

	current_context_example = templates.Examples(`
		# Display the current-context
		kubectl config current-context`)
)

func NewCmdConfigCurrentContext(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &CurrentContextOptions{ConfigAccess: configAccess}

	cmd := &cobra.Command{
		Use:     "current-context",
		Short:   i18n.T("Displays the current-context"),
		Long:    current_context_long,
		Example: current_context_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunCurrentContext(out, options)
			cmdutil.CheckErr(err)
		},
	}

	return cmd
}

func RunCurrentContext(out io.Writer, options *CurrentContextOptions) error {
	config, err := options.ConfigAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	if config.CurrentContext == "" {
		err = fmt.Errorf("current-context is not set\n")
		return err
	}

	fmt.Fprintf(out, "%s\n", config.CurrentContext)
	return nil
}
