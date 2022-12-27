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

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	getClustersExample = templates.Examples(`
		# List the clusters that kubectl knows about
		kubectl config get-clusters`)
)

// GetClusterOptions holds the data needed to run the command
type GetClusterOptions struct {
	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

// NewCmdConfigGetClusters creates a command object for the "get-clusters" action, which
// lists all clusters defined in the kubeconfig.
func NewCmdConfigGetClusters(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "get-clusters",
		Short:   i18n.T("Display clusters defined in the kubeconfig"),
		Long:    i18n.T("Display clusters defined in the kubeconfig."),
		Example: getClustersExample,
		Run: func(cmd *cobra.Command, args []string) {
			options := NewGetClusterOptions(streams, configAccess)
			cmdutil.CheckErr(options.RunGetClusters())
		},
	}

	return cmd
}

// NewGetClusterOptions creates the options for the command
func NewGetClusterOptions(ioStreams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *GetClusterOptions {
	return &GetClusterOptions{
		ConfigAccess: configAccess,
		IOStreams:    ioStreams,
	}
}

func (o *GetClusterOptions) RunGetClusters() error {
	config, _, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	if _, nil := fmt.Fprintf(o.IOStreams.Out, "NAME\n"); err != nil {
		return err
	}
	for name := range config.Clusters {
		if _, nil := fmt.Fprintf(o.IOStreams.Out, "%s\n", name); err != nil {
			return err
		}
	}

	return nil
}
