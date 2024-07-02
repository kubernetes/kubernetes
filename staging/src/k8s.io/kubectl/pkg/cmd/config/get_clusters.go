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

	"github.com/liggitt/tabwriter"
	"github.com/spf13/cobra"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type GetClustersOptions struct {
	configAccess clientcmd.ConfigAccess
	noHeaders bool
	showHeaders bool

	genericiooptions.IOStreams
}

var (
	getClustersLong = templates.LongDesc(i18n.T(`Display one or many clusters from the kubeconfig file.`))

	getClustersExample = templates.Examples(`
		# List the clusters that kubectl knows about
		kubectl config get-clusters`)
)

// NewCmdConfigGetClusters creates a command object for the "get-clusters" action, which
// lists all clusters defined in the kubeconfig.
func NewCmdConfigGetClusters(streams genericiooptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &GetClustersOptions{
		configAccess: configAccess,
		IOStreams: streams,
	}

	cmd := &cobra.Command{
		Use:     "get-clusters",
		Short:   i18n.T("Display clusters defined in the kubeconfig"),
		Long:    getClustersLong,
		Example: getClustersExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(cmd))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.RunGetClusters())
		},
	}

	cmd.Flags().BoolVar(&options.noHeaders, "no-headers", options.noHeaders, "When using the default output format, don't print headers (default print headers).")

	return cmd
}

// Complete assigns GetClustersOptions from the args
func (o *GetClustersOptions) Complete(cmd *cobra.Command) error {
	o.showHeaders = true
	if cmdutil.GetFlagBool(cmd, "no-headers") {
		o.showHeaders = false
	}

	return nil
}

func (o *GetClustersOptions) Validate() error {
	return nil
}

// RunGetClusters implements all the necessary functionality for cluster retrieval.
func (o *GetClustersOptions) RunGetClusters() error {
	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	out, found := o.Out.(*tabwriter.Writer)
	if !found {
		out = printers.GetNewTabWriter(o.Out)
		defer out.Flush()
	}

	if o.showHeaders {
		fmt.Fprintf(out, "NAME\n")
	}

	for name := range config.Clusters {
		fmt.Fprintf(out, "%s\n", name)
	}

	return nil
}
