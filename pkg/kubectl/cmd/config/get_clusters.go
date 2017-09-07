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
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/printers"
)

// GetClustersOptions contains the assignable options from the args.
type GetClustersOptions struct {
	configAccess clientcmd.ConfigAccess
	nameOnly     bool
	showHeaders  bool
	clusterNames []string
	out          io.Writer
}

var (
	get_clusters_example = templates.Examples(`
		# List the clusters kubectl knows about
		kubectl config get-clusters`)
)

// NewCmdConfigGetClusters creates a command object for the "get-clusters" action, which
// lists all clusters defined in the kubeconfig.
func NewCmdConfigGetClusters(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &GetClustersOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:     "get-clusters",
		Short:   i18n.T("Display clusters defined in the kubeconfig"),
		Long:    "Display clusters defined in the kubeconfig.",
		Example: get_clusters_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd, args, out))
			cmdutil.CheckErr(options.runGetClusters(configAccess))
		},
	}
	cmd.Flags().StringP("output", "o", "", "Output format. Only support 'name', use other output format will reset to default output format")
	cmdutil.AddNoHeadersFlags(cmd)

	return cmd
}

// complete assigns GetClustersOptions from the args.
func (o *GetClustersOptions) complete(cmd *cobra.Command, args []string, out io.Writer) error {
	o.clusterNames = args
	o.out = out
	o.nameOnly = false
	output := cmdutil.GetFlagString(cmd, "output")
	supportedOutputTypes := sets.NewString("", "name")
	if !supportedOutputTypes.Has(output) {
		fmt.Fprintf(out, "--output %v is not available in kubectl config get-contexts; resetting to default output format\n", output)
		cmd.Flags().Set("output", "")
	}
	if output == "name" {
		o.nameOnly = true
	}
	o.showHeaders = true
	if cmdutil.GetFlagBool(cmd, "no-headers") || o.nameOnly {
		o.showHeaders = false
	}

	return nil
}

func (o GetClustersOptions) runGetClusters(configAccess clientcmd.ConfigAccess) error {
	config, err := configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	out, found := o.out.(*tabwriter.Writer)
	if !found {
		out = printers.GetNewTabWriter(o.out)
		defer out.Flush()
	}

	allErrs := []error{}
	toPrint := []string{}
	if len(o.clusterNames) == 0 {
		for name := range config.Clusters {
			toPrint = append(toPrint, name)
		}
	} else {
		for _, name := range o.clusterNames {
			_, ok := config.Clusters[name]
			if ok {
				toPrint = append(toPrint, name)
			} else {
				allErrs = append(allErrs, fmt.Errorf("context %v not found", name))
			}
		}
	}
	if o.showHeaders {
		err = printClusterHeaders(out, o.nameOnly)
		if err != nil {
			allErrs = append(allErrs, err)
		}
	}

	for _, name := range toPrint {
		err = printCluster(name, config.Clusters[name], out, o.nameOnly)
		if err != nil {
			allErrs = append(allErrs, err)
		}
	}

	return utilerrors.NewAggregate(allErrs)
}

func printClusterHeaders(out io.Writer, nameOnly bool) error {
	columnNames := []string{"NAME", "SERVER"}
	if nameOnly {
		columnNames = columnNames[:1]
	}
	_, err := fmt.Fprintf(out, "%s\n", strings.Join(columnNames, "\t"))
	return err
}

func printCluster(name string, cluster *clientcmdapi.Cluster, w io.Writer, nameOnly bool) error {
	if nameOnly {
		_, err := fmt.Fprintf(w, "%s\n", name)
		return err
	}

	_, err := fmt.Fprintf(w, "%s\t%s\n", name, cluster.Server)
	return err
}
