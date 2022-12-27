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
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	deleteClusterExample = templates.Examples(`
		# Delete the minikube cluster
		kubectl config delete-cluster minikube`)
)

// DeleteClusterOptions holds the data needed to run the command
type DeleteClusterOptions struct {
	Cluster string

	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

// NewCmdConfigDeleteCluster returns a Command instance for 'config delete-cluster' sub command
func NewCmdConfigDeleteCluster(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "delete-cluster NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Delete the specified cluster from the kubeconfig"),
		Long:                  i18n.T("Delete the specified cluster from the kubeconfig."),
		Example:               deleteClusterExample,
		ValidArgsFunction:     completion.ClusterCompletionFunc,
		Run: func(cmd *cobra.Command, args []string) {
			options := NewDeleteClusterOptions(streams, configAccess)
			cmdutil.CheckErr(options.Complete(args))
			cmdutil.CheckErr(options.RunDeleteCluster())
		},
	}

	return cmd
}

// NewDeleteClusterOptions creates the options for the command
func NewDeleteClusterOptions(ioStreams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *DeleteClusterOptions {
	return &DeleteClusterOptions{
		ConfigAccess: configAccess,
		IOStreams:    ioStreams,
	}
}

func (o *DeleteClusterOptions) Complete(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("unexpected args: %v", args)
	}
	o.Cluster = args[0]
	return nil
}

func (o *DeleteClusterOptions) RunDeleteCluster() error {
	config, configFile, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	if _, ok := config.Clusters[o.Cluster]; !ok {
		return fmt.Errorf("cannot delete cluster \"%s\", not in file %s", o.Cluster, configFile)
	}

	delete(config.Clusters, o.Cluster)

	if err := clientcmd.ModifyConfig(o.ConfigAccess, *config, true); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(o.IOStreams.Out, "deleted cluster \"%s\" from %s\n", o.Cluster, configFile); err != nil {
		return err
	}

	return nil
}
