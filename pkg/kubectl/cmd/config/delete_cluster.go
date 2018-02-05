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

	"github.com/spf13/cobra"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	deleteClusterExample = templates.Examples(`
		# Delete the minikube cluster
		kubectl config delete-cluster minikube`)
)

func NewCmdConfigDeleteCluster(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use: "delete-cluster NAME",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Delete the specified cluster from the kubeconfig"),
		Long:    "Delete the specified cluster from the kubeconfig",
		Example: deleteClusterExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(runDeleteCluster(out, configAccess, args))
		},
	}

	return cmd
}

func runDeleteCluster(out io.Writer, configAccess clientcmd.ConfigAccess, args []string) error {
	config, err := configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	configFile := configAccess.GetDefaultFilename()
	if configAccess.IsExplicitFile() {
		configFile = configAccess.GetExplicitFile()
	}
	for _, arg := range args {
		_, ok := config.Clusters[arg]
		if !ok {
			return fmt.Errorf("cannot delete cluster %s, not in %s", arg, configFile)
		}

		delete(config.Clusters, arg)

		if err := clientcmd.ModifyConfig(configAccess, *config, true); err != nil {
			return err
		}

		_, err := fmt.Fprintf(out, "deleted cluster %s from %s\n", arg, configFile)
		if err != nil {
			return err
		}
	}
	return nil
}
