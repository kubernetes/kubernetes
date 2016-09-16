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
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func NewCmdConfigDeleteCluster(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "delete-cluster NAME",
		Short: "Delete the specified cluster from the kubeconfig",
		Run: func(cmd *cobra.Command, args []string) {
			err := runDeleteCluster(out, configAccess, cmd)
			cmdutil.CheckErr(err)
		},
	}

	return cmd
}

func runDeleteCluster(out io.Writer, configAccess clientcmd.ConfigAccess, cmd *cobra.Command) error {
	config, err := configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	args := cmd.Flags().Args()
	if len(args) != 1 {
		cmd.Help()
		return nil
	}

	configFile := configAccess.GetDefaultFilename()
	if configAccess.IsExplicitFile() {
		configFile = configAccess.GetExplicitFile()
	}

	name := args[0]
	_, ok := config.Clusters[name]
	if !ok {
		return fmt.Errorf("cannot delete cluster %s, not in %s", name, configFile)
	}

	delete(config.Clusters, name)

	if err := clientcmd.ModifyConfig(configAccess, *config, true); err != nil {
		return err
	}

	fmt.Fprintf(out, "deleted cluster %s from %s", name, configFile)

	return nil
}
