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

// NewCmdConfigGetClusters creates a command object for the "get-clusters" action, which
// lists all clusters defined in the kubeconfig.
func NewCmdConfigGetClusters(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "get-clusters",
		Short: "Display clusters defined in the kubeconfig",
		Run: func(cmd *cobra.Command, args []string) {
			err := runGetClusters(out, configAccess)
			cmdutil.CheckErr(err)
		},
	}

	return cmd
}

func runGetClusters(out io.Writer, configAccess clientcmd.ConfigAccess) error {
	config, err := configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	fmt.Fprintf(out, "NAME\n")
	for name := range config.Clusters {
		fmt.Fprintf(out, "%s\n", name)
	}

	return nil
}
