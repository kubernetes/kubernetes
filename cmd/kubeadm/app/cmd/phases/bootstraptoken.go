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

package phases

import (
	"fmt"

	"github.com/spf13/cobra"

	clientset "k8s.io/client-go/kubernetes"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	versionutil "k8s.io/kubernetes/pkg/util/version"
)

// NewCmdBootstrapToken returns the Cobra command for running the mark-master phase
func NewCmdBootstrapToken() *cobra.Command {
	var kubeConfigFile string
	cmd := &cobra.Command{
		Use:     "bootstrap-token",
		Short:   "Manage kubeadm-specific Bootstrap Token functions.",
		Aliases: []string{"bootstraptoken"},
		RunE:    cmdutil.SubCmdRunE("bootstrap-token"),
	}

	cmd.PersistentFlags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")

	// Add subcommands
	cmd.AddCommand(NewSubCmdClusterInfo(&kubeConfigFile))
	cmd.AddCommand(NewSubCmdNodeBootstrapToken(&kubeConfigFile))

	return cmd
}

// NewSubCmdClusterInfo returns the Cobra command for running the cluster-info sub-phase
func NewSubCmdClusterInfo(kubeConfigFile *string) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "cluster-info <clusterinfo-file>",
		Short:   "Uploads and exposes the cluster-info ConfigMap publicly from the given cluster-info file",
		Aliases: []string{"clusterinfo"},
		Run: func(cmd *cobra.Command, args []string) {
			err := cmdutil.ValidateExactArgNumber(args, []string{"clusterinfo-file"})
			kubeadmutil.CheckErr(err)

			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			// Here it's safe to get args[0], since we've validated that the argument exists above in validateExactArgNumber
			clusterInfoFile := args[0]
			// Create the cluster-info ConfigMap or update if it already exists
			err = clusterinfo.CreateBootstrapConfigMapIfNotExists(client, clusterInfoFile)
			kubeadmutil.CheckErr(err)

			// Create the RBAC rules that expose the cluster-info ConfigMap properly
			err = clusterinfo.CreateClusterInfoRBACRules(client)
			kubeadmutil.CheckErr(err)
		},
	}
	return cmd
}

// NewSubCmdNodeBootstrapToken returns the Cobra command for running the node sub-phase
func NewSubCmdNodeBootstrapToken(kubeConfigFile *string) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "node",
		Short:   "Manages Node Bootstrap Tokens",
		Aliases: []string{"clusterinfo"},
		RunE:    cmdutil.SubCmdRunE("node"),
	}

	cmd.AddCommand(NewSubCmdNodeBootstrapTokenPostCSRs(kubeConfigFile))
	cmd.AddCommand(NewSubCmdNodeBootstrapTokenAutoApprove(kubeConfigFile))

	return cmd
}

// NewSubCmdNodeBootstrapTokenPostCSRs returns the Cobra command for running the allow-post-csrs sub-phase
func NewSubCmdNodeBootstrapTokenPostCSRs(kubeConfigFile *string) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "allow-post-csrs",
		Short: "Configure RBAC to allow Node Bootstrap tokens to post CSRs in order for nodes to get long term certificate credentials",
		Run: func(cmd *cobra.Command, args []string) {
			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			clusterVersion, err := getClusterVersion(client)
			kubeadmutil.CheckErr(err)

			err = node.AllowBootstrapTokensToPostCSRs(client, clusterVersion)
			kubeadmutil.CheckErr(err)
		},
	}
	return cmd
}

// NewSubCmdNodeBootstrapTokenAutoApprove returns the Cobra command for running the allow-auto-approve sub-phase
func NewSubCmdNodeBootstrapTokenAutoApprove(kubeConfigFile *string) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "allow-auto-approve",
		Short: "Configure RBAC rules to allow the csrapprover controller automatically approve CSRs from a Node Bootstrap Token",
		Run: func(cmd *cobra.Command, args []string) {
			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			clusterVersion, err := getClusterVersion(client)
			kubeadmutil.CheckErr(err)

			err = node.AutoApproveNodeBootstrapTokens(client, clusterVersion)
			kubeadmutil.CheckErr(err)
		},
	}
	return cmd
}

// getClusterVersion fetches the API server version and parses it
func getClusterVersion(client clientset.Interface) (*versionutil.Version, error) {
	clusterVersionInfo, err := client.Discovery().ServerVersion()
	if err != nil {
		return nil, fmt.Errorf("failed to check server version: %v", err)
	}
	clusterVersion, err := versionutil.ParseSemantic(clusterVersionInfo.String())
	if err != nil {
		return nil, fmt.Errorf("failed to parse server version: %v", err)
	}
	return clusterVersion, nil
}
