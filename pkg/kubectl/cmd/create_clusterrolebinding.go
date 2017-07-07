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

package cmd

import (
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	clusterRoleBindingLong = templates.LongDesc(i18n.T(`
		Create a ClusterRoleBinding for a particular ClusterRole.`))

	clusterRoleBindingExample = templates.Examples(i18n.T(`
		  # Create a ClusterRoleBinding for user1, user2, and group1 using the cluster-admin ClusterRole
		  kubectl create clusterrolebinding cluster-admin --clusterrole=cluster-admin --user=user1 --user=user2 --group=group1`))
)

// ClusterRoleBinding is a command to ease creating ClusterRoleBindings.
func NewCmdCreateClusterRoleBinding(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "clusterrolebinding NAME --clusterrole=NAME [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run]",
		Short:   i18n.T("Create a ClusterRoleBinding for a particular ClusterRole"),
		Long:    clusterRoleBindingLong,
		Example: clusterRoleBindingExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateClusterRoleBinding(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ClusterRoleBindingV1GeneratorName)
	cmd.Flags().String("clusterrole", "", i18n.T("ClusterRole this ClusterRoleBinding should reference"))
	cmd.Flags().StringArray("user", []string{}, "Usernames to bind to the role")
	cmd.Flags().StringArray("group", []string{}, "Groups to bind to the role")
	cmd.Flags().StringArray("serviceaccount", []string{}, "Service accounts to bind to the role, in the format <namespace>:<name>")
	return cmd
}

// CreateClusterRoleBinding is the implementation of the create clusterrolebinding command.
func CreateClusterRoleBinding(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.ClusterRoleBindingV1GeneratorName:
		generator = &kubectl.ClusterRoleBindingGeneratorV1{
			Name:            name,
			ClusterRole:     cmdutil.GetFlagString(cmd, "clusterrole"),
			Users:           cmdutil.GetFlagStringArray(cmd, "user"),
			Groups:          cmdutil.GetFlagStringArray(cmd, "group"),
			ServiceAccounts: cmdutil.GetFlagStringArray(cmd, "serviceaccount"),
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
