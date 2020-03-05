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

package create

import (
	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/generate"
	generateversioned "k8s.io/kubectl/pkg/generate/versioned"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	clusterRoleBindingLong = templates.LongDesc(i18n.T(`
		Create a ClusterRoleBinding for a particular ClusterRole.`))

	clusterRoleBindingExample = templates.Examples(i18n.T(`
		  # Create a ClusterRoleBinding for user1, user2, and group1 using the cluster-admin ClusterRole
		  kubectl create clusterrolebinding cluster-admin --clusterrole=cluster-admin --user=user1 --user=user2 --group=group1`))
)

// ClusterRoleBindingOpts is returned by NewCmdCreateClusterRoleBinding
type ClusterRoleBindingOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateClusterRoleBinding returns an initialized command instance of ClusterRoleBinding
func NewCmdCreateClusterRoleBinding(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := &ClusterRoleBindingOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use:                   "clusterrolebinding NAME --clusterrole=NAME [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a ClusterRoleBinding for a particular ClusterRole"),
		Long:                  clusterRoleBindingLong,
		Example:               clusterRoleBindingExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Run())
		},
	}

	o.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, generateversioned.ClusterRoleBindingV1GeneratorName)
	cmd.Flags().String("clusterrole", "", i18n.T("ClusterRole this ClusterRoleBinding should reference"))
	cmd.MarkFlagCustom("clusterrole", "__kubectl_get_resource_clusterrole")
	cmd.Flags().StringArray("user", []string{}, "Usernames to bind to the clusterrole")
	cmd.Flags().StringArray("group", []string{}, "Groups to bind to the clusterrole")
	cmd.Flags().StringArray("serviceaccount", []string{}, "Service accounts to bind to the clusterrole, in the format <namespace>:<name>")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.CreateSubcommandOptions.FieldManager, "kubectl-create")
	return cmd
}

// Complete completes all the required options
func (o *ClusterRoleBindingOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	var generator generate.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case generateversioned.ClusterRoleBindingV1GeneratorName:
		generator = &generateversioned.ClusterRoleBindingGeneratorV1{
			Name:            name,
			ClusterRole:     cmdutil.GetFlagString(cmd, "clusterrole"),
			Users:           cmdutil.GetFlagStringArray(cmd, "user"),
			Groups:          cmdutil.GetFlagStringArray(cmd, "group"),
			ServiceAccounts: cmdutil.GetFlagStringArray(cmd, "serviceaccount"),
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// Run calls the CreateSubcommandOptions.Run in ClusterRoleBindingOpts instance
func (o *ClusterRoleBindingOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}
