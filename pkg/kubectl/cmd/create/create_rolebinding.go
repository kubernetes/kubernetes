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
	roleBindingLong = templates.LongDesc(i18n.T(`
		Create a RoleBinding for a particular Role or ClusterRole.`))

	roleBindingExample = templates.Examples(i18n.T(`
		  # Create a RoleBinding for user1, user2, and group1 using the admin ClusterRole
		  kubectl create rolebinding admin --clusterrole=admin --user=user1 --user=user2 --group=group1`))
)

// RoleBindingOpts holds the options for 'create rolebinding' sub command
type RoleBindingOpts struct {
	CreateSubcommandOptions *CreateSubcommandOptions
}

// NewCmdCreateRoleBinding returns an initialized Command instance for 'create rolebinding' sub command
func NewCmdCreateRoleBinding(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := &RoleBindingOpts{
		CreateSubcommandOptions: NewCreateSubcommandOptions(ioStreams),
	}

	cmd := &cobra.Command{
		Use:                   "rolebinding NAME --clusterrole=NAME|--role=NAME [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a RoleBinding for a particular Role or ClusterRole"),
		Long:                  roleBindingLong,
		Example:               roleBindingExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Run())
		},
	}

	o.CreateSubcommandOptions.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, generateversioned.RoleBindingV1GeneratorName)
	cmd.Flags().String("clusterrole", "", i18n.T("ClusterRole this RoleBinding should reference"))
	cmd.Flags().String("role", "", i18n.T("Role this RoleBinding should reference"))
	cmd.Flags().StringArray("user", []string{}, "Usernames to bind to the role")
	cmd.Flags().StringArray("group", []string{}, "Groups to bind to the role")
	cmd.Flags().StringArray("serviceaccount", []string{}, "Service accounts to bind to the role, in the format <namespace>:<name>")
	return cmd
}

// Complete completes all the required options
func (o *RoleBindingOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	var generator generate.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case generateversioned.RoleBindingV1GeneratorName:
		generator = &generateversioned.RoleBindingGeneratorV1{
			Name:            name,
			ClusterRole:     cmdutil.GetFlagString(cmd, "clusterrole"),
			Role:            cmdutil.GetFlagString(cmd, "role"),
			Users:           cmdutil.GetFlagStringArray(cmd, "user"),
			Groups:          cmdutil.GetFlagStringArray(cmd, "group"),
			ServiceAccounts: cmdutil.GetFlagStringArray(cmd, "serviceaccount"),
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}

	return o.CreateSubcommandOptions.Complete(f, cmd, args, generator)
}

// Run calls the CreateSubcommandOptions.Run in RoleBindingOpts instance
func (o *RoleBindingOpts) Run() error {
	return o.CreateSubcommandOptions.Run()
}
