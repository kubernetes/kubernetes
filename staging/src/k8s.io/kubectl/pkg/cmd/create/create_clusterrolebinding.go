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
	"context"
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	rbacclientv1 "k8s.io/client-go/kubernetes/typed/rbac/v1"
	"k8s.io/kubectl/pkg/cmd/get"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	clusterRoleBindingLong = templates.LongDesc(i18n.T(`
		Create a cluster role binding for a particular cluster role.`))

	clusterRoleBindingExample = templates.Examples(i18n.T(`
		  # Create a cluster role binding for user1, user2, and group1 using the cluster-admin cluster role
		  kubectl create clusterrolebinding cluster-admin --clusterrole=cluster-admin --user=user1 --user=user2 --group=group1`))
)

// ClusterRoleBindingOptions is returned by NewCmdCreateClusterRoleBinding
type ClusterRoleBindingOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	PrintObj   func(obj runtime.Object) error

	Name             string
	ClusterRole      string
	Users            []string
	Groups           []string
	ServiceAccounts  []string
	FieldManager     string
	CreateAnnotation bool

	Client         rbacclientv1.RbacV1Interface
	DryRunStrategy cmdutil.DryRunStrategy
	DryRunVerifier *resource.DryRunVerifier

	genericclioptions.IOStreams
}

// NewClusterRoleBindingOptions creates a new *ClusterRoleBindingOptions with sane defaults
func NewClusterRoleBindingOptions(ioStreams genericclioptions.IOStreams) *ClusterRoleBindingOptions {
	return &ClusterRoleBindingOptions{
		Users:           []string{},
		Groups:          []string{},
		ServiceAccounts: []string{},
		PrintFlags:      genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:       ioStreams,
	}
}

// NewCmdCreateClusterRoleBinding returns an initialized command instance of ClusterRoleBinding
func NewCmdCreateClusterRoleBinding(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewClusterRoleBindingOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "clusterrolebinding NAME --clusterrole=NAME [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a cluster role binding for a particular cluster role"),
		Long:                  clusterRoleBindingLong,
		Example:               clusterRoleBindingExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringVar(&o.ClusterRole, "clusterrole", "", i18n.T("ClusterRole this ClusterRoleBinding should reference"))
	cmd.MarkFlagRequired("clusterrole")
	cmd.Flags().StringArrayVar(&o.Users, "user", o.Users, "Usernames to bind to the clusterrole")
	cmd.Flags().StringArrayVar(&o.Groups, "group", o.Groups, "Groups to bind to the clusterrole")
	cmd.Flags().StringArrayVar(&o.ServiceAccounts, "serviceaccount", o.ServiceAccounts, "Service accounts to bind to the clusterrole, in the format <namespace>:<name>")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")

	// Completion for relevant flags
	cmdutil.CheckErr(cmd.RegisterFlagCompletionFunc(
		"clusterrole",
		func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
			return get.CompGetResource(f, cmd, "clusterrole", toComplete), cobra.ShellCompDirectiveNoFileComp
		}))

	return cmd
}

// Complete completes all the required options
func (o *ClusterRoleBindingOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Name, err = NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	cs, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.Client = cs.RbacV1()

	o.CreateAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resource.NewDryRunVerifier(dynamicClient, f.OpenAPIGetter())
	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	return nil
}

// Run calls the CreateSubcommandOptions.Run in ClusterRoleBindingOptions instance
func (o *ClusterRoleBindingOptions) Run() error {
	clusterRoleBinding, err := o.createClusterRoleBinding()
	if err != nil {
		return err
	}

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, clusterRoleBinding, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(clusterRoleBinding.GroupVersionKind()); err != nil {
				return err
			}
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		var err error
		clusterRoleBinding, err = o.Client.ClusterRoleBindings().Create(context.TODO(), clusterRoleBinding, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create clusterrolebinding: %v", err)
		}
	}

	return o.PrintObj(clusterRoleBinding)
}

func (o *ClusterRoleBindingOptions) createClusterRoleBinding() (*rbacv1.ClusterRoleBinding, error) {
	clusterRoleBinding := &rbacv1.ClusterRoleBinding{
		TypeMeta: metav1.TypeMeta{APIVersion: rbacv1.SchemeGroupVersion.String(), Kind: "ClusterRoleBinding"},
		ObjectMeta: metav1.ObjectMeta{
			Name: o.Name,
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "ClusterRole",
			Name:     o.ClusterRole,
		},
	}

	for _, user := range o.Users {
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbacv1.Subject{
			Kind:     rbacv1.UserKind,
			APIGroup: rbacv1.GroupName,
			Name:     user,
		})
	}

	for _, group := range o.Groups {
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbacv1.Subject{
			Kind:     rbacv1.GroupKind,
			APIGroup: rbacv1.GroupName,
			Name:     group,
		})
	}

	for _, sa := range o.ServiceAccounts {
		tokens := strings.Split(sa, ":")
		if len(tokens) != 2 || tokens[0] == "" || tokens[1] == "" {
			return nil, fmt.Errorf("serviceaccount must be <namespace>:<name>")
		}
		clusterRoleBinding.Subjects = append(clusterRoleBinding.Subjects, rbacv1.Subject{
			Kind:      rbacv1.ServiceAccountKind,
			APIGroup:  "",
			Namespace: tokens[0],
			Name:      tokens[1],
		})
	}

	return clusterRoleBinding, nil
}
