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
	"k8s.io/cli-runtime/pkg/genericiooptions"
	rbacclientv1 "k8s.io/client-go/kubernetes/typed/rbac/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	roleBindingLong = templates.LongDesc(i18n.T(`
		Create a role binding for a particular role or cluster role.`))

	roleBindingExample = templates.Examples(i18n.T(`
		# Create a role binding for user1, user2, and group1 using the admin cluster role
		kubectl create rolebinding admin --clusterrole=admin --user=user1 --user=user2 --group=group1

		# Create a role binding for serviceaccount monitoring:sa-dev using the admin role
		kubectl create rolebinding admin-binding --role=admin --serviceaccount=monitoring:sa-dev`))
)

// RoleBindingOptions holds the options for 'create rolebinding' sub command
type RoleBindingOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	PrintObj   func(obj runtime.Object) error

	Name             string
	Namespace        string
	EnforceNamespace bool
	ClusterRole      string
	Role             string
	Users            []string
	Groups           []string
	ServiceAccounts  []string
	FieldManager     string
	CreateAnnotation bool

	Client              rbacclientv1.RbacV1Interface
	DryRunStrategy      cmdutil.DryRunStrategy
	ValidationDirective string

	genericiooptions.IOStreams
}

// NewRoleBindingOptions creates a new *RoleBindingOptions with sane defaults
func NewRoleBindingOptions(ioStreams genericiooptions.IOStreams) *RoleBindingOptions {
	return &RoleBindingOptions{
		Users:           []string{},
		Groups:          []string{},
		ServiceAccounts: []string{},
		PrintFlags:      genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:       ioStreams,
	}
}

// NewCmdCreateRoleBinding returns an initialized Command instance for 'create rolebinding' sub command
func NewCmdCreateRoleBinding(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewRoleBindingOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "rolebinding NAME --clusterrole=NAME|--role=NAME [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a role binding for a particular role or cluster role"),
		Long:                  roleBindingLong,
		Example:               roleBindingExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringVar(&o.ClusterRole, "clusterrole", "", i18n.T("ClusterRole this RoleBinding should reference"))
	cmd.Flags().StringVar(&o.Role, "role", "", i18n.T("Role this RoleBinding should reference"))
	cmd.Flags().StringArrayVar(&o.Users, "user", o.Users, "Usernames to bind to the role. The flag can be repeated to add multiple users.")
	cmd.Flags().StringArrayVar(&o.Groups, "group", o.Groups, "Groups to bind to the role. The flag can be repeated to add multiple groups.")
	cmd.Flags().StringArrayVar(&o.ServiceAccounts, "serviceaccount", o.ServiceAccounts, "Service accounts to bind to the role, in the format <namespace>:<name>. The flag can be repeated to add multiple service accounts.")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")
	return cmd
}

// Complete completes all the required options
func (o *RoleBindingOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Name, err = NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.Client, err = rbacclientv1.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.CreateAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)
	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	o.ValidationDirective, err = cmdutil.GetValidationDirective(cmd)
	return err
}

// Validate validates required fields are set
func (o *RoleBindingOptions) Validate() error {
	if len(o.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if (len(o.ClusterRole) == 0) == (len(o.Role) == 0) {
		return fmt.Errorf("exactly one of clusterrole or role must be specified")
	}
	return nil
}

// Run performs the execution of 'create rolebinding' sub command
func (o *RoleBindingOptions) Run() error {
	roleBinding, err := o.createRoleBinding()
	if err != nil {
		return err
	}
	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, roleBinding, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		createOptions.FieldValidation = o.ValidationDirective
		if o.DryRunStrategy == cmdutil.DryRunServer {
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		roleBinding, err = o.Client.RoleBindings(o.Namespace).Create(context.TODO(), roleBinding, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create rolebinding: %v", err)
		}
	}
	return o.PrintObj(roleBinding)
}

func (o *RoleBindingOptions) createRoleBinding() (*rbacv1.RoleBinding, error) {
	namespace := ""
	if o.EnforceNamespace {
		namespace = o.Namespace
	}

	roleBinding := &rbacv1.RoleBinding{
		TypeMeta: metav1.TypeMeta{APIVersion: rbacv1.SchemeGroupVersion.String(), Kind: "RoleBinding"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      o.Name,
			Namespace: namespace,
		},
	}

	switch {
	case len(o.Role) > 0:
		roleBinding.RoleRef = rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "Role",
			Name:     o.Role,
		}
	case len(o.ClusterRole) > 0:
		roleBinding.RoleRef = rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "ClusterRole",
			Name:     o.ClusterRole,
		}
	}

	for _, user := range o.Users {
		roleBinding.Subjects = append(roleBinding.Subjects, rbacv1.Subject{
			Kind:     rbacv1.UserKind,
			APIGroup: rbacv1.GroupName,
			Name:     user,
		})
	}

	for _, group := range o.Groups {
		roleBinding.Subjects = append(roleBinding.Subjects, rbacv1.Subject{
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
		roleBinding.Subjects = append(roleBinding.Subjects, rbacv1.Subject{
			Kind:      rbacv1.ServiceAccountKind,
			APIGroup:  "",
			Namespace: tokens[0],
			Name:      tokens[1],
		})
	}
	return roleBinding, nil
}
