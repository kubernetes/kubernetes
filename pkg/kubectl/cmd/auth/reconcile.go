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

package auth

import (
	"errors"
	"fmt"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	rbacv1 "k8s.io/api/rbac/v1"
	rbacv1alpha1 "k8s.io/api/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/printers"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	rbacv1client "k8s.io/client-go/kubernetes/typed/rbac/v1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/registry/rbac/reconciliation"
)

// ReconcileOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ReconcileOptions struct {
	PrintFlags             *genericclioptions.PrintFlags
	FilenameOptions        *resource.FilenameOptions
	DryRun                 bool
	RemoveExtraPermissions bool
	RemoveExtraSubjects    bool

	Visitor         resource.Visitor
	RBACClient      rbacv1client.RbacV1Interface
	NamespaceClient corev1client.CoreV1Interface

	PrintObject printers.ResourcePrinterFunc

	genericclioptions.IOStreams
}

var (
	reconcileLong = templates.LongDesc(`
		Reconciles rules for RBAC Role, RoleBinding, ClusterRole, and ClusterRole binding objects.

		This is preferred to 'apply' for RBAC resources so that proper rule coverage checks are done.`)

	reconcileExample = templates.Examples(`
		# Reconcile rbac resources from a file
		kubectl auth reconcile -f my-rbac-rules.yaml`)
)

func NewReconcileOptions(ioStreams genericclioptions.IOStreams) *ReconcileOptions {
	return &ReconcileOptions{
		FilenameOptions: &resource.FilenameOptions{},
		PrintFlags:      genericclioptions.NewPrintFlags("reconciled").WithTypeSetter(scheme.Scheme),
		IOStreams:       ioStreams,
	}
}

func NewCmdReconcile(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewReconcileOptions(streams)

	cmd := &cobra.Command{
		Use: "reconcile -f FILENAME",
		DisableFlagsInUseLine: true,
		Short:   "Reconciles rules for RBAC Role, RoleBinding, ClusterRole, and ClusterRole binding objects",
		Long:    reconcileLong,
		Example: reconcileExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(cmd, f, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunReconcile())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddFilenameOptionFlags(cmd, o.FilenameOptions, "identifying the resource to reconcile.")
	cmd.Flags().BoolVar(&o.DryRun, "dry-run", o.DryRun, "If true, display results but do not submit changes")
	cmd.Flags().BoolVar(&o.RemoveExtraPermissions, "remove-extra-permissions", o.RemoveExtraPermissions, "If true, removes extra permissions added to roles")
	cmd.Flags().BoolVar(&o.RemoveExtraSubjects, "remove-extra-subjects", o.RemoveExtraSubjects, "If true, removes extra subjects added to rolebindings")
	cmd.MarkFlagRequired("filename")

	return cmd
}

func (o *ReconcileOptions) Complete(cmd *cobra.Command, f cmdutil.Factory, args []string) error {
	if len(args) > 0 {
		return errors.New("no arguments are allowed")
	}

	namespace, enforceNamespace, err := f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		FilenameParam(enforceNamespace, o.FilenameOptions).
		Flatten().
		Do()

	if err := r.Err(); err != nil {
		return err
	}
	o.Visitor = r

	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.RBACClient, err = rbacv1client.NewForConfig(clientConfig)
	if err != nil {
		return err
	}
	o.NamespaceClient, err = corev1client.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	if o.DryRun {
		o.PrintFlags.Complete("%s (dry run)")
	}
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObject = printer.PrintObj
	return nil
}

func (o *ReconcileOptions) Validate() error {
	if o.Visitor == nil {
		return errors.New("ReconcileOptions.Visitor must be set")
	}
	if o.RBACClient == nil {
		return errors.New("ReconcileOptions.RBACClient must be set")
	}
	if o.NamespaceClient == nil {
		return errors.New("ReconcileOptions.NamespaceClient must be set")
	}
	if o.PrintObject == nil {
		return errors.New("ReconcileOptions.Print must be set")
	}
	if o.Out == nil {
		return errors.New("ReconcileOptions.Out must be set")
	}
	if o.ErrOut == nil {
		return errors.New("ReconcileOptions.Err must be set")
	}
	return nil
}

func (o *ReconcileOptions) RunReconcile() error {
	return o.Visitor.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		switch t := info.Object.(type) {
		case *rbacv1.Role:
			reconcileOptions := reconciliation.ReconcileRoleOptions{
				Confirm:                !o.DryRun,
				RemoveExtraPermissions: o.RemoveExtraPermissions,
				Role: reconciliation.RoleRuleOwner{Role: t},
				Client: reconciliation.RoleModifier{
					NamespaceClient: o.NamespaceClient.Namespaces(),
					Client:          o.RBACClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			o.PrintObject(result.Role.GetObject(), o.Out)

		case *rbacv1.ClusterRole:
			reconcileOptions := reconciliation.ReconcileRoleOptions{
				Confirm:                !o.DryRun,
				RemoveExtraPermissions: o.RemoveExtraPermissions,
				Role: reconciliation.ClusterRoleRuleOwner{ClusterRole: t},
				Client: reconciliation.ClusterRoleModifier{
					Client: o.RBACClient.ClusterRoles(),
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			o.PrintObject(result.Role.GetObject(), o.Out)

		case *rbacv1.RoleBinding:
			reconcileOptions := reconciliation.ReconcileRoleBindingOptions{
				Confirm:             !o.DryRun,
				RemoveExtraSubjects: o.RemoveExtraSubjects,
				RoleBinding:         reconciliation.RoleBindingAdapter{RoleBinding: t},
				Client: reconciliation.RoleBindingClientAdapter{
					Client:          o.RBACClient,
					NamespaceClient: o.NamespaceClient.Namespaces(),
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			o.PrintObject(result.RoleBinding.GetObject(), o.Out)

		case *rbacv1.ClusterRoleBinding:
			reconcileOptions := reconciliation.ReconcileRoleBindingOptions{
				Confirm:             !o.DryRun,
				RemoveExtraSubjects: o.RemoveExtraSubjects,
				RoleBinding:         reconciliation.ClusterRoleBindingAdapter{ClusterRoleBinding: t},
				Client: reconciliation.ClusterRoleBindingClientAdapter{
					Client: o.RBACClient.ClusterRoleBindings(),
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			o.PrintObject(result.RoleBinding.GetObject(), o.Out)

		case *rbacv1beta1.Role,
			*rbacv1beta1.RoleBinding,
			*rbacv1beta1.ClusterRole,
			*rbacv1beta1.ClusterRoleBinding,
			*rbacv1alpha1.Role,
			*rbacv1alpha1.RoleBinding,
			*rbacv1alpha1.ClusterRole,
			*rbacv1alpha1.ClusterRoleBinding:
			return fmt.Errorf("only rbac.authorization.k8s.io/v1 is supported: not %T", t)

		default:
			glog.V(1).Infof("skipping %#v", info.Object.GetObjectKind())
			// skip ignored resources
		}

		return nil
	})
}
