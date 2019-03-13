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

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog"

	rbacv1 "k8s.io/api/rbac/v1"
	rbacv1alpha1 "k8s.io/api/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	rbacv1client "k8s.io/client-go/kubernetes/typed/rbac/v1"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
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

		Missing objects are created, and the containing namespace is created for namespaced objects, if required.

		Existing roles are updated to include the permissions in the input objects,
		and remove extra permissions if --remove-extra-permissions is specified.

		Existing bindings are updated to include the subjects in the input objects,
		and remove extra subjects if --remove-extra-subjects is specified.

		This is preferred to 'apply' for RBAC resources so that semantically-aware merging of rules and subjects is done.`)

	reconcileExample = templates.Examples(`
		# Reconcile rbac resources from a file
		kubectl auth reconcile -f my-rbac-rules.yaml`)
)

// NewReconcileOptions returns a new ReconcileOptions instance
func NewReconcileOptions(ioStreams genericclioptions.IOStreams) *ReconcileOptions {
	return &ReconcileOptions{
		FilenameOptions: &resource.FilenameOptions{},
		PrintFlags:      genericclioptions.NewPrintFlags("reconciled").WithTypeSetter(scheme.Scheme),
		IOStreams:       ioStreams,
	}
}

// NewCmdReconcile holds the options for 'auth reconcile' sub command
func NewCmdReconcile(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewReconcileOptions(streams)

	cmd := &cobra.Command{
		Use:                   "reconcile -f FILENAME",
		DisableFlagsInUseLine: true,
		Short:                 "Reconciles rules for RBAC Role, RoleBinding, ClusterRole, and ClusterRole binding objects",
		Long:                  reconcileLong,
		Example:               reconcileExample,
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

	return cmd
}

// Complete completes all the required options
func (o *ReconcileOptions) Complete(cmd *cobra.Command, f cmdutil.Factory, args []string) error {
	if err := o.FilenameOptions.RequireFilenameOrKustomize(); err != nil {
		return err
	}

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

// Validate makes sure provided values for ReconcileOptions are valid
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

// RunReconcile performs the execution
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
				Role:                   reconciliation.RoleRuleOwner{Role: t},
				Client: reconciliation.RoleModifier{
					NamespaceClient: o.NamespaceClient.Namespaces(),
					Client:          o.RBACClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			o.printResults(result.Role.GetObject(), nil, nil, result.MissingRules, result.ExtraRules, result.Operation, result.Protected)

		case *rbacv1.ClusterRole:
			reconcileOptions := reconciliation.ReconcileRoleOptions{
				Confirm:                !o.DryRun,
				RemoveExtraPermissions: o.RemoveExtraPermissions,
				Role:                   reconciliation.ClusterRoleRuleOwner{ClusterRole: t},
				Client: reconciliation.ClusterRoleModifier{
					Client: o.RBACClient.ClusterRoles(),
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			o.printResults(result.Role.GetObject(), nil, nil, result.MissingRules, result.ExtraRules, result.Operation, result.Protected)

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
			o.printResults(result.RoleBinding.GetObject(), result.MissingSubjects, result.ExtraSubjects, nil, nil, result.Operation, result.Protected)

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
			o.printResults(result.RoleBinding.GetObject(), result.MissingSubjects, result.ExtraSubjects, nil, nil, result.Operation, result.Protected)

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
			klog.V(1).Infof("skipping %#v", info.Object.GetObjectKind())
			// skip ignored resources
		}

		return nil
	})
}

func (o *ReconcileOptions) printResults(object runtime.Object,
	missingSubjects, extraSubjects []rbacv1.Subject,
	missingRules, extraRules []rbacv1.PolicyRule,
	operation reconciliation.ReconcileOperation,
	protected bool) {

	o.PrintObject(object, o.Out)

	caveat := ""
	if protected {
		caveat = ", but object opted out (rbac.authorization.kubernetes.io/autoupdate: false)"
	}
	switch operation {
	case reconciliation.ReconcileNone:
		return
	case reconciliation.ReconcileCreate:
		fmt.Fprintf(o.ErrOut, "\treconciliation required create%s\n", caveat)
	case reconciliation.ReconcileUpdate:
		fmt.Fprintf(o.ErrOut, "\treconciliation required update%s\n", caveat)
	case reconciliation.ReconcileRecreate:
		fmt.Fprintf(o.ErrOut, "\treconciliation required recreate%s\n", caveat)
	}

	if len(missingSubjects) > 0 {
		fmt.Fprintf(o.ErrOut, "\tmissing subjects added:\n")
		for _, s := range missingSubjects {
			fmt.Fprintf(o.ErrOut, "\t\t%+v\n", s)
		}
	}
	if o.RemoveExtraSubjects {
		if len(extraSubjects) > 0 {
			fmt.Fprintf(o.ErrOut, "\textra subjects removed:\n")
			for _, s := range extraSubjects {
				fmt.Fprintf(o.ErrOut, "\t\t%+v\n", s)
			}
		}
	}
	if len(missingRules) > 0 {
		fmt.Fprintf(o.ErrOut, "\tmissing rules added:\n")
		for _, r := range missingRules {
			fmt.Fprintf(o.ErrOut, "\t\t%+v\n", r)
		}
	}
	if o.RemoveExtraPermissions {
		if len(extraRules) > 0 {
			fmt.Fprintf(o.ErrOut, "\textra rules removed:\n")
			for _, r := range extraRules {
				fmt.Fprintf(o.ErrOut, "\t\t%+v\n", r)
			}
		}
	}
}
