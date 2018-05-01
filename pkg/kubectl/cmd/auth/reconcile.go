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

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/rbac"
	internalcoreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	internalrbacclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/printers"
	"k8s.io/kubernetes/pkg/registry/rbac/reconciliation"
)

// ReconcileOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ReconcileOptions struct {
	PrintFlags      *printers.PrintFlags
	FilenameOptions *resource.FilenameOptions

	Visitor         resource.Visitor
	RBACClient      internalrbacclient.RbacInterface
	NamespaceClient internalcoreclient.NamespaceInterface

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
		PrintFlags:      printers.NewPrintFlags("reconciled"),
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
	cmd.MarkFlagRequired("filename")

	return cmd
}

func (o *ReconcileOptions) Complete(cmd *cobra.Command, f cmdutil.Factory, args []string) error {
	if len(args) > 0 {
		return errors.New("no arguments are allowed")
	}

	namespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		WithScheme(legacyscheme.Scheme).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		FilenameParam(enforceNamespace, o.FilenameOptions).
		Flatten().
		Do()

	if err := r.Err(); err != nil {
		return err
	}
	o.Visitor = r

	client, err := f.ClientSet()
	if err != nil {
		return err
	}
	o.RBACClient = client.Rbac()
	o.NamespaceClient = client.Core().Namespaces()

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
		case *rbac.Role:
			reconcileOptions := reconciliation.ReconcileRoleOptions{
				Confirm:                true,
				RemoveExtraPermissions: false,
				Role: reconciliation.RoleRuleOwner{Role: t},
				Client: reconciliation.RoleModifier{
					NamespaceClient: o.NamespaceClient,
					Client:          o.RBACClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			o.PrintObject(result.Role.GetObject(), o.Out)

		case *rbac.ClusterRole:
			reconcileOptions := reconciliation.ReconcileRoleOptions{
				Confirm:                true,
				RemoveExtraPermissions: false,
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

		case *rbac.RoleBinding:
			reconcileOptions := reconciliation.ReconcileRoleBindingOptions{
				Confirm:             true,
				RemoveExtraSubjects: false,
				RoleBinding:         reconciliation.RoleBindingAdapter{RoleBinding: t},
				Client: reconciliation.RoleBindingClientAdapter{
					Client:          o.RBACClient,
					NamespaceClient: o.NamespaceClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			o.PrintObject(result.RoleBinding.GetObject(), o.Out)

		case *rbac.ClusterRoleBinding:
			reconcileOptions := reconciliation.ReconcileRoleBindingOptions{
				Confirm:             true,
				RemoveExtraSubjects: false,
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

		default:
			glog.V(1).Infof("skipping %#v", info.Object.GetObjectKind())
			// skip ignored resources
		}

		return nil
	})
}
