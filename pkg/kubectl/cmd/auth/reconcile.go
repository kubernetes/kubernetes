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
	"io"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	rbacv1 "k8s.io/api/rbac/v1"
	v1alpha1 "k8s.io/api/rbac/v1alpha1"
	v1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientrbacv1 "k8s.io/client-go/kubernetes/typed/rbac/v1"
	clientrbacv1alpha1 "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1"
	clientrbacv1beta1 "k8s.io/client-go/kubernetes/typed/rbac/v1beta1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	reconciliationv1 "k8s.io/kubernetes/pkg/kubectl/reconciliation/v1"
	reconciliationv1alpha1 "k8s.io/kubernetes/pkg/kubectl/reconciliation/v1alpha1"
	reconciliationv1beta1 "k8s.io/kubernetes/pkg/kubectl/reconciliation/v1beta1"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

// ReconcileOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ReconcileOptions struct {
	Visitor            resource.Visitor
	RBACClientV1       clientrbacv1.RbacV1Interface
	RBACClientV1aplha1 clientrbacv1alpha1.RbacV1alpha1Interface
	RBACClientV1beta1  clientrbacv1beta1.RbacV1beta1Interface

	NamespaceClient core.NamespaceInterface

	Print func(*resource.Info) error

	Out io.Writer
	Err io.Writer
}

var (
	reconcileLong = templates.LongDesc(`
		Reconciles rules for RBAC Role, RoleBinding, ClusterRole, and ClusterRole binding objects.

		This is preferred to 'apply' for RBAC resources so that proper rule coverage checks are done.`)

	reconcileExample = templates.Examples(`
		# Reconcile rbac resources from a file
		kubectl auth reconcile -f my-rbac-rules.yaml`)
)

func NewCmdReconcile(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	fileOptions := &resource.FilenameOptions{}
	o := &ReconcileOptions{
		Out: out,
		Err: err,
	}

	cmd := &cobra.Command{
		Use:     "reconcile -f FILENAME",
		Short:   "Reconciles rules for RBAC Role, RoleBinding, ClusterRole, and ClusterRole binding objects",
		Long:    reconcileLong,
		Example: reconcileExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(cmd, f, args, fileOptions))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunReconcile())
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "identifying the resource to reconcile."
	cmdutil.AddFilenameOptionFlags(cmd, fileOptions, usage)
	cmd.MarkFlagRequired("filename")

	return cmd
}

func (o *ReconcileOptions) Complete(cmd *cobra.Command, f cmdutil.Factory, args []string, options *resource.FilenameOptions) error {
	if len(args) > 0 {
		return errors.New("no arguments are allowed")
	}

	namespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		Unstructured().
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options).
		Flatten().
		Do()

	if err := r.Err(); err != nil {
		return err
	}
	o.Visitor = r

	client, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.RBACClientV1 = client.RbacV1()
	o.RBACClientV1aplha1 = client.RbacV1alpha1()
	o.RBACClientV1beta1 = client.RbacV1beta1()
	o.NamespaceClient = client.Core().Namespaces()

	mapper, _ := f.Object()
	dryRun := false
	output := cmdutil.GetFlagString(cmd, "output")
	shortOutput := output == "name"
	o.Print = func(info *resource.Info) error {
		if len(output) > 0 && !shortOutput {
			return f.PrintResourceInfoForCommand(cmd, info, o.Out)
		}
		f.PrintSuccess(mapper, shortOutput, o.Out, info.Mapping.Resource, info.Name, dryRun, "reconciled")
		return nil
	}

	return nil
}

func (o *ReconcileOptions) Validate() error {
	if o.Visitor == nil {
		return errors.New("ReconcileOptions.Visitor must be set")
	}
	if o.RBACClientV1 == nil {
		return errors.New("ReconcileOptions.RBACClientV1 must be set")
	}
	if o.RBACClientV1aplha1 == nil {
		return errors.New("ReconcileOptions.RBACClientV1aplha1 must be set")
	}
	if o.RBACClientV1beta1 == nil {
		return errors.New("ReconcileOptions.RBACClientV1beta1 must be set")
	}
	if o.NamespaceClient == nil {
		return errors.New("ReconcileOptions.NamespaceClient must be set")
	}
	if o.Print == nil {
		return errors.New("ReconcileOptions.Print must be set")
	}
	if o.Out == nil {
		return errors.New("ReconcileOptions.Out must be set")
	}
	if o.Err == nil {
		return errors.New("ReconcileOptions.Err must be set")
	}
	return nil
}

func (o *ReconcileOptions) RunReconcile() error {
	return o.Visitor.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		// shallowInfoCopy this is used to later twiddle the Object for printing
		// we really need more straightforward printing options
		shallowInfoCopy := *info

		_, isUnstructured := info.Object.(runtime.Unstructured)
		if isUnstructured {
			info.Object = info.AsVersioned()
		}

		switch t := info.Object.(type) {
		//v1alpha1
		case *v1alpha1.Role:
			reconcileOptions := reconciliationv1alpha1.ReconcileRoleOptions{
				Confirm:                true,
				RemoveExtraPermissions: false,
				Role: reconciliationv1alpha1.RoleRuleOwner{Role: t},
				Client: reconciliationv1alpha1.RoleModifier{
					NamespaceClient: o.NamespaceClient,
					Client:          o.RBACClientV1aplha1,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.Role.GetObject()
			o.Print(&shallowInfoCopy)

		case *v1alpha1.ClusterRole:
			reconcileOptions := reconciliationv1alpha1.ReconcileRoleOptions{
				Confirm:                true,
				RemoveExtraPermissions: false,
				Role: reconciliationv1alpha1.ClusterRoleRuleOwner{ClusterRole: t},
				Client: reconciliationv1alpha1.ClusterRoleModifier{
					Client: o.RBACClientV1aplha1.ClusterRoles(),
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.Role.GetObject()
			o.Print(&shallowInfoCopy)

		case *v1alpha1.RoleBinding:
			reconcileOptions := reconciliationv1alpha1.ReconcileRoleBindingOptions{
				Confirm:             true,
				RemoveExtraSubjects: false,
				RoleBinding:         reconciliationv1alpha1.RoleBindingAdapter{RoleBinding: t},
				Client: reconciliationv1alpha1.RoleBindingClientAdapter{
					Client:          o.RBACClientV1aplha1,
					NamespaceClient: o.NamespaceClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.RoleBinding.GetObject()
			o.Print(&shallowInfoCopy)

		case *v1alpha1.ClusterRoleBinding:
			reconcileOptions := reconciliationv1alpha1.ReconcileRoleBindingOptions{
				Confirm:             true,
				RemoveExtraSubjects: false,
				RoleBinding:         reconciliationv1alpha1.ClusterRoleBindingAdapter{ClusterRoleBinding: t},
				Client: reconciliationv1alpha1.RoleBindingClientAdapter{
					Client:          o.RBACClientV1aplha1,
					NamespaceClient: o.NamespaceClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.RoleBinding.GetObject()
			o.Print(&shallowInfoCopy)

		//	v1beta1
		case *v1beta1.Role:
			reconcileOptions := reconciliationv1beta1.ReconcileRoleOptions{
				Confirm:                true,
				RemoveExtraPermissions: false,
				Role: reconciliationv1beta1.RoleRuleOwner{Role: t},
				Client: reconciliationv1beta1.RoleModifier{
					NamespaceClient: o.NamespaceClient,
					Client:          o.RBACClientV1beta1,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.Role.GetObject()
			o.Print(&shallowInfoCopy)

		case *v1beta1.ClusterRole:
			reconcileOptions := reconciliationv1beta1.ReconcileRoleOptions{
				Confirm:                true,
				RemoveExtraPermissions: false,
				Role: reconciliationv1beta1.ClusterRoleRuleOwner{ClusterRole: t},
				Client: reconciliationv1beta1.RoleModifier{
					NamespaceClient: o.NamespaceClient,
					Client:          o.RBACClientV1beta1,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.Role.GetObject()
			o.Print(&shallowInfoCopy)

		case *v1beta1.RoleBinding:
			reconcileOptions := reconciliationv1beta1.ReconcileRoleBindingOptions{
				Confirm:             true,
				RemoveExtraSubjects: false,
				RoleBinding:         reconciliationv1beta1.RoleBindingAdapter{RoleBinding: t},
				Client: reconciliationv1beta1.RoleBindingClientAdapter{
					Client:          o.RBACClientV1beta1,
					NamespaceClient: o.NamespaceClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.RoleBinding.GetObject()
			o.Print(&shallowInfoCopy)

		case *v1beta1.ClusterRoleBinding:
			reconcileOptions := reconciliationv1beta1.ReconcileRoleBindingOptions{
				Confirm:             true,
				RemoveExtraSubjects: false,
				RoleBinding:         reconciliationv1beta1.ClusterRoleBindingAdapter{ClusterRoleBinding: t},
				Client: reconciliationv1beta1.RoleBindingClientAdapter{
					Client:          o.RBACClientV1beta1,
					NamespaceClient: o.NamespaceClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.RoleBinding.GetObject()
			o.Print(&shallowInfoCopy)

		// v1
		case *rbacv1.Role:
			reconcileOptions := reconciliationv1.ReconcileRoleOptions{
				Confirm:                true,
				RemoveExtraPermissions: false,
				Role: reconciliationv1.RoleRuleOwner{Role: t},
				Client: reconciliationv1.RoleModifier{
					NamespaceClient: o.NamespaceClient,
					Client:          o.RBACClientV1,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.Role.GetObject()
			o.Print(&shallowInfoCopy)

		case *rbacv1.ClusterRole:
			reconcileOptions := reconciliationv1.ReconcileRoleOptions{
				Confirm:                true,
				RemoveExtraPermissions: false,
				Role: reconciliationv1.ClusterRoleRuleOwner{ClusterRole: t},
				Client: reconciliationv1.ClusterRoleModifier{
					Client: o.RBACClientV1.ClusterRoles(),
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.Role.GetObject()
			o.Print(&shallowInfoCopy)

		case *rbacv1.RoleBinding:
			reconcileOptions := reconciliationv1.ReconcileRoleBindingOptions{
				Confirm:             true,
				RemoveExtraSubjects: false,
				RoleBinding:         reconciliationv1.RoleBindingAdapter{RoleBinding: t},
				Client: reconciliationv1.RoleBindingClientAdapter{
					Client:          o.RBACClientV1,
					NamespaceClient: o.NamespaceClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.RoleBinding.GetObject()
			o.Print(&shallowInfoCopy)

		case *rbacv1.ClusterRoleBinding:
			reconcileOptions := reconciliationv1.ReconcileRoleBindingOptions{
				Confirm:             true,
				RemoveExtraSubjects: false,
				RoleBinding:         reconciliationv1.ClusterRoleBindingAdapter{ClusterRoleBinding: t},
				Client: reconciliationv1.ClusterRoleBindingClientAdapter{
					Client: o.RBACClientV1.ClusterRoleBindings(),
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.RoleBinding.GetObject()
			o.Print(&shallowInfoCopy)

		default:
			glog.V(1).Infof("skipping %#v", info.Object.GetObjectKind())
			// skip ignored resources
		}

		return nil
	})
}
