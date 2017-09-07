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

	"k8s.io/kubernetes/pkg/apis/rbac"
	internalcoreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	internalrbacclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/registry/rbac/reconciliation"
)

// ReconcileOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ReconcileOptions struct {
	ResourceBuilder *resource.Builder
	RBACClient      internalrbacclient.RbacInterface
	CoreClient      internalcoreclient.CoreInterface

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
	o.ResourceBuilder = f.NewBuilder(true).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options).
		Flatten()

	client, err := f.ClientSet()
	if err != nil {
		return err
	}
	o.RBACClient = client.Rbac()
	o.CoreClient = client.Core()

	mapper, _ := f.Object()
	dryRun := false
	output := cmdutil.GetFlagString(cmd, "output")
	shortOutput := output == "name"
	o.Print = func(info *resource.Info) error {
		if len(output) > 0 && !shortOutput {
			return cmdutil.PrintResourceInfoForCommand(cmd, info, f, o.Out)
		}
		cmdutil.PrintSuccess(mapper, shortOutput, o.Out, info.Mapping.Resource, info.Name, dryRun, "reconciled")
		return nil
	}

	return nil
}

func (o *ReconcileOptions) Validate() error {
	return nil
}

func (o *ReconcileOptions) RunReconcile() error {
	r := o.ResourceBuilder.Do()
	err := r.Err()
	if err != nil {
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		// shallowInfoCopy this is used to later twiddle the Object for printing
		// we really need more straightforward printing options
		shallowInfoCopy := *info

		switch t := info.Object.(type) {
		case *rbac.Role:
			reconcileOptions := reconciliation.ReconcileRoleOptions{
				Confirm:                true,
				RemoveExtraPermissions: false,
				Role: reconciliation.RoleRuleOwner{Role: t},
				Client: reconciliation.RoleModifier{
					NamespaceClient: o.CoreClient.Namespaces(),
					Client:          o.RBACClient,
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.Role.GetObject()
			o.Print(&shallowInfoCopy)
			return nil

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
			shallowInfoCopy.Object = result.Role.GetObject()
			o.Print(&shallowInfoCopy)
			return nil

		case *rbac.RoleBinding:
			reconcileOptions := reconciliation.ReconcileRoleBindingOptions{
				Confirm:             true,
				RemoveExtraSubjects: false,
				RoleBinding:         reconciliation.RoleBindingAdapter{RoleBinding: t},
				Client: reconciliation.RoleBindingClientAdapter{
					Client:          o.RBACClient,
					NamespaceClient: o.CoreClient.Namespaces(),
				},
			}
			result, err := reconcileOptions.Run()
			if err != nil {
				return err
			}
			shallowInfoCopy.Object = result.RoleBinding.GetObject()
			o.Print(&shallowInfoCopy)
			return nil

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
			shallowInfoCopy.Object = result.RoleBinding.GetObject()
			o.Print(&shallowInfoCopy)
			return nil

		default:
			glog.V(1).Infof("skipping %#v", info.Object.GetObjectKind())
			// skip ignored resources
		}

		return nil
	})

	return err
}
