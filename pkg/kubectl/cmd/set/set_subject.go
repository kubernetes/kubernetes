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

package set

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
)

var (
	subjectLong = templates.LongDesc(`
	Update User, Group or ServiceAccount in a RoleBinding/ClusterRoleBinding.`)

	subjectExample = templates.Examples(`
	# Update a ClusterRoleBinding for serviceaccount1
	kubectl set subject clusterrolebinding admin --serviceaccount=namespace:serviceaccount1

	# Update a RoleBinding for user1, user2, and group1
	kubectl set subject rolebinding admin --user=user1 --user=user2 --group=group1

	# Print the result (in yaml format) of updating rolebinding subjects from a local, without hitting the server
	kubectl create rolebinding admin --role=admin --user=admin -o yaml --dry-run | kubectl set subject --local -f - --user=foo -o yaml`)
)

type updateSubjects func(existings []rbacv1.Subject, targets []rbacv1.Subject) (bool, []rbacv1.Subject)

// SubjectOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags
type SubjectOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	resource.FilenameOptions

	Infos             []*resource.Info
	Selector          string
	ContainerSelector string
	Output            string
	All               bool
	DryRun            bool
	Local             bool

	Users           []string
	Groups          []string
	ServiceAccounts []string

	namespace string

	PrintObj printers.ResourcePrinterFunc

	genericclioptions.IOStreams
}

// NewSubjectOptions returns an initialized SubjectOptions instance
func NewSubjectOptions(streams genericclioptions.IOStreams) *SubjectOptions {
	return &SubjectOptions{
		PrintFlags: genericclioptions.NewPrintFlags("subjects updated").WithTypeSetter(scheme.Scheme),

		IOStreams: streams,
	}
}

// NewCmdSubject returns the "new subject" sub command
func NewCmdSubject(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewSubjectOptions(streams)
	cmd := &cobra.Command{
		Use:                   "subject (-f FILENAME | TYPE NAME) [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Update User, Group or ServiceAccount in a RoleBinding/ClusterRoleBinding"),
		Long:                  subjectLong,
		Example:               subjectExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run(addSubjects))
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, "the resource to update the subjects")
	cmd.Flags().BoolVar(&o.All, "all", o.All, "Select all resources, including uninitialized ones, in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, not including uninitialized ones, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&o.Local, "local", o.Local, "If true, set subject will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringArrayVar(&o.Users, "user", o.Users, "Usernames to bind to the role")
	cmd.Flags().StringArrayVar(&o.Groups, "group", o.Groups, "Groups to bind to the role")
	cmd.Flags().StringArrayVar(&o.ServiceAccounts, "serviceaccount", o.ServiceAccounts, "Service accounts to bind to the role")
	cmdutil.AddIncludeUninitializedFlag(cmd)
	return cmd
}

// Complete completes all required options
func (o *SubjectOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.DryRun = cmdutil.GetDryRunFlag(cmd)

	if o.DryRun {
		o.PrintFlags.Complete("%s (dry run)")
	}
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = printer.PrintObj

	var enforceNamespace bool
	o.namespace, enforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	builder := f.NewBuilder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		LocalParam(o.Local).
		ContinueOnError().
		NamespaceParam(o.namespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		Flatten()

	if o.Local {
		// if a --local flag was provided, and a resource was specified in the form
		// <resource>/<name>, fail immediately as --local cannot query the api server
		// for the specified resource.
		if len(args) > 0 {
			return resource.LocalResourceError
		}
	} else {
		builder = builder.
			LabelSelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, args...).
			Latest()
	}

	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}

	return nil
}

// Validate makes sure provided values in SubjectOptions are valid
func (o *SubjectOptions) Validate() error {
	if o.All && len(o.Selector) > 0 {
		return fmt.Errorf("cannot set --all and --selector at the same time")
	}
	if len(o.Users) == 0 && len(o.Groups) == 0 && len(o.ServiceAccounts) == 0 {
		return fmt.Errorf("you must specify at least one value of user, group or serviceaccount")
	}

	for _, sa := range o.ServiceAccounts {
		tokens := strings.Split(sa, ":")
		if len(tokens) != 2 || tokens[1] == "" {
			return fmt.Errorf("serviceaccount must be <namespace>:<name>")
		}

		for _, info := range o.Infos {
			_, ok := info.Object.(*rbacv1.ClusterRoleBinding)
			if ok && tokens[0] == "" {
				return fmt.Errorf("serviceaccount must be <namespace>:<name>, namespace must be specified")
			}
		}
	}

	return nil
}

// Run performs the execution of "set subject" sub command
func (o *SubjectOptions) Run(fn updateSubjects) error {
	patches := CalculatePatches(o.Infos, scheme.DefaultJSONEncoder(), func(obj runtime.Object) ([]byte, error) {
		subjects := []rbacv1.Subject{}
		for _, user := range sets.NewString(o.Users...).List() {
			subject := rbacv1.Subject{
				Kind:     rbacv1.UserKind,
				APIGroup: rbacv1.GroupName,
				Name:     user,
			}
			subjects = append(subjects, subject)
		}
		for _, group := range sets.NewString(o.Groups...).List() {
			subject := rbacv1.Subject{
				Kind:     rbacv1.GroupKind,
				APIGroup: rbacv1.GroupName,
				Name:     group,
			}
			subjects = append(subjects, subject)
		}
		for _, sa := range sets.NewString(o.ServiceAccounts...).List() {
			tokens := strings.Split(sa, ":")
			namespace := tokens[0]
			name := tokens[1]
			if len(namespace) == 0 {
				namespace = o.namespace
			}
			subject := rbacv1.Subject{
				Kind:      rbacv1.ServiceAccountKind,
				Namespace: namespace,
				Name:      name,
			}
			subjects = append(subjects, subject)
		}

		transformed, err := updateSubjectForObject(obj, subjects, fn)
		if transformed && err == nil {
			// TODO: switch UpdatePodSpecForObject to work on v1.PodSpec
			return runtime.Encode(scheme.DefaultJSONEncoder(), obj)
		}
		return nil, err
	})

	allErrs := []error{}
	for _, patch := range patches {
		info := patch.Info
		name := info.ObjectName()
		if patch.Err != nil {
			allErrs = append(allErrs, fmt.Errorf("error: %s %v\n", name, patch.Err))
			continue
		}

		//no changes
		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			allErrs = append(allErrs, fmt.Errorf("info: %s was not changed\n", name))
			continue
		}

		if o.Local || o.DryRun {
			if err := o.PrintObj(info.Object, o.Out); err != nil {
				allErrs = append(allErrs, err)
			}
			continue
		}

		actual, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch, nil)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch subjects to rolebinding: %v", err))
			continue
		}

		if err := o.PrintObj(actual, o.Out); err != nil {
			allErrs = append(allErrs, err)
		}
	}
	return utilerrors.NewAggregate(allErrs)
}

//Note: the obj mutates in the function
func updateSubjectForObject(obj runtime.Object, subjects []rbacv1.Subject, fn updateSubjects) (bool, error) {
	switch t := obj.(type) {
	case *rbacv1.RoleBinding:
		transformed, result := fn(t.Subjects, subjects)
		t.Subjects = result
		return transformed, nil
	case *rbacv1.ClusterRoleBinding:
		transformed, result := fn(t.Subjects, subjects)
		t.Subjects = result
		return transformed, nil
	default:
		return false, fmt.Errorf("setting subjects is only supported for RoleBinding/ClusterRoleBinding")
	}
}

func addSubjects(existings []rbacv1.Subject, targets []rbacv1.Subject) (bool, []rbacv1.Subject) {
	transformed := false
	updated := existings
	for _, item := range targets {
		if !contain(existings, item) {
			updated = append(updated, item)
			transformed = true
		}
	}
	return transformed, updated
}

func contain(slice []rbacv1.Subject, item rbacv1.Subject) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}
