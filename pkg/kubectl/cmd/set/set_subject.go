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
	"io"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	subject_long = templates.LongDesc(`
	Update User, Group or ServiceAccount in a RoleBinding/ClusterRoleBinding.`)

	subject_example = templates.Examples(`
	# Update a ClusterRoleBinding for serviceaccount1
	kubectl set subject clusterrolebinding admin --serviceaccount=namespace:serviceaccount1

	# Update a RoleBinding for user1, user2, and group1
	kubectl set subject rolebinding admin --user=user1 --user=user2 --group=group1

	# Print the result (in yaml format) of updating rolebinding subjects from a local, without hitting the server
	kubectl create rolebinding admin --role=admin --user=admin -o yaml --dry-run | kubectl set subject --local -f - --user=foo -o yaml`)
)

type updateSubjects func(existings []rbac.Subject, targets []rbac.Subject) (bool, []rbac.Subject)

// SubjectOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags
type SubjectOptions struct {
	resource.FilenameOptions

	Mapper            meta.RESTMapper
	Typer             runtime.ObjectTyper
	Infos             []*resource.Info
	Encoder           runtime.Encoder
	Out               io.Writer
	Err               io.Writer
	Selector          string
	ContainerSelector string
	ShortOutput       bool
	All               bool
	DryRun            bool
	Local             bool

	Users           []string
	Groups          []string
	ServiceAccounts []string

	PrintObject func(mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
}

func NewCmdSubject(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	options := &SubjectOptions{
		Out: out,
		Err: errOut,
	}

	cmd := &cobra.Command{
		Use:     "subject (-f FILENAME | TYPE NAME) [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run]",
		Short:   i18n.T("Update User, Group or ServiceAccount in a RoleBinding/ClusterRoleBinding"),
		Long:    subject_long,
		Example: subject_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run(f, addSubjects))
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "the resource to update the subjects"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().BoolVar(&options.All, "all", false, "Select all resources, including uninitialized ones, in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, not including uninitialized ones, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set resources will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringArrayVar(&options.Users, "user", []string{}, "Usernames to bind to the role")
	cmd.Flags().StringArrayVar(&options.Groups, "group", []string{}, "Groups to bind to the role")
	cmd.Flags().StringArrayVar(&options.ServiceAccounts, "serviceaccount", []string{}, "Service accounts to bind to the role")
	cmdutil.AddIncludeUninitializedFlag(cmd)
	return cmd
}

func (o *SubjectOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Local = cmdutil.GetFlagBool(cmd, "local")
	o.Mapper, o.Typer = f.Object()
	o.Encoder = f.JSONEncoder()
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.PrintObject = func(mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
		return f.PrintObject(cmd, o.Local, mapper, obj, out)
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	builder := f.NewBuilder(!o.Local).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten()

	if !o.Local {
		builder = builder.
			SelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, args...).
			Latest()
	}
	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}

	return nil
}

func (o *SubjectOptions) Validate() error {
	if len(o.Users) == 0 && len(o.Groups) == 0 && len(o.ServiceAccounts) == 0 {
		return fmt.Errorf("you must specify at least one value of user, group or serviceaccount")
	}

	for _, sa := range o.ServiceAccounts {
		tokens := strings.Split(sa, ":")
		if len(tokens) != 2 || tokens[1] == "" {
			return fmt.Errorf("serviceaccount must be <namespace>:<name>")
		}

		for _, info := range o.Infos {
			_, ok := info.Object.(*rbac.ClusterRoleBinding)
			if ok && tokens[0] == "" {
				return fmt.Errorf("serviceaccount must be <namespace>:<name>, namespace must be specified")
			}
		}
	}

	return nil
}

func (o *SubjectOptions) Run(f cmdutil.Factory, fn updateSubjects) error {
	var err error
	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		subjects := []rbac.Subject{}
		for _, user := range sets.NewString(o.Users...).List() {
			subject := rbac.Subject{
				Kind:     rbac.UserKind,
				APIGroup: rbac.GroupName,
				Name:     user,
			}
			subjects = append(subjects, subject)
		}
		for _, group := range sets.NewString(o.Groups...).List() {
			subject := rbac.Subject{
				Kind:     rbac.GroupKind,
				APIGroup: rbac.GroupName,
				Name:     group,
			}
			subjects = append(subjects, subject)
		}
		for _, sa := range sets.NewString(o.ServiceAccounts...).List() {
			tokens := strings.Split(sa, ":")
			namespace := tokens[0]
			name := tokens[1]
			if len(namespace) == 0 {
				namespace, _, err = f.DefaultNamespace()
				if err != nil {
					return nil, err
				}
			}
			subject := rbac.Subject{
				Kind:      rbac.ServiceAccountKind,
				Namespace: namespace,
				Name:      name,
			}
			subjects = append(subjects, subject)
		}

		transformed, err := updateSubjectForObject(info.Object, subjects, fn)
		if transformed && err == nil {
			return runtime.Encode(o.Encoder, info.Object)
		}
		return nil, err
	})

	allErrs := []error{}
	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			allErrs = append(allErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}

		//no changes
		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			allErrs = append(allErrs, fmt.Errorf("info: %s %q was not changed\n", info.Mapping.Resource, info.Name))
			continue
		}

		if o.Local || o.DryRun {
			return o.PrintObject(o.Mapper, info.Object, o.Out)
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch subjects to rolebinding: %v\n", err))
			continue
		}
		info.Refresh(obj, true)

		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, false, "subjects updated")
	}
	return utilerrors.NewAggregate(allErrs)
}

//Note: the obj mutates in the function
func updateSubjectForObject(obj runtime.Object, subjects []rbac.Subject, fn updateSubjects) (bool, error) {
	switch t := obj.(type) {
	case *rbac.RoleBinding:
		transformed, result := fn(t.Subjects, subjects)
		t.Subjects = result
		return transformed, nil
	case *rbac.ClusterRoleBinding:
		transformed, result := fn(t.Subjects, subjects)
		t.Subjects = result
		return transformed, nil
	default:
		return false, fmt.Errorf("setting subjects is only supported for RoleBinding/ClusterRoleBinding")
	}
}

func addSubjects(existings []rbac.Subject, targets []rbac.Subject) (bool, []rbac.Subject) {
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

func contain(slice []rbac.Subject, item rbac.Subject) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}
