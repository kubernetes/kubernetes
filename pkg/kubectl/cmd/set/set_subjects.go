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
	"reflect"
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
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	subjects_long = templates.LongDesc(`
	Update User, Group or ServiceAccount in a RoleBinding/ClusterRoleBinding.`)

	subjects_example = templates.Examples(`
	# Update a ClusterRoleBinding for serviceaccount1
	kubectl set subject clusterrolebinding admin --serviceaccount1=namespace:serviceaccount1

	# Update a RoleBinding for user1, user2, and group1
	kubectl set subject rolebinding admin --user=user1,user2 --group=group1

	# remove user1, user2, and group1 from a RoleBinding
	kubectl set subject rolebinding admin --user=user1,user2 --group=group1 --remove`)
)

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
	Local             bool
	Remove            bool
	Cmd               *cobra.Command

	users           []string
	groups          []string
	serviceaccounts []string

	PrintObject func(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
}

func NewCmdSubjects(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	options := &SubjectOptions{
		Out: out,
		Err: errOut,
	}

	resourceTypesWithPodTemplate := []string{}
	for _, resource := range f.SuggestedPodTemplateResources() {
		resourceTypesWithPodTemplate = append(resourceTypesWithPodTemplate, resource.Resource)
	}

	cmd := &cobra.Command{
		Use:     "subjects (-f FILENAME | TYPE NAME) [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run]",
		Short:   i18n.T("Update User, Group or ServiceAccount in a RoleBinding/ClusterRoleBinding"),
		Long:    subjects_long,
		Example: subjects_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().BoolVar(&options.All, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set resources will NOT contact api-server but run locally.")
	cmd.Flags().BoolVar(&options.Remove, "remove", false, "If true, set resources remove from rolebinding/clusterrolebinding.")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmd.Flags().StringSliceVar(&options.users, "user", []string{}, "usernames to bind to the role")
	cmd.Flags().StringSliceVar(&options.groups, "group", []string{}, "groups to bind to the role")
	cmd.Flags().StringSliceVar(&options.serviceaccounts, "serviceaccount", []string{}, "service accounts to bind to the role")
	return cmd
}

func (o *SubjectOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Mapper, o.Typer = f.Object()
	o.Encoder = f.JSONEncoder()
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.PrintObject = f.PrintObject
	o.Cmd = cmd

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	builder := resource.NewBuilder(o.Mapper, o.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
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
	if len(o.users) == 0 && len(o.groups) == 0 && len(o.serviceaccounts) == 0 {
		return fmt.Errorf("you must specify at least one value of user, group or serviceaccount")
	}

	for _, sa := range o.serviceaccounts {
		tokens := strings.Split(sa, ":")
		if len(tokens) != 2 {
			return fmt.Errorf("serviceaccount must be <namespace>:<name>")
		}
	}

	return nil
}

func (o *SubjectOptions) Run() error {
	allErrs := []error{}
	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		subjects := []rbac.Subject{}
		for _, user := range sets.NewString(o.users...).List() {
			subject := rbac.Subject{
				Kind:     rbac.UserKind,
				APIGroup: rbac.GroupName,
				Name:     user,
			}
			subjects = append(subjects, subject)
		}
		for _, group := range sets.NewString(o.groups...).List() {
			subject := rbac.Subject{
				Kind:     rbac.GroupKind,
				APIGroup: rbac.GroupName,
				Name:     group,
			}
			subjects = append(subjects, subject)
		}
		for _, sa := range sets.NewString(o.serviceaccounts...).List() {
			tokens := strings.Split(sa, ":")
			subject := rbac.Subject{
				Kind:      rbac.ServiceAccountKind,
				Namespace: tokens[0],
				Name:      tokens[1],
			}
			subjects = append(subjects, subject)
		}

		transformed, err := updateSubjectForObject(info.Object, subjects, o.Remove)
		if transformed && err == nil {
			return runtime.Encode(o.Encoder, info.Object)
		}
		return nil, err
	})

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

		if o.Local || cmdutil.GetDryRunFlag(o.Cmd) {
			return o.PrintObject(o.Cmd, o.Mapper, info.Object, o.Out)
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch subjects to rolebinding: %v\n", err))
			continue
		}
		info.Refresh(obj, true)

		if obj, err = resource.NewHelper(info.Client, info.Mapping).Replace(info.Namespace, info.Name, false, obj); err != nil {
			allErrs = append(allErrs, fmt.Errorf("changes to %s/%s can't be recorded: %v\n", info.Mapping.Resource, info.Name, err))
		}
		info.Refresh(obj, true)
		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, false, "resource requirements updated")
	}
	return utilerrors.NewAggregate(allErrs)
}

func updateSubjectForObject(obj runtime.Object, subjects []rbac.Subject, remove bool) (bool, error) {
	switch t := obj.(type) {
	case *rbac.RoleBinding:
		transformed, result := updateSubjects(t.Subjects, subjects, remove)
		t.Subjects = result
		return transformed, nil
	case *rbac.ClusterRoleBinding:
		transformed, result := updateSubjects(t.Subjects, subjects, remove)
		t.Subjects = result
		return transformed, nil
	default:
		return false, fmt.Errorf("setting a subject is only supported for RoleBinding/ClusterRoleBinding")
	}
}

func updateSubjects(existings []rbac.Subject, targets []rbac.Subject, remove bool) (bool, []rbac.Subject) {
	transformed := false
	updated := []rbac.Subject{}
	if remove {
		// remove subjects
		for _, item := range existings {
			if !contain(targets, item) {
				updated = append(updated, item)
			} else {
				transformed = true
			}
		}
	} else {
		// add subjects
		updated = existings
		for _, item := range targets {
			if !contain(existings, item) {
				updated = append(updated, item)
				transformed = true
			}
		}
	}
	return transformed, updated
}

func contain(slice []rbac.Subject, item rbac.Subject) bool {
	for _, v := range slice {
		if reflect.DeepEqual(v, item) {
			return true
		}
	}
	return false
}
