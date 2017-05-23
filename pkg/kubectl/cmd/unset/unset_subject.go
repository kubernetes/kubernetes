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

package unset

import (
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdset "k8s.io/kubernetes/pkg/kubectl/cmd/set"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	subject_long = templates.LongDesc(`
	Remove User, Group or ServiceAccount from a RoleBinding/ClusterRoleBinding.`)

	subject_example = templates.Examples(`
	# Remove serviceaccount1 from a ClusterRoleBinding
	kubectl unset subject clusterrolebinding admin --serviceaccount=namespace:serviceaccount1

	# Remove user1, user2, and group1 from a RoleBinding
	kubectl unset subject rolebinding admin --user=user1 --user=user2 --group=group1

	# Print the result (in yaml format) of remove rolebinding subjects from a local, without hitting the server
	kubectl create rolebinding admin --role=admin --user=admin --user=foo -o yaml --dry-run | kubectl unset subject --local -f - --user=foo -o yaml`)
)

// SubjectOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags
type SubjectOptions struct {
	*cmdset.SubjectOptions
}

func NewCmdSubject(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	options := &SubjectOptions{
		&cmdset.SubjectOptions{
			Out: out,
			Err: errOut,
		},
	}

	cmd := &cobra.Command{
		Use:     "subject (-f FILENAME | TYPE NAME) [--user=username] [--group=groupname] [--serviceaccount=namespace:serviceaccountname] [--dry-run]",
		Short:   i18n.T("Remove User, Group or ServiceAccount from a RoleBinding/ClusterRoleBinding"),
		Long:    subject_long,
		Example: subject_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run(f, removeSubjects))
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "the resource to remove the subjects"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().BoolVar(&options.All, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, unset resources will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringArrayVar(&options.Users, "user", []string{}, "usernames to bind to the role")
	cmd.Flags().StringArrayVar(&options.Groups, "group", []string{}, "groups to bind to the role")
	cmd.Flags().StringArrayVar(&options.ServiceAccounts, "serviceaccount", []string{}, "service accounts to bind to the role")
	return cmd
}

func removeSubjects(existings []rbac.Subject, targets []rbac.Subject) (bool, []rbac.Subject) {
	transformed := false
	updated := []rbac.Subject{}
	for _, item := range existings {
		if !contain(targets, item) {
			updated = append(updated, item)
		} else {
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
