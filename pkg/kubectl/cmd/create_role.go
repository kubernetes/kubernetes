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

package cmd

import (
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	roleLong = templates.LongDesc(i18n.T(`
		Create a role with single rule.`))

	roleExample = templates.Examples(i18n.T(`
		# Create a Role named "pod-reader" that allows user to perform "get", "watch" and "list" on pods
		kubectl create role pod-reader --verb=get --verb=list --verb=watch --resource=pods

		# Create a Role named "pod-reader" with ResourceName specified
		kubectl create role pod-reader --verb=get,list,watch --resource=pods --resource-name=readablepod --resource-name=anotherpod

		# Create a Role named "foo" with API Group specified
		kubectl create role foo --verb=get,list,watch --resource=rs.extensions

		# Create a Role named "foo" with SubResource specified
		kubectl create role foo --verb=get,list,watch --resource=pods,pods/status`))
)

// Role is a command to ease creating Roles.
func NewCmdCreateRole(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "role NAME [--verb=verb] [--resource=resource.group/subresource] [--resource-name=resourcename] [--dry-run]",
		Short:   roleLong,
		Long:    roleLong,
		Example: roleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(CreateRole(f, cmdOut, cmd, args))
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.RoleV1GeneratorName)
	cmd.Flags().StringSlice("verb", []string{}, "verb that applies to the resources contained in the rule")
	cmd.Flags().StringSlice("resource", []string{}, "resource that the rule applies to")
	cmd.Flags().StringArray("resource-name", []string{}, "resource in the white list that the rule applies to, repeat this flag for multiple items")

	return cmd
}

func CreateRole(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	mapper, _ := f.Object()

	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.RoleV1GeneratorName:
		generator = &kubectl.RoleGeneratorV1{
			Mapper:        mapper,
			Name:          name,
			Verbs:         cmdutil.GetFlagStringSlice(cmd, "verb"),
			Resources:     cmdutil.GetFlagStringSlice(cmd, "resource"),
			ResourceNames: cmdutil.GetFlagStringArray(cmd, "resource-name"),
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
