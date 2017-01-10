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
	"fmt"
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	roleLong = templates.LongDesc(`
		Create a Role to hold a logic grouping of permissions.`)

	roleExample = templates.Examples(`
		# Create a Role named "pod-reader" that allows user to perform "get", "watch" and "list" on "pods"
		kubectl create role pod-reader --verb=get --verb=list --verb=watch --resource=pods`)
)

// Role is a command to ease creating Roles.
func NewCmdCreateRole(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "role NAME --verb=verb --resource=resource [--api-group=apigroup] [--resource-name=resourcename] [--dry-run]",
		Short:   "Create a Role to hold a logic grouping of permissions",
		Long:    roleLong,
		Example: roleExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateRole(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.RoleV1GeneratorName)
	cmd.Flags().StringSlice("verb", []string{}, "verb that applies to the resources contained in the rule")
	cmd.Flags().StringSlice("resource", []string{}, "resource that the rule applies to")
	cmd.Flags().StringSlice("api-group", []string{}, "name of the APIGroup that contains the resources")
	cmd.Flags().StringSlice("resource-name", []string{}, "resource in the white list that the rule applies to")
	return cmd
}

func CreateRole(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	mapper, _ := f.Object()
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.RoleV1GeneratorName:
		generator = &kubectl.RoleGeneratorV1{
			Name:          name,
			Verbs:         cmdutil.GetFlagStringSlice(cmd, "verb"),
			Resources:     cmdutil.GetFlagStringSlice(cmd, "resource"),
			APIGroups:     cmdutil.GetFlagStringSlice(cmd, "api-group"),
			ResourceNames: cmdutil.GetFlagStringSlice(cmd, "resource-name"),
			Mapper:        mapper,
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}

	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
