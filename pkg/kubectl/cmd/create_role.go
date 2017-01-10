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
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

var (
	roleLong = templates.LongDesc(`
		Create a role with single rule.`)

	roleExample = templates.Examples(`
		# Create a Role named "pod-reader" that allows user to perform "get", "watch" and "list" on pods
		kubectl create role pod-reader --verb=get --verb=list --verb=watch --resource=pods

		# Create a Role named "pod-reader" with ResourceName specified
		kubectl create role pod-reader --verb=get --verg=list --verb=watch --resource=pods --resource-name=readablepod`)
)

type CreateRoleOptions struct {
	Name          string
	Verbs         []string
	Resources     []string
	APIGroups     []string
	ResourceNames []string
}

// Role is a command to ease creating Roles.
func NewCmdCreateRole(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	c := &CreateRoleOptions{}
	cmd := &cobra.Command{
		Use:     "role NAME --verb=verb --resource=resource.group [--resource-name=resourcename] [--dry-run]",
		Short:   roleLong,
		Long:    roleLong,
		Example: roleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(c.Complete(cmd, args))
			cmdutil.CheckErr(c.RunCreateRole(f, cmdOut, cmd, args))
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.RoleV1GeneratorName)
	cmd.Flags().StringSlice("verb", []string{}, "verb that applies to the resources contained in the rule")
	cmd.Flags().StringSlice("resource", []string{}, "resource that the rule applies to")
	cmd.Flags().StringSlice("resource-name", []string{}, "resource in the white list that the rule applies to")

	return cmd
}

func (c *CreateRoleOptions) Complete(cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	c.Name = name

	// support specify multiple verbs together
	// e.g. --verb=get,watch,list
	verbs := cmdutil.GetFlagStringSlice(cmd, "verb")
	for _, v := range verbs {
		c.Verbs = mergeArrays(c.Verbs, strings.Split(v, ","))
	}

	// support specify multiple resources together
	// e.g. --resource=pods,deployments.extensions
	candidateResources := []string{}
	resources := cmdutil.GetFlagStringSlice(cmd, "resource")
	for _, r := range resources {
		candidateResources = mergeArrays(candidateResources, strings.Split(r, ","))
	}
	for _, r := range candidateResources {
		// support resource.group pattern
		index := strings.Index(r, ".")
		resourceName, apiGroup := "", ""

		// No API Group specified, use "" as core API Group
		if index == -1 {
			resourceName = r
		} else {
			resourceName, apiGroup = r[0:index], r[index+1:]
		}

		if !arrayContains(c.APIGroups, apiGroup) {
			c.APIGroups = append(c.APIGroups, apiGroup)
		}
		if !arrayContains(c.Resources, resourceName) {
			c.Resources = append(c.Resources, resourceName)
		}
	}

	// support specify multiple resource names together
	// e.g. --resource-name=foo,boo
	resourceNames := cmdutil.GetFlagStringSlice(cmd, "resource-name")
	for _, n := range resourceNames {
		c.ResourceNames = mergeArrays(c.ResourceNames, strings.Split(n, ","))
	}

	return nil
}

func (c *CreateRoleOptions) RunCreateRole(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	mapper, typer := f.Object()
	dryRun, outputFormat := cmdutil.GetDryRunFlag(cmd), cmdutil.GetFlagString(cmd, "output")

	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.RoleV1GeneratorName:
		generator = &kubectl.RoleGeneratorV1{
			Name:          c.Name,
			Verbs:         c.Verbs,
			Resources:     c.Resources,
			ResourceNames: c.ResourceNames,
			APIGroups:     c.APIGroups,
			Mapper:        mapper,
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}

	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	obj, err := generator.StructuredGenerate()
	if err != nil {
		return err
	}

	role := obj.(*rbac.Role)

	// Transfer resource name from short form to long form.
	resources := []string{}
	for _, r := range role.Rules[0].Resources {
		resource, err := mapper.ResourceFor(schema.GroupVersionResource{Resource: r})
		if err != nil {
			return err
		}
		resources = append(resources, resource.Resource)
	}
	role.Rules[0].Resources = resources

	gvks, _, err := typer.ObjectKinds(role)
	if err != nil {
		return err
	}
	gvk := gvks[0]
	mapping, err := mapper.RESTMapping(schema.GroupKind{Group: gvk.Group, Kind: gvk.Kind}, gvk.Version)
	if err != nil {
		return err
	}
	client, err := f.ClientForMapping(mapping)
	if err != nil {
		return err
	}
	resourceMapper := &resource.Mapper{
		ObjectTyper:  typer,
		RESTMapper:   mapper,
		ClientMapper: resource.ClientMapperFunc(f.ClientForMapping),
	}
	info, err := resourceMapper.InfoForObject(role, nil)
	if err != nil {
		return err
	}
	if err := kubectl.UpdateApplyAnnotation(info, f.JSONEncoder()); err != nil {
		return err
	}
	if !dryRun {
		_, err = resource.NewHelper(client, mapping).Create(namespace, false, info.Object)
		if err != nil {
			return err
		}
	}

	if useShortOutput := outputFormat == "name"; useShortOutput || len(outputFormat) == 0 {
		cmdutil.PrintSuccess(mapper, useShortOutput, cmdOut, mapping.Resource, c.Name, dryRun, "created")
		return nil
	}

	return f.PrintObject(cmd, mapper, role, cmdOut)
}

// mergeArrays merges two string arrays with no duplicate element.
func mergeArrays(a []string, b []string) []string {
	for _, v := range b {
		if !arrayContains(a, v) {
			a = append(a, v)
		}
	}
	return a
}

func arrayContains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}
