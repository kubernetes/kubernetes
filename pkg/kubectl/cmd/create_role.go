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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	roleLong = templates.LongDesc(`
		Create a role with single rule.`)

	roleExample = templates.Examples(`
		# Create a Role named "pod-reader" that allows user to perform "get", "watch" and "list" on pods
		kubectl create role pod-reader --verb=get --verb=list --verb=watch --resource=pods

		# Create a Role named "pod-reader" with ResourceName specified
		kubectl create role pod-reader --verb=get --verg=list --verb=watch --resource=pods --resource-name=readablepod`)

	// Valid resource verb list for validation.
	validResourceVerbs = []string{"*", "get", "delete", "list", "create", "update", "patch", "watch", "proxy", "redirect", "deletecollection"}
)

type CreateRoleOptions struct {
	Name          string
	Verbs         []string
	Resources     []schema.GroupVersionResource
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
			cmdutil.CheckErr(c.Validate(f))
			cmdutil.CheckErr(c.RunCreateRole(f, cmdOut, cmd, args))
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringSliceVar(&c.Verbs, "verb", []string{}, "verb that applies to the resources contained in the rule")
	cmd.Flags().StringSlice("resource", []string{}, "resource that the rule applies to")
	cmd.Flags().StringSliceVar(&c.ResourceNames, "resource-name", []string{}, "resource in the white list that the rule applies to")

	return cmd
}

func (c *CreateRoleOptions) Complete(cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	c.Name = name

	// Remove duplicate verbs.
	verbs := []string{}
	for _, v := range c.Verbs {
		// VerbAll respresents all kinds of verbs.
		if v == "*" {
			verbs = []string{"*"}
			break
		}
		if !arrayContains(verbs, v) {
			verbs = append(verbs, v)
		}
	}
	c.Verbs = verbs

	// Support resource.group pattern. If no API Group specified, use "" as core API Group.
	// e.g. --resource=pods,deployments.extensions
	resources := cmdutil.GetFlagStringSlice(cmd, "resource")
	for _, r := range resources {
		sections := strings.Split(r, ".")

		if len(sections) == 1 {
			c.Resources = append(c.Resources, schema.GroupVersionResource{Resource: r})
		} else {
			c.Resources = append(c.Resources, schema.GroupVersionResource{Resource: sections[0], Group: strings.Join(sections[1:], ".")})
		}
	}

	// Remove duplicate resource names.
	resourceNames := []string{}
	for _, n := range c.ResourceNames {
		if !arrayContains(resourceNames, n) {
			resourceNames = append(resourceNames, n)
		}
	}
	c.ResourceNames = resourceNames

	return nil
}

func (c *CreateRoleOptions) Validate(f cmdutil.Factory) error {
	if c.Name == "" {
		return fmt.Errorf("name must be specified")
	}

	// validate verbs.
	if len(c.Verbs) == 0 {
		return fmt.Errorf("at least one verb must be specified")
	}

	for _, v := range c.Verbs {
		if !arrayContains(validResourceVerbs, v) {
			return fmt.Errorf("invalid verb: '%s'", v)
		}
	}

	// validate resources.
	mapper, _ := f.Object()

	if len(c.Resources) == 0 {
		return fmt.Errorf("at least one resource must be specified")
	}

	for _, r := range c.Resources {
		_, err := mapper.ResourceFor(r)
		if err != nil {
			return err
		}
	}

	// validate resource names, can not apply resource names to multiple resources.
	if len(c.ResourceNames) > 0 && len(c.Resources) > 1 {
		return fmt.Errorf("resource name(s) can not be applied to multiple resources")
	}

	return nil
}

func (c *CreateRoleOptions) RunCreateRole(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	mapper, _ := f.Object()
	dryRun, outputFormat := cmdutil.GetDryRunFlag(cmd), cmdutil.GetFlagString(cmd, "output")

	// groupResourceMapping is a apigroup-resource map. The key of this map is api group, while the value
	// is a string array of resources under this api group.
	// E.g.  groupResourceMapping = {"extensions": ["replicasets", "deployments"], "batch":["jobs"]}
	groupResourceMapping := map[string][]string{}

	// This loop does the following work:
	// 1. Constructs groupResourceMapping based on input resources.
	// 2. Prevents pointing to non-existent resources.
	// 3. Transfers resource short name to long name. E.g. rs.extensions is transferred to replicasets.extensions
	for _, r := range c.Resources {
		resource, err := mapper.ResourceFor(r)
		if err != nil {
			return err
		}
		if !arrayContains(groupResourceMapping[resource.Group], resource.Resource) {
			groupResourceMapping[resource.Group] = append(groupResourceMapping[resource.Group], resource.Resource)
		}
	}

	role := &rbac.Role{}

	// Create separate rule for each of the api group.
	rules := []rbac.PolicyRule{}
	for _, g := range sets.StringKeySet(groupResourceMapping).List() {
		rule := rbac.PolicyRule{}
		rule.Verbs = c.Verbs
		rule.Resources = groupResourceMapping[g]
		rule.APIGroups = []string{g}
		rule.ResourceNames = c.ResourceNames
		rules = append(rules, rule)
	}
	role.Name = c.Name
	role.Rules = rules

	// Create role.
	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if !dryRun {
		client, err := f.ClientSet()
		if err != nil {
			return err
		}
		_, err = client.Rbac().Roles(namespace).Create(role)
		if err != nil {
			return err
		}
	}

	if useShortOutput := outputFormat == "name"; useShortOutput || len(outputFormat) == 0 {
		cmdutil.PrintSuccess(mapper, useShortOutput, cmdOut, "roles", c.Name, dryRun, "created")
		return nil
	}

	return f.PrintObject(cmd, mapper, role, cmdOut)
}

func arrayContains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}
