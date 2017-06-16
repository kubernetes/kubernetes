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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apis/rbac"
	internalversionrbac "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
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

	// Valid resource verb list for validation.
	validResourceVerbs = []string{"*", "get", "delete", "list", "create", "update", "patch", "watch", "proxy", "redirect", "deletecollection", "use", "bind", "impersonate"}

	// Specialized verbs and GroupResources
	specialVerbs = map[string][]schema.GroupResource{
		"use": {
			{
				Group:    "extensions",
				Resource: "podsecuritypolicies",
			},
		},
		"bind": {
			{
				Group:    "rbac.authorization.k8s.io",
				Resource: "roles",
			},
			{
				Group:    "rbac.authorization.k8s.io",
				Resource: "clusterroles",
			},
		},
		"impersonate": {
			{
				Group:    "",
				Resource: "users",
			},
			{
				Group:    "",
				Resource: "groups",
			},
			{
				Group:    "authentication.k8s.io",
				Resource: "userextras",
			},
		},
	}
)

type ResourceOptions struct {
	Group       string
	Resource    string
	SubResource string
}

type CreateRoleOptions struct {
	Name          string
	Verbs         []string
	Resources     []ResourceOptions
	ResourceNames []string

	DryRun       bool
	OutputFormat string
	Namespace    string
	Client       internalversionrbac.RbacInterface
	Mapper       meta.RESTMapper
	Out          io.Writer
	PrintObject  func(obj runtime.Object) error
}

// Role is a command to ease creating Roles.
func NewCmdCreateRole(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	c := &CreateRoleOptions{
		Out: cmdOut,
	}
	cmd := &cobra.Command{
		Use:     "role NAME --verb=verb --resource=resource.group/subresource [--resource-name=resourcename] [--dry-run]",
		Short:   roleLong,
		Long:    roleLong,
		Example: roleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(c.Complete(f, cmd, args))
			cmdutil.CheckErr(c.Validate())
			cmdutil.CheckErr(c.RunCreateRole())
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringSliceVar(&c.Verbs, "verb", []string{}, "verb that applies to the resources contained in the rule")
	cmd.Flags().StringSlice("resource", []string{}, "resource that the rule applies to")
	cmd.Flags().StringArrayVar(&c.ResourceNames, "resource-name", []string{}, "resource in the white list that the rule applies to, repeat this flag for multiple items")

	return cmd
}

func (c *CreateRoleOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
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
		sections := strings.SplitN(r, "/", 2)

		resource := &ResourceOptions{}
		if len(sections) == 2 {
			resource.SubResource = sections[1]
		}

		parts := strings.SplitN(sections[0], ".", 2)
		if len(parts) == 2 {
			resource.Group = parts[1]
		}
		resource.Resource = parts[0]

		c.Resources = append(c.Resources, *resource)
	}

	// Remove duplicate resource names.
	resourceNames := []string{}
	for _, n := range c.ResourceNames {
		if !arrayContains(resourceNames, n) {
			resourceNames = append(resourceNames, n)
		}
	}
	c.ResourceNames = resourceNames

	// Complete other options for Run.
	c.Mapper, _ = f.Object()

	c.DryRun = cmdutil.GetDryRunFlag(cmd)
	c.OutputFormat = cmdutil.GetFlagString(cmd, "output")

	c.Namespace, _, err = f.DefaultNamespace()
	if err != nil {
		return err
	}

	c.PrintObject = func(obj runtime.Object) error {
		return f.PrintObject(cmd, false, c.Mapper, obj, c.Out)
	}

	clientSet, err := f.ClientSet()
	if err != nil {
		return err
	}
	c.Client = clientSet.Rbac()

	return nil
}

func (c *CreateRoleOptions) Validate() error {
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
	if len(c.Resources) == 0 {
		return fmt.Errorf("at least one resource must be specified")
	}

	return c.validateResource()
}

func (c *CreateRoleOptions) validateResource() error {
	for _, r := range c.Resources {
		if len(r.Resource) == 0 {
			return fmt.Errorf("resource must be specified if apiGroup/subresource specified")
		}

		resource := schema.GroupVersionResource{Resource: r.Resource, Group: r.Group}
		groupVersionResource, err := c.Mapper.ResourceFor(schema.GroupVersionResource{Resource: r.Resource, Group: r.Group})
		if err == nil {
			resource = groupVersionResource
		}

		for _, v := range c.Verbs {
			if groupResources, ok := specialVerbs[v]; ok {
				match := false
				for _, extra := range groupResources {
					if resource.Resource == extra.Resource && resource.Group == extra.Group {
						match = true
						err = nil
						break
					}
				}
				if !match {
					return fmt.Errorf("can not perform '%s' on '%s' in group '%s'", v, resource.Resource, resource.Group)
				}
			}
		}

		if err != nil {
			return err
		}
	}
	return nil
}

func (c *CreateRoleOptions) RunCreateRole() error {
	role := &rbac.Role{}
	role.Name = c.Name
	rules, err := generateResourcePolicyRules(c.Mapper, c.Verbs, c.Resources, c.ResourceNames, []string{})
	if err != nil {
		return err
	}
	role.Rules = rules

	// Create role.
	if !c.DryRun {
		_, err = c.Client.Roles(c.Namespace).Create(role)
		if err != nil {
			return err
		}
	}

	if useShortOutput := c.OutputFormat == "name"; useShortOutput || len(c.OutputFormat) == 0 {
		cmdutil.PrintSuccess(c.Mapper, useShortOutput, c.Out, "roles", c.Name, c.DryRun, "created")
		return nil
	}

	return c.PrintObject(role)
}

func arrayContains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

func generateResourcePolicyRules(mapper meta.RESTMapper, verbs []string, resources []ResourceOptions, resourceNames []string, nonResourceURLs []string) ([]rbac.PolicyRule, error) {
	// groupResourceMapping is a apigroup-resource map. The key of this map is api group, while the value
	// is a string array of resources under this api group.
	// E.g.  groupResourceMapping = {"extensions": ["replicasets", "deployments"], "batch":["jobs"]}
	groupResourceMapping := map[string][]string{}

	// This loop does the following work:
	// 1. Constructs groupResourceMapping based on input resources.
	// 2. Prevents pointing to non-existent resources.
	// 3. Transfers resource short name to long name. E.g. rs.extensions is transferred to replicasets.extensions
	for _, r := range resources {
		resource := schema.GroupVersionResource{Resource: r.Resource, Group: r.Group}
		groupVersionResource, err := mapper.ResourceFor(schema.GroupVersionResource{Resource: r.Resource, Group: r.Group})
		if err == nil {
			resource = groupVersionResource
		}

		if len(r.SubResource) > 0 {
			resource.Resource = resource.Resource + "/" + r.SubResource
		}
		if !arrayContains(groupResourceMapping[resource.Group], resource.Resource) {
			groupResourceMapping[resource.Group] = append(groupResourceMapping[resource.Group], resource.Resource)
		}
	}

	// Create separate rule for each of the api group.
	rules := []rbac.PolicyRule{}
	for _, g := range sets.StringKeySet(groupResourceMapping).List() {
		rule := rbac.PolicyRule{}
		rule.Verbs = verbs
		rule.Resources = groupResourceMapping[g]
		rule.APIGroups = []string{g}
		rule.ResourceNames = resourceNames
		rules = append(rules, rule)
	}

	if len(nonResourceURLs) > 0 {
		rule := rbac.PolicyRule{}
		rule.Verbs = verbs
		rule.NonResourceURLs = nonResourceURLs
		rules = append(rules, rule)
	}

	return rules, nil
}
