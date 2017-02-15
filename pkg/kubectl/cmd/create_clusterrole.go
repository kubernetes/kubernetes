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
	clusterRoleLong = templates.LongDesc(`
		Create a ClusterRole.`)

	clusterRoleExample = templates.Examples(`
		# Create a ClusterRole named "pod-reader" that allows user to perform "get", "watch" and "list" on pods
		kubectl create clusterrole pod-reader --verb=get,list,watch --resource=pods

		# Create a ClusterRole named "pod-reader" with ResourceName specified
		kubectl create clusterrole pod-reader --verb=get,list,watch --resource=pods --resource-name=readablepod

		# Create a ClusterRole named "api-reader" that allows user to perform "get" on endpoint "/api"
		kubectl create clusterrole api-reader --verb=get --non-resource-url=/api`)
)

type CreateClusterRoleOptions struct {
	Name            string
	Verbs           []string
	Resources       []schema.GroupVersionResource
	ResourceNames   []string
	NonResourceURLs []string
}

// ClusterRole is a command to ease creating ClusterRoles.
func NewCmdCreateClusterRole(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	c := &CreateClusterRoleOptions{}
	cmd := &cobra.Command{
		Use:     "clusterrole NAME --verb=verb [--resource=resource.group] [--resource-name=resourcename] [--non-resource-url=nonresourceurl] [--dry-run]",
		Short:   clusterRoleLong,
		Long:    clusterRoleLong,
		Example: clusterRoleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(c.Complete(cmd, args))
			cmdutil.CheckErr(c.Validate(f))
			cmdutil.CheckErr(c.RunCreateClusterRole(f, cmdOut, cmd, args))
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringSliceVar(&c.Verbs, "verb", []string{}, "verb that applies to the resources contained in the rule")
	cmd.Flags().StringSlice("resource", []string{}, "resource that the rule applies to")
	cmd.Flags().StringSliceVar(&c.ResourceNames, "resource-name", []string{}, "resource in the white list that the rule applies to")
	cmd.Flags().StringSliceVar(&c.NonResourceURLs, "non-resource-url", []string{}, "non-resource URL is a partial URL that a user should have access to")

	return cmd
}

func (c *CreateClusterRoleOptions) Complete(cmd *cobra.Command, args []string) error {
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

	// Remove duplicate non-resource URLs.
	nonResourceURLs := []string{}
	for _, n := range c.NonResourceURLs {
		if !arrayContains(nonResourceURLs, n) {
			nonResourceURLs = append(nonResourceURLs, n)
		}
	}
	c.NonResourceURLs = nonResourceURLs

	return nil
}

func (c *CreateClusterRoleOptions) Validate(f cmdutil.Factory) error {
	if c.Name == "" {
		return fmt.Errorf("name must be specified")
	}

	// validate verbs.
	containsResourceVerbs, containsNonResourceVerbs := false, false
	for _, v := range c.Verbs {
		if v == "*" {
			containsResourceVerbs = true
			containsNonResourceVerbs = true
			break
		}

		validVerb := false

		// Since there are overlapping verbs between resource verbs and non resource verbs, need to verify separately.
		if arrayContains(validResourceVerbs, v) {
			containsResourceVerbs = true
			validVerb = true
		}

		if arrayContains(validNonResourceVerbs, v) {
			containsNonResourceVerbs = true
			validVerb = true
		}

		if !validVerb {
			return fmt.Errorf("invalid verb: '%s'", v)
		}
	}

	if !containsNonResourceVerbs && len(c.NonResourceURLs) > 0 {
		return fmt.Errorf("at least one non-resource verb must be specified when non-resource URL(s) provided")
	}

	if !containsResourceVerbs && len(c.Resources) > 0 {
		return fmt.Errorf("at least one resource verb must be specified when resource(s) provided")
	}

	// validate resources.
	mapper, _ := f.Object()
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

	// validate resource names, at least one resource must be specified if resource name(s) provided.
	if len(c.ResourceNames) > 0 && len(c.Resources) == 0 {
		return fmt.Errorf("at least one resource must be specified when resource name(s) provided")
	}

	if len(c.Resources) == 0 && len(c.NonResourceURLs) == 0 {
		return fmt.Errorf("at least one resource or non-resource URL must be specified")
	}

	return nil
}

func (c *CreateClusterRoleOptions) RunCreateClusterRole(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
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

	// Split verbs to resource verbs and non-resource verbs
	resourceVerbs, nonResourceVerbs := []string{}, []string{}
	for _, v := range c.Verbs {
		if v == "*" {
			resourceVerbs = []string{"*"}
			nonResourceVerbs = []string{"*"}
			break
		}

		if arrayContains(validResourceVerbs, v) {
			resourceVerbs = append(resourceVerbs, v)
		}

		if arrayContains(validNonResourceVerbs, v) {
			nonResourceVerbs = append(nonResourceVerbs, v)
		}
	}

	clusterRole := &rbac.ClusterRole{}

	// Create separate rule for each of the api group.
	rules := []rbac.PolicyRule{}
	for _, g := range sets.StringKeySet(groupResourceMapping).List() {
		rule := rbac.PolicyRule{}
		rule.Verbs = resourceVerbs
		rule.Resources = groupResourceMapping[g]
		rule.APIGroups = []string{g}
		rule.ResourceNames = c.ResourceNames
		rules = append(rules, rule)
	}

	// Create separate rule for non-resource URLs.
	if len(c.NonResourceURLs) != 0 {
		rule := rbac.PolicyRule{}
		rule.Verbs = nonResourceVerbs
		rule.NonResourceURLs = c.NonResourceURLs
		rules = append(rules, rule)
	}

	clusterRole.Name = c.Name
	clusterRole.Rules = rules

	// Create ClusterRole.
	if !dryRun {
		client, err := f.ClientSet()
		if err != nil {
			return err
		}
		_, err = client.Rbac().ClusterRoles().Create(clusterRole)
		if err != nil {
			return err
		}
	}

	if useShortOutput := outputFormat == "name"; useShortOutput || len(outputFormat) == 0 {
		cmdutil.PrintSuccess(mapper, useShortOutput, cmdOut, "clusterroles", c.Name, dryRun, "created")
		return nil
	}

	return f.PrintObject(cmd, mapper, clusterRole, cmdOut)
}
