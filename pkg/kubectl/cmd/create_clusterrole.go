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

	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	clusterRoleLong = templates.LongDesc(i18n.T(`
		Create a ClusterRole.`))

	clusterRoleExample = templates.Examples(i18n.T(`
		# Create a ClusterRole named "pod-reader" that allows user to perform "get", "watch" and "list" on pods
		kubectl create clusterrole pod-reader --verb=get,list,watch --resource=pods

		# Create a ClusterRole named "pod-reader" with ResourceName specified
		kubectl create clusterrole pod-reader --verb=get,list,watch --resource=pods --resource-name=readablepod --resource-name=anotherpod

		# Create a ClusterRole named "foo" with API Group specified
		kubectl create clusterrole foo --verb=get,list,watch --resource=rs.extensions

		# Create a ClusterRole named "foo" with SubResource specified
		kubectl create clusterrole foo --verb=get,list,watch --resource=pods,pods/status

		# Create a ClusterRole name "foo" with NonResourceURL specified
		kubectl create clusterrole "foo" --verb=get --non-resource-url=/logs/*`))

	// Valid nonResource verb list for validation.
	validNonResourceVerbs = []string{"*", "get", "post", "put", "delete", "patch", "head", "options"}
)

type CreateClusterRoleOptions struct {
	*CreateRoleOptions
	NonResourceURLs []string
}

// ClusterRole is a command to ease creating ClusterRoles.
func NewCmdCreateClusterRole(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	c := &CreateClusterRoleOptions{
		CreateRoleOptions: &CreateRoleOptions{
			Out: cmdOut,
		},
	}
	cmd := &cobra.Command{
		Use:     "clusterrole NAME --verb=verb --resource=resource.group [--resource-name=resourcename] [--dry-run]",
		Short:   clusterRoleLong,
		Long:    clusterRoleLong,
		Example: clusterRoleExample,
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
	cmd.Flags().StringSliceVar(&c.NonResourceURLs, "non-resource-url", []string{}, "a partial url that user should have access to.")
	cmd.Flags().StringSlice("resource", []string{}, "resource that the rule applies to")
	cmd.Flags().StringArrayVar(&c.ResourceNames, "resource-name", []string{}, "resource in the white list that the rule applies to, repeat this flag for multiple items")

	return cmd
}

func (c *CreateClusterRoleOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	// Remove duplicate nonResourceURLs
	nonResourceURLs := []string{}
	for _, n := range c.NonResourceURLs {
		if !arrayContains(nonResourceURLs, n) {
			nonResourceURLs = append(nonResourceURLs, n)
		}
	}
	c.NonResourceURLs = nonResourceURLs

	return c.CreateRoleOptions.Complete(f, cmd, args)
}

func (c *CreateClusterRoleOptions) Validate() error {
	if c.Name == "" {
		return fmt.Errorf("name must be specified")
	}

	// validate verbs.
	if len(c.Verbs) == 0 {
		return fmt.Errorf("at least one verb must be specified")
	}

	if len(c.Resources) == 0 && len(c.NonResourceURLs) == 0 {
		return fmt.Errorf("one of resource or nonResourceURL must be specified")
	}

	// validate resources
	if len(c.Resources) > 0 {
		for _, v := range c.Verbs {
			if !arrayContains(validResourceVerbs, v) {
				return fmt.Errorf("invalid verb: '%s'", v)
			}
		}
		if err := c.validateResource(); err != nil {
			return err
		}
	}

	//validate non-resource-url
	if len(c.NonResourceURLs) > 0 {
		for _, v := range c.Verbs {
			if !arrayContains(validNonResourceVerbs, v) {
				return fmt.Errorf("invalid verb: '%s' for nonResourceURL", v)
			}
		}
	}

	return nil

}

func (c *CreateClusterRoleOptions) RunCreateRole() error {
	clusterRole := &rbac.ClusterRole{}
	clusterRole.Name = c.Name
	rules, err := generateResourcePolicyRules(c.Mapper, c.Verbs, c.Resources, c.ResourceNames, c.NonResourceURLs)
	if err != nil {
		return err
	}
	clusterRole.Rules = rules

	// Create ClusterRole.
	if !c.DryRun {
		_, err = c.Client.ClusterRoles().Create(clusterRole)
		if err != nil {
			return err
		}
	}

	if useShortOutput := c.OutputFormat == "name"; useShortOutput || len(c.OutputFormat) == 0 {
		cmdutil.PrintSuccess(c.Mapper, useShortOutput, c.Out, "clusterroles", c.Name, c.DryRun, "created")
		return nil
	}

	return c.PrintObject(clusterRole)
}
