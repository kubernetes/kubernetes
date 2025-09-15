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

package create

import (
	"context"
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	clusterRoleLong = templates.LongDesc(i18n.T(`
		Create a cluster role.`))

	clusterRoleExample = templates.Examples(i18n.T(`
		# Create a cluster role named "pod-reader" that allows user to perform "get", "watch" and "list" on pods
		kubectl create clusterrole pod-reader --verb=get,list,watch --resource=pods

		# Create a cluster role named "pod-reader" with ResourceName specified
		kubectl create clusterrole pod-reader --verb=get --resource=pods --resource-name=readablepod --resource-name=anotherpod

		# Create a cluster role named "foo" with API Group specified
		kubectl create clusterrole foo --verb=get,list,watch --resource=rs.apps

		# Create a cluster role named "foo" with SubResource specified
		kubectl create clusterrole foo --verb=get,list,watch --resource=pods,pods/status

		# Create a cluster role name "foo" with NonResourceURL specified
		kubectl create clusterrole "foo" --verb=get --non-resource-url=/logs/*

		# Create a cluster role name "monitoring" with AggregationRule specified
		kubectl create clusterrole monitoring --aggregation-rule="rbac.example.com/aggregate-to-monitoring=true"`))

	// Valid nonResource verb list for validation.
	validNonResourceVerbs = []string{"*", "get", "post", "put", "delete", "patch", "head", "options"}
)

// CreateClusterRoleOptions is returned by NewCmdCreateClusterRole
type CreateClusterRoleOptions struct {
	*CreateRoleOptions
	NonResourceURLs []string
	AggregationRule map[string]string
	FieldManager    string
}

// NewCmdCreateClusterRole initializes and returns new ClusterRoles command
func NewCmdCreateClusterRole(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	c := &CreateClusterRoleOptions{
		CreateRoleOptions: NewCreateRoleOptions(ioStreams),
		AggregationRule:   map[string]string{},
	}
	cmd := &cobra.Command{
		Use:                   "clusterrole NAME --verb=verb --resource=resource.group [--resource-name=resourcename] [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a cluster role"),
		Long:                  clusterRoleLong,
		Example:               clusterRoleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(c.Complete(f, cmd, args))
			cmdutil.CheckErr(c.Validate())
			cmdutil.CheckErr(c.RunCreateRole())
		},
	}

	c.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringSliceVar(&c.Verbs, "verb", c.Verbs, "Verb that applies to the resources contained in the rule")
	cmd.Flags().StringSliceVar(&c.NonResourceURLs, "non-resource-url", c.NonResourceURLs, "A partial url that user should have access to.")
	cmd.Flags().StringSlice("resource", []string{}, "Resource that the rule applies to")
	cmd.Flags().StringArrayVar(&c.ResourceNames, "resource-name", c.ResourceNames, "Resource in the white list that the rule applies to, repeat this flag for multiple items")
	cmd.Flags().Var(cliflag.NewMapStringString(&c.AggregationRule), "aggregation-rule", "An aggregation label selector for combining ClusterRoles.")
	cmdutil.AddFieldManagerFlagVar(cmd, &c.FieldManager, "kubectl-create")

	return cmd
}

// Complete completes all the required options
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

// Validate makes sure there is no discrepency in CreateClusterRoleOptions
func (c *CreateClusterRoleOptions) Validate() error {
	if c.Name == "" {
		return fmt.Errorf("name must be specified")
	}

	if len(c.AggregationRule) > 0 {
		if len(c.NonResourceURLs) > 0 || len(c.Verbs) > 0 || len(c.Resources) > 0 || len(c.ResourceNames) > 0 {
			return fmt.Errorf("aggregation rule must be specified without nonResourceURLs, verbs, resources or resourceNames")
		}
		return nil
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
				fmt.Fprintf(c.ErrOut, "Warning: '%s' is not a standard resource verb\n", v)
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

		for _, nonResourceURL := range c.NonResourceURLs {
			if nonResourceURL == "*" {
				continue
			}

			if nonResourceURL == "" || !strings.HasPrefix(nonResourceURL, "/") {
				return fmt.Errorf("nonResourceURL should start with /")
			}

			if strings.ContainsRune(nonResourceURL[:len(nonResourceURL)-1], '*') {
				return fmt.Errorf("nonResourceURL only supports wildcard matches when '*' is at the end")
			}
		}
	}

	return nil

}

// RunCreateRole creates a new clusterRole
func (c *CreateClusterRoleOptions) RunCreateRole() error {
	clusterRole := &rbacv1.ClusterRole{
		// this is ok because we know exactly how we want to be serialized
		TypeMeta: metav1.TypeMeta{APIVersion: rbacv1.SchemeGroupVersion.String(), Kind: "ClusterRole"},
	}
	clusterRole.Name = c.Name

	var err error
	if len(c.AggregationRule) == 0 {
		rules, err := generateResourcePolicyRules(c.Mapper, c.Verbs, c.Resources, c.ResourceNames, c.NonResourceURLs)
		if err != nil {
			return err
		}
		clusterRole.Rules = rules
	} else {
		clusterRole.AggregationRule = &rbacv1.AggregationRule{
			ClusterRoleSelectors: []metav1.LabelSelector{
				{
					MatchLabels: c.AggregationRule,
				},
			},
		}
	}

	if err := util.CreateOrUpdateAnnotation(c.CreateAnnotation, clusterRole, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	// Create ClusterRole.
	if c.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if c.FieldManager != "" {
			createOptions.FieldManager = c.FieldManager
		}
		createOptions.FieldValidation = c.ValidationDirective
		if c.DryRunStrategy == cmdutil.DryRunServer {
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		clusterRole, err = c.Client.ClusterRoles().Create(context.TODO(), clusterRole, createOptions)
		if err != nil {
			return err
		}
	}

	return c.PrintObj(clusterRole)
}
