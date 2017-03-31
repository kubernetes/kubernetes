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
	"fmt"
	"io"
	"reflect"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdset "k8s.io/kubernetes/pkg/kubectl/cmd/set"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	ruleLong = templates.LongDesc(`
		Remove existing rule form roles.

		Possible resources include (case insensitive): role, clusterrole`)

	ruleExample = templates.Examples(`
		# Remove the rule to role/clusterrole
		kubectl unset rule role foo --resource=rs.extensions --verb=get --verb=list

		# Remove the subresource rule to role/clusterrole
		kubectl unset rule policy role foo --resource=rs.extensions/scale --verb=get,list,delete

		# Remove the non resource rule to clusterrole
		kubectl unset rule clusterrole test --non-resource-url="*" --verb=get,post,put,delete

		# Print the result (in yaml format) of updating role/clusterrole from a local, without hitting the server
		kubectl unset rule -f path/to/file.yaml --resource=pods --verb=get --local -o yaml`)
)

type updatePolicyRule func(existingRules []rbac.PolicyRule, targetRule *rbac.PolicyRule) ([]rbac.PolicyRule, error)

// RuleOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags
type RuleOptions struct {
	*cmdset.RuleOptions
}

func NewCmdRule(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	options := &RuleOptions{
		&cmdset.RuleOptions{
			Out: out,
			Err: errOut,
		},
	}

	cmd := &cobra.Command{
		Use:     "rule (-f FILENAME | TYPE NAME) --verb=verb --resource=resource.group/subresource [--resource-name=resourcename] [--dry-run]",
		Short:   "Unset the Rule on a role/clusterrole",
		Long:    fmt.Sprintf(ruleLong),
		Example: ruleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate(f))
			cmdutil.CheckErr(options.Run(f, removePolicyRule))
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "the resource to remove the rules"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().BoolVar(&options.All, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, unset resources will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringSlice("resource", []string{}, "resource that the rule applies to")
	cmd.Flags().StringSliceVar(&options.Verbs, "verb", []string{}, "verb that applies to the resources/non-resources contained in the rule")
	cmd.Flags().StringSliceVar(&options.ResourceNames, "resource-name", []string{}, "resource in the white list that the rule applies to")
	cmd.Flags().StringSliceVar(&options.NonResourceURLs, "non-resource-url", []string{}, "a set of partial urls that a user should have access to")
	return cmd
}

func removePolicyRule(existingRules []rbac.PolicyRule, targetRule *rbac.PolicyRule) ([]rbac.PolicyRule, error) {
	var err error
	rules := []rbac.PolicyRule{}

	for _, rule := range existingRules {
		// don't change the rule which resource name is mismatch
		if !reflect.DeepEqual(rule.ResourceNames, targetRule.ResourceNames) {
			rules = append(rules, rule)
			continue
		}

		if len(rule.APIGroups) > 0 && len(targetRule.APIGroups) > 0 {
			// If multiple API groups are specified
			apiGroups := []string{}
			for _, group := range rule.APIGroups {
				resources := setIntersect(rule.Resources, targetRule.Resources)
				if group == targetRule.APIGroups[0] && len(resources) > 0 {
					// clone rule to reserve old resource
					holdRule := rule
					holdRule.APIGroups = []string{group}
					holdRule.Resources = setDifference(rule.Resources, resources)
					if len(holdRule.Resources) > 0 {
						rules = append(rules, holdRule)
					}

					// clone rule to reserve modified resource
					modifiedRule := rule
					modifiedRule.APIGroups = []string{group}
					modifiedRule.Resources = resources
					modifiedRule.Verbs, err = setDifferenceVerbs(rule.Verbs, targetRule.Verbs)
					if err != nil {
						return nil, err
					}
					if len(modifiedRule.Verbs) > 0 {
						rules = append(rules, modifiedRule)
					}

				} else {
					apiGroups = append(apiGroups, group)
				}
			}
			rule.APIGroups = apiGroups
			if len(rule.APIGroups) == 0 {
				continue
			}
		}

		if len(rule.NonResourceURLs) > 0 && len(targetRule.NonResourceURLs) > 0 {
			if nonResourceURLs := setIntersect(rule.NonResourceURLs, targetRule.NonResourceURLs); len(nonResourceURLs) > 0 {
				// clone rule to reserve old resource
				holdRule := rule
				holdRule.NonResourceURLs = setDifference(rule.NonResourceURLs, nonResourceURLs)
				if len(holdRule.NonResourceURLs) > 0 {
					rules = append(rules, holdRule)
				}

				// clone rule to reserve modified resource
				modifiedRule := rule
				modifiedRule.NonResourceURLs = nonResourceURLs
				modifiedRule.Verbs, err = setDifferenceVerbs(rule.Verbs, targetRule.Verbs)
				if err != nil {
					return nil, err
				}
				if len(modifiedRule.Verbs) > 0 {
					rules = append(rules, modifiedRule)
				}
				continue
			}
		}

		rules = append(rules, rule)
	}
	return rules, nil
}

func setIntersect(a []string, b []string) []string {
	output := []string{}
	for _, v := range a {
		if contain(b, v) {
			output = append(output, v)
		}
	}
	return output
}

func setDifference(a []string, b []string) []string {
	output := []string{}
	for _, v := range a {
		if !contain(b, v) {
			output = append(output, v)
		}
	}
	return output
}

func setDifferenceVerbs(a []string, b []string) ([]string, error) {
	output := []string{}

	reta := contain(a, "*")
	retb := contain(b, "*")

	if reta && retb {
		return []string{}, nil
	}
	if reta || retb {
		return nil, fmt.Errorf("verb with wildcard is not supported")
	}

	for _, v := range a {
		if !contain(b, v) {
			output = append(output, v)
		}
	}
	return output, nil
}

func contain(slice []string, item string) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}
