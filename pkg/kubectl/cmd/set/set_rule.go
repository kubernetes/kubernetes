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

package set

import (
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/registry/rbac/validation"
)

var (
	ruleLong = templates.LongDesc(`
		Add new rule existing rule of roles.

		Possible resources include (case insensitive): role, clusterrole`)

	ruleExample = templates.Examples(`
		# Add the rule to role/clusterrole
		kubectl set rule role foo --resource=rs.extensions --verb=get --verb=list

		# Add the subresource rule to role/clusterrole
		kubectl set rule policy role foo --resource=rs.extensions/scale --verb=get,list,delete

		# Add the non resource rule to clusterrole
		kubectl set rule clusterrole test --non-resource-url="*" --verb=get,post,put,delete

		# Print the result (in yaml format) of updating role/clusterrole from a local, without hitting the server
		kubectl set rule -f path/to/file.yaml --resource=pods --verb=get --local -o yaml`)

	// Valid resource verb list for validation.
	validResourceVerbs = []string{"*", "get", "delete", "list", "create", "update", "patch", "watch", "proxy", "redirect", "deletecollection", "use", "bind", "impersonate"}

	// Valid non-resource verb list for validation.
	validNonResourceVerbs = []string{"*", "get", "post", "put", "delete"}
)

type ResourceOptions struct {
	Group       string
	Resource    string
	SubResource string
}

type updatePolicyRule func(existingRules []rbac.PolicyRule, targetRule *rbac.PolicyRule) ([]rbac.PolicyRule, error)

// RuleOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags
type RuleOptions struct {
	resource.FilenameOptions

	Mapper      meta.RESTMapper
	Typer       runtime.ObjectTyper
	Infos       []*resource.Info
	Encoder     runtime.Encoder
	Out         io.Writer
	Err         io.Writer
	Selector    string
	ShortOutput bool
	All         bool
	Local       bool
	Cmd         *cobra.Command

	Resources       []ResourceOptions
	ResourceNames   []string
	Verbs           []string
	NonResourceURLs []string

	PrintObject func(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
}

func NewCmdRule(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	options := &RuleOptions{
		Out: out,
		Err: errOut,
	}

	cmd := &cobra.Command{
		Use:     "rule (-f FILENAME | TYPE NAME) --verb=verb --resource=resource.group/subresource [--resource-name=resourcename] [--dry-run]",
		Short:   "Set the Rule to a role/clusterrole",
		Long:    fmt.Sprintf(ruleLong),
		Example: ruleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate(f))
			cmdutil.CheckErr(options.Run(f, addPolicyRule))
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "the resource to update the rules"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().BoolVar(&options.All, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set resources will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringSlice("resource", []string{}, "resource that the rule applies to")
	cmd.Flags().StringSliceVar(&options.Verbs, "verb", []string{}, "verb that applies to the resources/non-resources contained in the rule")
	cmd.Flags().StringSliceVar(&options.ResourceNames, "resource-name", []string{}, "resource in the white list that the rule applies to")
	cmd.Flags().StringSliceVar(&options.NonResourceURLs, "non-resource-url", []string{}, "a set of partial urls that a user should have access to")
	return cmd
}

func (o *RuleOptions) UpdateRuleOptions(cmd *cobra.Command) error {
	// Remove duplicate verbs.
	verbs := []string{}
	for _, v := range o.Verbs {
		// VerbAll respresents all kinds of verbs.
		if v == "*" {
			verbs = []string{"*"}
			break
		}
		if !contain(verbs, v) {
			verbs = append(verbs, v)
		}
	}
	o.Verbs = verbs

	// Support resource.group pattern. If no API Group specified, use "" as core API Group.
	// e.g. --resource=pods,deployments.extensions/scale
	o.Resources = []ResourceOptions{}
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

		o.Resources = append(o.Resources, *resource)
	}

	// Remove duplicate resource names.
	resourceNames := []string{}
	for _, n := range o.ResourceNames {
		if !contain(resourceNames, n) {
			resourceNames = append(resourceNames, n)
		}
	}
	o.ResourceNames = resourceNames

	// Remove duplicate non-resource.
	nonResourceURLs := []string{}
	for _, n := range o.NonResourceURLs {
		if !contain(nonResourceURLs, n) {
			nonResourceURLs = append(nonResourceURLs, n)
		}
	}
	o.NonResourceURLs = nonResourceURLs

	return nil
}

func (o *RuleOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Local = cmdutil.GetFlagBool(cmd, "local")
	o.Mapper, o.Typer = f.Object()
	o.Encoder = f.JSONEncoder()
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.PrintObject = f.PrintObject
	o.Cmd = cmd
	o.UpdateRuleOptions(cmd)

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	builder := resource.NewBuilder(o.Mapper, o.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		Flatten()

	if !o.Local {
		builder = builder.
			SelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, args...).
			Latest()
	}

	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}

	return nil
}

func (o *RuleOptions) Validate(f cmdutil.Factory) error {
	if len(o.Verbs) == 0 {
		return fmt.Errorf("you must specify an update to verbs (in the form of --verb)")
	}
	if len(o.Resources) == 0 && len(o.NonResourceURLs) == 0 {
		return fmt.Errorf("you must specify an update to resources (in the form of --resource/--non-resource-url)")
	}

	// validate verbs.
	for _, v := range o.Verbs {
		if len(o.Resources) > 0 && !contain(validResourceVerbs, v) {
			return fmt.Errorf("invalid verb: '%s'", v)
		}
		if len(o.NonResourceURLs) > 0 && !contain(validNonResourceVerbs, v) {
			return fmt.Errorf("invalid verb: '%s'", v)
		}
	}

	// validate resources.
	mapper, _ := f.Object()
	for _, r := range o.Resources {
		if len(r.Resource) == 0 {
			return fmt.Errorf("resource must be specified if apiGroup/subresource specified")
		}
		if _, err := mapper.ResourceFor(schema.GroupVersionResource{Resource: r.Resource, Group: r.Group}); err != nil {
			return err
		}
	}

	// validate resource names, can not apply resource names to multiple resources.
	if len(o.ResourceNames) > 0 && len(o.Resources) > 1 {
		return fmt.Errorf("resource name(s) can not be applied to multiple resources")
	}

	return nil
}

func (o *RuleOptions) Run(f cmdutil.Factory, fn updatePolicyRule) error {
	allErrs := []error{}
	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		extraRules, err := o.GetNewRules(f)
		if err != nil {
			return nil, err
		}

		// Compute extra rules
		_, extraRules = validation.Covers([]rbac.PolicyRule{}, extraRules)

		for _, rule := range extraRules {
			err := updatePolicyRuleForObject(info.Object, &rule, fn)
			if err != nil {
				return nil, err
			}
		}
		return runtime.Encode(o.Encoder, info.Object)
	})

	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			allErrs = append(allErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}

		//no changes
		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			allErrs = append(allErrs, fmt.Errorf("info: %s %q was not changed\n", info.Mapping.Resource, info.Name))
			continue
		}

		if o.Local || cmdutil.GetDryRunFlag(o.Cmd) {
			fmt.Fprintln(o.Out, "running in local/dry-run mode...")
			return o.PrintObject(o.Cmd, o.Mapper, info.Object, o.Out)
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch rule update to role: %v\n", err))
			continue
		}
		info.Refresh(obj, true)

		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, false, "rules updated")
	}
	return utilerrors.NewAggregate(allErrs)
}

func (o *RuleOptions) GetNewRules(f cmdutil.Factory) ([]rbac.PolicyRule, error) {
	rules := []rbac.PolicyRule{}

	// groupResourceMapping is a apigroup-resource map. The key of this map is api group, while the value
	// is a string array of resources under this api group.
	// E.g.  groupResourceMapping = {"extensions": ["replicasets", "deployments"], "batch":["jobs"]}
	groupResourceMapping := map[string][]string{}

	// This loop does the following work:
	// 1. Constructs groupResourceMapping based on input resources.
	// 2. Prevents pointing to non-existent resources.
	// 3. Transfers resource short name to long name. E.g. rs.extensions is transferred to replicasets.extensions
	mapper, _ := f.Object()
	for _, r := range o.Resources {
		resource, err := mapper.ResourceFor(schema.GroupVersionResource{Resource: r.Resource, Group: r.Group})
		if err != nil {
			return rules, err
		}
		if len(r.SubResource) > 0 {
			resource.Resource = resource.Resource + "/" + r.SubResource
		}
		if !contain(groupResourceMapping[resource.Group], resource.Resource) {
			groupResourceMapping[resource.Group] = append(groupResourceMapping[resource.Group], resource.Resource)
		}
	}

	// Create separate rule for each of the api group.
	for _, g := range sets.StringKeySet(groupResourceMapping).List() {
		rule := rbac.PolicyRule{
			APIGroups:     []string{g},
			Resources:     groupResourceMapping[g],
			Verbs:         o.Verbs,
			ResourceNames: o.ResourceNames,
		}
		rules = append(rules, rule)
	}

	// Create rule for non-resource
	if len(o.NonResourceURLs) > 0 {
		rule := rbac.PolicyRule{
			Verbs:           o.Verbs,
			NonResourceURLs: o.NonResourceURLs,
		}
		rules = append(rules, rule)
	}

	return rules, nil
}

func getPolicyRuleFromObject(obj runtime.Object) []rbac.PolicyRule {
	switch t := obj.(type) {
	case *rbac.Role:
		return t.Rules
	case *rbac.ClusterRole:
		return t.Rules
	}
	return []rbac.PolicyRule{}
}

func updatePolicyRuleForObject(obj runtime.Object, targetRule *rbac.PolicyRule, fn updatePolicyRule) error {
	var err error
	switch t := obj.(type) {
	case *rbac.Role:
		t.Rules, err = fn(splitPolicyRule(t.Rules), targetRule)
		return err
	case *rbac.ClusterRole:
		t.Rules, err = fn(splitPolicyRule(t.Rules), targetRule)
		return err
	default:
		err = fmt.Errorf("setting rules is only supported for Role/ClusterRole")
		return err
	}
}

// split rule by resource name
func splitPolicyRule(originalRules []rbac.PolicyRule) []rbac.PolicyRule {
	rules := []rbac.PolicyRule{}
	for _, rule := range originalRules {
		if len(rule.ResourceNames) <= 1 {
			rules = append(rules, rule)
			continue
		}
		for _, name := range rule.ResourceNames {
			subRule := rule
			subRule.ResourceNames = []string{name}
			rules = append(rules, subRule)
		}
	}
	return rules
}

func addPolicyRule(existingRules []rbac.PolicyRule, targetRule *rbac.PolicyRule) ([]rbac.PolicyRule, error) {
	rules := []rbac.PolicyRule{}
	merged := false

	for _, rule := range existingRules {
		// don't change the rule which resource name is mismatch
		if !reflect.DeepEqual(rule.ResourceNames, targetRule.ResourceNames) {
			rules = append(rules, rule)
			continue
		}

		if !merged && len(rule.APIGroups) > 0 && len(targetRule.APIGroups) > 0 {
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
					modifiedRule.Verbs = setUnionVerbs(rule.Verbs, targetRule.Verbs)
					rules = append(rules, modifiedRule)

					// update target rule resources
					targetRule.Resources = setDifference(targetRule.Resources, resources)
					if len(targetRule.Resources) == 0 {
						merged = true
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

		if !merged && len(rule.NonResourceURLs) > 0 && len(targetRule.NonResourceURLs) > 0 {
			if nonResourceURLs := setIntersect(rule.NonResourceURLs, targetRule.NonResourceURLs); len(nonResourceURLs) > 0 {
				// clone rule to reserve old non-resource
				holdRule := rule
				holdRule.NonResourceURLs = setDifference(rule.NonResourceURLs, nonResourceURLs)
				if len(holdRule.NonResourceURLs) > 0 {
					rules = append(rules, holdRule)
				}

				// clone rule to reserve modified non-resource
				modifiedRule := rule
				modifiedRule.NonResourceURLs = nonResourceURLs
				modifiedRule.Verbs = setUnionVerbs(rule.Verbs, targetRule.Verbs)
				rules = append(rules, modifiedRule)

				// update target rule non-resources
				targetRule.NonResourceURLs = setDifference(targetRule.NonResourceURLs, nonResourceURLs)
				if len(targetRule.Resources) == 0 {
					merged = true
				}
				continue
			}
		}

		rules = append(rules, rule)
	}

	if !merged {
		// if cann't merge with existing rule, add a new item
		rules = append(rules, *targetRule)
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

func setUnionVerbs(a []string, b []string) []string {
	output := a
	for _, v := range b {
		if v == "*" {
			return []string{"*"}
		}
		if !contain(a, v) {
			output = append(output, v)
		}
	}
	return output
}

func contain(slice []string, item string) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}
