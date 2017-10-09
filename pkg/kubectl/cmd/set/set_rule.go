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
	"sort"
	"strings"

	"github.com/spf13/cobra"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
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
		kubectl set rbac-rule role foo --resource=rs.extensions --verb=get --verb=list

		# Add the subresource rule to role/clusterrole
		kubectl set rbac-rule policy role foo --resource=rs.extensions/scale --verb=get,list,delete

		# Add the non resource rule to clusterrole
		kubectl set rbac-rule clusterrole test --non-resource-url="*" --verb=get,post,put,delete

		# Print the result (in yaml format) of updating role/clusterrole from a local, without hitting the server
		kubectl set rbac-rule -f path/to/file.yaml --resource=pods --verb=get --local -o yaml`)

	// Valid resource verb list for validation.
	validResourceVerbs = []string{"*", "get", "delete", "list", "create", "update", "patch", "watch", "proxy", "deletecollection", "use", "bind", "impersonate"}

	// Valid non-resource verb list for validation.
	validNonResourceVerbs = []string{"*", "get", "post", "put", "delete", "patch", "head", "options"}

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
				Resource: "serviceaccounts",
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

type updatePolicyRule func(existingRules []rbacv1.PolicyRule, targetRules []rbacv1.PolicyRule) ([]rbacv1.PolicyRule, error)

// RBACRuleOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags
type RBACRuleOptions struct {
	resource.FilenameOptions

	Mapper      meta.RESTMapper
	Typer       runtime.ObjectTyper
	Infos       []*resource.Info
	Encoder     runtime.Encoder
	Out         io.Writer
	Err         io.Writer
	Selector    string
	Output      string
	ShortOutput bool
	All         bool
	Local       bool
	Cmd         *cobra.Command

	Resources       []ResourceOptions
	ResourceNames   []string
	Verbs           []string
	NonResourceURLs []string

	PrintObject func(mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
}

func NewCmdRBACRule(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	options := &RBACRuleOptions{
		Out: out,
		Err: errOut,
	}

	cmd := &cobra.Command{
		Use:     "rbac-rule (-f FILENAME | TYPE NAME) --verb=verb --resource=resource.group/subresource [--resource-name=resourcename] [--dry-run]",
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

func (o *RBACRuleOptions) UpdateRBACRuleOptions(cmd *cobra.Command) error {
	// Remove duplicate verbs.
	verbs := []string{}
	for _, v := range o.Verbs {
		// VerbAll respresents all kinds of verbs.
		if v == "*" {
			verbs = []string{"*"}
			break
		}
		if !containString(verbs, v) {
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
		if !containString(resourceNames, n) {
			resourceNames = append(resourceNames, n)
		}
	}
	o.ResourceNames = resourceNames

	// Remove duplicate non-resource.
	nonResourceURLs := []string{}
	for _, n := range o.NonResourceURLs {
		if !containString(nonResourceURLs, n) {
			nonResourceURLs = append(nonResourceURLs, n)
		}
	}
	o.NonResourceURLs = nonResourceURLs

	return nil
}

func (o *RBACRuleOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Local = cmdutil.GetFlagBool(cmd, "local")
	o.Mapper, o.Typer = f.Object()
	o.Encoder = f.JSONEncoder()
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.Cmd = cmd
	o.UpdateRBACRuleOptions(cmd)
	o.PrintObject = func(mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
		return f.PrintObject(cmd, o.Local, mapper, obj, out)
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	builder := f.NewBuilder().
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten()

	if !o.Local {
		builder = builder.
			LabelSelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, args...).
			Latest()
	} else {
		// if a --local flag was provided, and a resource was specified in the form
		// <resource>/<name>, fail immediately as --local cannot query the api server
		// for the specified resource.
		if len(args) > 0 {
			return resource.LocalResourceError
		}

		builder = builder.Local(f.ClientForMapping)
	}

	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}

	return nil
}

func (o *RBACRuleOptions) Validate(f cmdutil.Factory) error {
	if len(o.Verbs) == 0 {
		return fmt.Errorf("you must specify an update to verbs (in the form of --verb)")
	}
	if len(o.Resources) == 0 && len(o.NonResourceURLs) == 0 {
		return fmt.Errorf("you must specify an update to resources (in the form of --resource/--non-resource-url)")
	}

	// validate verbs.
	for _, v := range o.Verbs {
		if len(o.Resources) > 0 && !containString(validResourceVerbs, v) {
			return fmt.Errorf("invalid verb: '%s'", v)
		}
		if len(o.NonResourceURLs) > 0 && !containString(validNonResourceVerbs, v) {
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

	return o.validateResource()
}

func (o *RBACRuleOptions) validateResource() error {
	for _, r := range o.Resources {
		if len(r.Resource) == 0 {
			return fmt.Errorf("resource must be specified if apiGroup/subresource specified")
		}

		resource := schema.GroupVersionResource{Resource: r.Resource, Group: r.Group}
		groupVersionResource, err := o.Mapper.ResourceFor(schema.GroupVersionResource{Resource: r.Resource, Group: r.Group})
		if err == nil {
			resource = groupVersionResource
		}

		for _, v := range o.Verbs {
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

func (o *RBACRuleOptions) Run(f cmdutil.Factory, fn updatePolicyRule) error {
	allErrs := []error{}
	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		extraRules, err := o.GetNewRules(f)
		if err != nil {
			return nil, err
		}

		if err := updatePolicyRuleForObject(info.VersionedObject, extraRules, fn); err != nil {
			return nil, err
		}
		return runtime.Encode(o.Encoder, info.VersionedObject)
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
			return o.PrintObject(o.Mapper, info.VersionedObject, o.Out)
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch rule update to role: %v\n", err))
			continue
		}
		info.Refresh(obj, true)

		if len(o.Output) > 0 {
			versionedObject, err := patch.Info.Mapping.ConvertToVersion(obj, patch.Info.Mapping.GroupVersionKind.GroupVersion())
			if err != nil {
				return err
			}
			if err := o.PrintObject(o.Mapper, versionedObject, o.Out); err != nil {
				return err
			}
			continue
		}
		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, false, "rules updated")
	}
	return utilerrors.NewAggregate(allErrs)
}

func (o *RBACRuleOptions) GetNewRules(f cmdutil.Factory) ([]rbacv1.PolicyRule, error) {
	rules := []rbacv1.PolicyRule{}

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
		if !containString(groupResourceMapping[resource.Group], resource.Resource) {
			groupResourceMapping[resource.Group] = append(groupResourceMapping[resource.Group], resource.Resource)
		}
	}

	// Create separate rule for each of the api group.
	for _, g := range sets.StringKeySet(groupResourceMapping).List() {
		rule := rbacv1.PolicyRule{
			APIGroups:     []string{g},
			Resources:     groupResourceMapping[g],
			Verbs:         o.Verbs,
			ResourceNames: o.ResourceNames,
		}
		rules = append(rules, rule)
	}

	// Create rule for non-resource
	if len(o.NonResourceURLs) > 0 {
		rule := rbacv1.PolicyRule{
			Verbs:           o.Verbs,
			NonResourceURLs: o.NonResourceURLs,
		}
		rules = append(rules, rule)
	}

	return rules, nil
}

func updatePolicyRuleForObject(obj runtime.Object, targetRules []rbacv1.PolicyRule, fn updatePolicyRule) error {
	var err error
	switch t := obj.(type) {
	case *rbacv1.Role:
		t.Rules, err = fn(t.Rules, targetRules)
		return err
	case *rbacv1.ClusterRole:
		t.Rules, err = fn(t.Rules, targetRules)
		return err
	default:
		err = fmt.Errorf("setting rules is only supported for Role/ClusterRole")
		return err
	}
}

func addPolicyRule(existingRules []rbacv1.PolicyRule, targetRules []rbacv1.PolicyRule) ([]rbacv1.PolicyRule, error) {
	v1Rules := []rbacv1.PolicyRule{}
	v1Rules = append(existingRules, targetRules...)

	rules, err := convertToInternalRules(v1Rules)
	if err != nil {
		return nil, err
	}
	compactRules, err := validation.CompactRules(rules)
	if err != nil {
		return nil, err
	}
	compacted := compactNonResourceRules(compactRules)

	for i, rule := range compacted {
		p := &compacted[i]
		cleaned := []string{}
		for _, verb := range rule.Verbs {
			if verb == "*" {
				cleaned = []string{"*"}
				break
			}
			if !containString(cleaned, verb) {
				cleaned = append(cleaned, verb)
			}
		}
		p.Verbs = cleaned
	}
	sort.Stable(rbac.SortableRuleSlice(compacted))

	v1Rules, err = convertToV1Rules(compacted)
	if err != nil {
		return nil, err
	}
	return v1Rules, nil
}

func convertToInternalRules(v1Rules []rbacv1.PolicyRule) ([]rbac.PolicyRule, error) {
	rules := []rbac.PolicyRule{}
	for _, v1Rule := range v1Rules {
		rule := rbac.PolicyRule{}
		if err := rbachelper.Convert_v1_PolicyRule_To_rbac_PolicyRule(&v1Rule, &rule, nil); err != nil {
			return nil, err
		}
		rules = append(rules, validation.BreakdownRule(rule)...)
	}
	return rules, nil
}

func convertToV1Rules(internalRules []rbac.PolicyRule) ([]rbacv1.PolicyRule, error) {
	rules := []rbacv1.PolicyRule{}
	for _, internalRule := range internalRules {
		rule := rbacv1.PolicyRule{}
		if err := rbachelper.Convert_rbac_PolicyRule_To_v1_PolicyRule(&internalRule, &rule, nil); err != nil {
			return nil, err
		}
		rules = append(rules, rule)
	}
	return rules, nil
}

func compactNonResourceRules(originalRules []rbac.PolicyRule) []rbac.PolicyRule {
	compactRules := []rbac.PolicyRule{}
	nonResourceRules := map[string]*rbac.PolicyRule{}

	for _, rule := range originalRules {
		if len(rule.NonResourceURLs) == 1 {
			key := rule.NonResourceURLs[0]
			if existingRule, ok := nonResourceRules[key]; ok {
				if existingRule.Verbs == nil {
					existingRule.Verbs = []string{}
				}
				existingRule.Verbs = append(existingRule.Verbs, rule.Verbs...)
			} else {
				nonResourceRules[key] = rule.DeepCopy()
			}
		} else {
			compactRules = append(compactRules, rule)
		}
	}

	for _, nonResourceRule := range nonResourceRules {
		compactRules = append(compactRules, *nonResourceRule)
	}
	return compactRules
}

func containString(slice []string, item string) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}
