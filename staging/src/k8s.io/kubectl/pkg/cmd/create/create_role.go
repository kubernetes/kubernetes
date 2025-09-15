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
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	clientgorbacv1 "k8s.io/client-go/kubernetes/typed/rbac/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	roleLong = templates.LongDesc(i18n.T(`
		Create a role with single rule.`))

	roleExample = templates.Examples(i18n.T(`
		# Create a role named "pod-reader" that allows user to perform "get", "watch" and "list" on pods
		kubectl create role pod-reader --verb=get --verb=list --verb=watch --resource=pods

		# Create a role named "pod-reader" with ResourceName specified
		kubectl create role pod-reader --verb=get --resource=pods --resource-name=readablepod --resource-name=anotherpod

		# Create a role named "foo" with API Group specified
		kubectl create role foo --verb=get,list,watch --resource=rs.apps

		# Create a role named "foo" with SubResource specified
		kubectl create role foo --verb=get,list,watch --resource=pods,pods/status`))

	// Valid resource verb list for validation.
	validResourceVerbs = []string{"*", "get", "delete", "list", "create", "update", "patch", "watch", "proxy", "deletecollection", "use", "bind", "escalate", "impersonate"}

	// Specialized verbs and GroupResources
	specialVerbs = map[string][]schema.GroupResource{
		"use": {
			{
				Group:    "policy",
				Resource: "podsecuritypolicies",
			},
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
		"escalate": {
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

// AddSpecialVerb allows the addition of items to the `specialVerbs` map for non-k8s native resources.
func AddSpecialVerb(verb string, gr schema.GroupResource) {
	resources, ok := specialVerbs[verb]
	if !ok {
		resources = make([]schema.GroupResource, 0, 1)
	}
	resources = append(resources, gr)
	specialVerbs[verb] = resources
}

// ResourceOptions holds the related options for '--resource' option
type ResourceOptions struct {
	Group       string
	Resource    string
	SubResource string
}

// CreateRoleOptions holds the options for 'create role' sub command
type CreateRoleOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	Name          string
	Verbs         []string
	Resources     []ResourceOptions
	ResourceNames []string

	DryRunStrategy      cmdutil.DryRunStrategy
	ValidationDirective string
	OutputFormat        string
	Namespace           string
	EnforceNamespace    bool
	Client              clientgorbacv1.RbacV1Interface
	Mapper              meta.RESTMapper
	PrintObj            func(obj runtime.Object) error
	FieldManager        string
	CreateAnnotation    bool

	genericiooptions.IOStreams
}

// NewCreateRoleOptions returns an initialized CreateRoleOptions instance
func NewCreateRoleOptions(ioStreams genericiooptions.IOStreams) *CreateRoleOptions {
	return &CreateRoleOptions{
		PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),

		IOStreams: ioStreams,
	}
}

// NewCmdCreateRole returnns an initialized Command instance for 'create role' sub command
func NewCmdCreateRole(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewCreateRoleOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "role NAME --verb=verb --resource=resource.group/subresource [--resource-name=resourcename] [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Create a role with single rule"),
		Long:                  roleLong,
		Example:               roleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunCreateRole())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringSliceVar(&o.Verbs, "verb", o.Verbs, "Verb that applies to the resources contained in the rule")
	cmd.Flags().StringSlice("resource", []string{}, "Resource that the rule applies to")
	cmd.Flags().StringArrayVar(&o.ResourceNames, "resource-name", o.ResourceNames, "Resource in the white list that the rule applies to, repeat this flag for multiple items")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")
	return cmd
}

// Complete completes all the required options
func (o *CreateRoleOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	o.Name = name

	// Remove duplicate verbs.
	verbs := []string{}
	for _, v := range o.Verbs {
		// VerbAll respresents all kinds of verbs.
		if v == "*" {
			verbs = []string{"*"}
			break
		}
		if !arrayContains(verbs, v) {
			verbs = append(verbs, v)
		}
	}
	o.Verbs = verbs

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

		if resource.Resource == "*" && len(parts) == 1 && len(sections) == 1 {
			o.Resources = []ResourceOptions{*resource}
			break
		}

		o.Resources = append(o.Resources, *resource)
	}

	// Remove duplicate resource names.
	resourceNames := []string{}
	for _, n := range o.ResourceNames {
		if !arrayContains(resourceNames, n) {
			resourceNames = append(resourceNames, n)
		}
	}
	o.ResourceNames = resourceNames

	// Complete other options for Run.
	o.Mapper, err = f.ToRESTMapper()
	if err != nil {
		return err
	}

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	o.OutputFormat = cmdutil.GetFlagString(cmd, "output")
	o.CreateAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)

	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	o.ValidationDirective, err = cmdutil.GetValidationDirective(cmd)
	if err != nil {
		return err
	}

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.Client = clientset.RbacV1()

	return nil
}

// Validate makes sure there is no discrepency in provided option values
func (o *CreateRoleOptions) Validate() error {
	if o.Name == "" {
		return fmt.Errorf("name must be specified")
	}

	// validate verbs.
	if len(o.Verbs) == 0 {
		return fmt.Errorf("at least one verb must be specified")
	}

	for _, v := range o.Verbs {
		if !arrayContains(validResourceVerbs, v) {
			fmt.Fprintf(o.ErrOut, "Warning: '%s' is not a standard resource verb\n", v)
		}
	}

	// validate resources.
	if len(o.Resources) == 0 {
		return fmt.Errorf("at least one resource must be specified")
	}

	return o.validateResource()
}

func (o *CreateRoleOptions) validateResource() error {
	for _, r := range o.Resources {
		if len(r.Resource) == 0 {
			return fmt.Errorf("resource must be specified if apiGroup/subresource specified")
		}
		if r.Resource == "*" {
			return nil
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

// RunCreateRole performs the execution of 'create role' sub command
func (o *CreateRoleOptions) RunCreateRole() error {
	role := &rbacv1.Role{
		// this is ok because we know exactly how we want to be serialized
		TypeMeta: metav1.TypeMeta{APIVersion: rbacv1.SchemeGroupVersion.String(), Kind: "Role"},
	}
	role.Name = o.Name
	rules, err := generateResourcePolicyRules(o.Mapper, o.Verbs, o.Resources, o.ResourceNames, []string{})
	if err != nil {
		return err
	}
	role.Rules = rules
	if o.EnforceNamespace {
		role.Namespace = o.Namespace
	}

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, role, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	// Create role.
	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		createOptions.FieldValidation = o.ValidationDirective
		if o.DryRunStrategy == cmdutil.DryRunServer {
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		role, err = o.Client.Roles(o.Namespace).Create(context.TODO(), role, createOptions)
		if err != nil {
			return err
		}
	}

	return o.PrintObj(role)
}

func arrayContains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

func generateResourcePolicyRules(mapper meta.RESTMapper, verbs []string, resources []ResourceOptions, resourceNames []string, nonResourceURLs []string) ([]rbacv1.PolicyRule, error) {
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
	rules := []rbacv1.PolicyRule{}
	for _, g := range sets.List(sets.KeySet(groupResourceMapping)) {
		rule := rbacv1.PolicyRule{}
		rule.Verbs = verbs
		rule.Resources = groupResourceMapping[g]
		rule.APIGroups = []string{g}
		rule.ResourceNames = resourceNames
		rules = append(rules, rule)
	}

	if len(nonResourceURLs) > 0 {
		rule := rbacv1.PolicyRule{}
		rule.Verbs = verbs
		rule.NonResourceURLs = nonResourceURLs
		rules = append(rules, rule)
	}

	return rules, nil
}
