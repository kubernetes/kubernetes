/*
Copyright 2016 The Kubernetes Authors.

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

package rbac

import (
	"fmt"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// ResourceMatches returns the result of the rule.Resources matching.
func ResourceMatches(rule *PolicyRule, combinedRequestedResource, requestedSubresource string) bool {
	for _, ruleResource := range rule.Resources {
		// if everything is allowed, we match
		if ruleResource == ResourceAll {
			return true
		}
		// if we have an exact match, we match
		if ruleResource == combinedRequestedResource {
			return true
		}

		// We can also match a */subresource.
		// if there isn't a subresource, then continue
		if len(requestedSubresource) == 0 {
			continue
		}
		// if the rule isn't in the format */subresource, then we don't match, continue
		if len(ruleResource) == len(requestedSubresource)+2 &&
			strings.HasPrefix(ruleResource, "*/") &&
			strings.HasSuffix(ruleResource, requestedSubresource) {
			return true

		}
	}

	return false
}

// SubjectsStrings returns users, groups, serviceaccounts, unknown for display purposes.
func SubjectsStrings(subjects []Subject) ([]string, []string, []string, []string) {
	users := []string{}
	groups := []string{}
	sas := []string{}
	others := []string{}

	for _, subject := range subjects {
		switch subject.Kind {
		case ServiceAccountKind:
			sas = append(sas, fmt.Sprintf("%s/%s", subject.Namespace, subject.Name))

		case UserKind:
			users = append(users, subject.Name)

		case GroupKind:
			groups = append(groups, subject.Name)

		default:
			others = append(others, fmt.Sprintf("%s/%s/%s", subject.Kind, subject.Namespace, subject.Name))
		}
	}

	return users, groups, sas, others
}

func (r PolicyRule) String() string {
	return "PolicyRule" + r.CompactString()
}

// CompactString exposes a compact string representation for use in escalation error messages
func (r PolicyRule) CompactString() string {
	formatStringParts := []string{}
	formatArgs := []interface{}{}
	if len(r.APIGroups) > 0 {
		formatStringParts = append(formatStringParts, "APIGroups:%q")
		formatArgs = append(formatArgs, r.APIGroups)
	}
	if len(r.Resources) > 0 {
		formatStringParts = append(formatStringParts, "Resources:%q")
		formatArgs = append(formatArgs, r.Resources)
	}
	if len(r.NonResourceURLs) > 0 {
		formatStringParts = append(formatStringParts, "NonResourceURLs:%q")
		formatArgs = append(formatArgs, r.NonResourceURLs)
	}
	if len(r.ResourceNames) > 0 {
		formatStringParts = append(formatStringParts, "ResourceNames:%q")
		formatArgs = append(formatArgs, r.ResourceNames)
	}
	if len(r.Verbs) > 0 {
		formatStringParts = append(formatStringParts, "Verbs:%q")
		formatArgs = append(formatArgs, r.Verbs)
	}
	formatString := "{" + strings.Join(formatStringParts, ", ") + "}"
	return fmt.Sprintf(formatString, formatArgs...)
}

// PolicyRuleBuilder let's us attach methods.  A no-no for API types.
// We use it to construct rules in code.  It's more compact than trying to write them
// out in a literal and allows us to perform some basic checking during construction
// +k8s:deepcopy-gen=false
type PolicyRuleBuilder struct {
	PolicyRule PolicyRule
}

// NewRule returns new PolicyRule made by input verbs.
func NewRule(verbs ...string) *PolicyRuleBuilder {
	return &PolicyRuleBuilder{
		PolicyRule: PolicyRule{Verbs: sets.NewString(verbs...).List()},
	}
}

// Groups combines the PolicyRule.APIGroups and input groups.
func (r *PolicyRuleBuilder) Groups(groups ...string) *PolicyRuleBuilder {
	r.PolicyRule.APIGroups = combine(r.PolicyRule.APIGroups, groups)
	return r
}

// Resources combines the PolicyRule.Rule and input resources.
func (r *PolicyRuleBuilder) Resources(resources ...string) *PolicyRuleBuilder {
	r.PolicyRule.Resources = combine(r.PolicyRule.Resources, resources)
	return r
}

// Names combines the PolicyRule.ResourceNames and input names.
func (r *PolicyRuleBuilder) Names(names ...string) *PolicyRuleBuilder {
	r.PolicyRule.ResourceNames = combine(r.PolicyRule.ResourceNames, names)
	return r
}

// URLs combines the PolicyRule.NonResourceURLs and input urls.
func (r *PolicyRuleBuilder) URLs(urls ...string) *PolicyRuleBuilder {
	r.PolicyRule.NonResourceURLs = combine(r.PolicyRule.NonResourceURLs, urls)
	return r
}

// RuleOrDie calls the binding method and panics if there is an error.
func (r *PolicyRuleBuilder) RuleOrDie() PolicyRule {
	ret, err := r.Rule()
	if err != nil {
		panic(err)
	}
	return ret
}

func combine(s1, s2 []string) []string {
	s := sets.NewString(s1...)
	s.Insert(s2...)
	return s.List()
}

// Rule returns PolicyRule and error.
func (r *PolicyRuleBuilder) Rule() (PolicyRule, error) {
	if len(r.PolicyRule.Verbs) == 0 {
		return PolicyRule{}, fmt.Errorf("verbs are required: %#v", r.PolicyRule)
	}

	switch {
	case len(r.PolicyRule.NonResourceURLs) > 0:
		if len(r.PolicyRule.APIGroups) != 0 || len(r.PolicyRule.Resources) != 0 || len(r.PolicyRule.ResourceNames) != 0 {
			return PolicyRule{}, fmt.Errorf("non-resource rule may not have apiGroups, resources, or resourceNames: %#v", r.PolicyRule)
		}
	case len(r.PolicyRule.Resources) > 0:
		// resource rule may not have nonResourceURLs

		if len(r.PolicyRule.APIGroups) == 0 {
			// this a common bug
			return PolicyRule{}, fmt.Errorf("resource rule must have apiGroups: %#v", r.PolicyRule)
		}
		// if resource names are set, then the verb must not be list, watch, create, or deletecollection
		// since verbs are largely opaque, we don't want to accidentally prevent things like "impersonate", so
		// we will backlist common mistakes, not whitelist acceptable options.
		if len(r.PolicyRule.ResourceNames) != 0 {
			illegalVerbs := []string{}
			for _, verb := range r.PolicyRule.Verbs {
				switch verb {
				case "list", "watch", "create", "deletecollection":
					illegalVerbs = append(illegalVerbs, verb)
				}
			}
			if len(illegalVerbs) > 0 {
				return PolicyRule{}, fmt.Errorf("verbs %v do not have names available: %#v", illegalVerbs, r.PolicyRule)
			}
		}

	default:
		return PolicyRule{}, fmt.Errorf("a rule must have either nonResourceURLs or resources: %#v", r.PolicyRule)
	}

	return r.PolicyRule, nil
}

// ClusterRoleBindingBuilder let's us attach methods.  A no-no for API types.
// We use it to construct bindings in code.  It's more compact than trying to write them
// out in a literal.
// +k8s:deepcopy-gen=false
type ClusterRoleBindingBuilder struct {
	ClusterRoleBinding ClusterRoleBinding
}

// NewClusterBinding creates a ClusterRoleBinding builder that can be used
// to define the subjects of a cluster role binding. At least one of
// the `Groups`, `Users` or `SAs` method must be called before
// calling the `Binding*` methods.
func NewClusterBinding(clusterRoleName string) *ClusterRoleBindingBuilder {
	return &ClusterRoleBindingBuilder{
		ClusterRoleBinding: ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: clusterRoleName},
			RoleRef: RoleRef{
				APIGroup: GroupName,
				Kind:     "ClusterRole",
				Name:     clusterRoleName,
			},
		},
	}
}

// Groups adds the specified groups as the subjects of the ClusterRoleBinding.
func (r *ClusterRoleBindingBuilder) Groups(groups ...string) *ClusterRoleBindingBuilder {
	for _, group := range groups {
		r.ClusterRoleBinding.Subjects = append(r.ClusterRoleBinding.Subjects, Subject{Kind: GroupKind, APIGroup: GroupName, Name: group})
	}
	return r
}

// Users adds the specified users as the subjects of the ClusterRoleBinding.
func (r *ClusterRoleBindingBuilder) Users(users ...string) *ClusterRoleBindingBuilder {
	for _, user := range users {
		r.ClusterRoleBinding.Subjects = append(r.ClusterRoleBinding.Subjects, Subject{Kind: UserKind, APIGroup: GroupName, Name: user})
	}
	return r
}

// SAs adds the specified sas as the subjects of the ClusterRoleBinding.
func (r *ClusterRoleBindingBuilder) SAs(namespace string, serviceAccountNames ...string) *ClusterRoleBindingBuilder {
	for _, saName := range serviceAccountNames {
		r.ClusterRoleBinding.Subjects = append(r.ClusterRoleBinding.Subjects, Subject{Kind: ServiceAccountKind, Namespace: namespace, Name: saName})
	}
	return r
}

// BindingOrDie calls the binding method and panics if there is an error.
func (r *ClusterRoleBindingBuilder) BindingOrDie() ClusterRoleBinding {
	ret, err := r.Binding()
	if err != nil {
		panic(err)
	}
	return ret
}

// Binding builds and returns the ClusterRoleBinding API object from the builder
// object.
func (r *ClusterRoleBindingBuilder) Binding() (ClusterRoleBinding, error) {
	if len(r.ClusterRoleBinding.Subjects) == 0 {
		return ClusterRoleBinding{}, fmt.Errorf("subjects are required: %#v", r.ClusterRoleBinding)
	}

	return r.ClusterRoleBinding, nil
}

// RoleBindingBuilder let's us attach methods. It is similar to
// ClusterRoleBindingBuilder above.
// +k8s:deepcopy-gen=false
type RoleBindingBuilder struct {
	RoleBinding RoleBinding
}

// NewRoleBinding creates a RoleBinding builder that can be used
// to define the subjects of a role binding. At least one of
// the `Groups`, `Users` or `SAs` method must be called before
// calling the `Binding*` methods.
func NewRoleBinding(roleName, namespace string) *RoleBindingBuilder {
	return &RoleBindingBuilder{
		RoleBinding: RoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:      roleName,
				Namespace: namespace,
			},
			RoleRef: RoleRef{
				APIGroup: GroupName,
				Kind:     "Role",
				Name:     roleName,
			},
		},
	}
}

// NewRoleBindingForClusterRole creates a RoleBinding builder that can be used
// to define the subjects of a cluster role binding. At least one of
// the `Groups`, `Users` or `SAs` method must be called before
// calling the `Binding*` methods.
func NewRoleBindingForClusterRole(roleName, namespace string) *RoleBindingBuilder {
	return &RoleBindingBuilder{
		RoleBinding: RoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:      roleName,
				Namespace: namespace,
			},
			RoleRef: RoleRef{
				APIGroup: GroupName,
				Kind:     "ClusterRole",
				Name:     roleName,
			},
		},
	}
}

// Groups adds the specified groups as the subjects of the RoleBinding.
func (r *RoleBindingBuilder) Groups(groups ...string) *RoleBindingBuilder {
	for _, group := range groups {
		r.RoleBinding.Subjects = append(r.RoleBinding.Subjects, Subject{Kind: GroupKind, APIGroup: GroupName, Name: group})
	}
	return r
}

// Users adds the specified users as the subjects of the RoleBinding.
func (r *RoleBindingBuilder) Users(users ...string) *RoleBindingBuilder {
	for _, user := range users {
		r.RoleBinding.Subjects = append(r.RoleBinding.Subjects, Subject{Kind: UserKind, APIGroup: GroupName, Name: user})
	}
	return r
}

// SAs adds the specified service accounts as the subjects of the
// RoleBinding.
func (r *RoleBindingBuilder) SAs(namespace string, serviceAccountNames ...string) *RoleBindingBuilder {
	for _, saName := range serviceAccountNames {
		r.RoleBinding.Subjects = append(r.RoleBinding.Subjects, Subject{Kind: ServiceAccountKind, Namespace: namespace, Name: saName})
	}
	return r
}

// BindingOrDie calls the binding method and panics if there is an error.
func (r *RoleBindingBuilder) BindingOrDie() RoleBinding {
	ret, err := r.Binding()
	if err != nil {
		panic(err)
	}
	return ret
}

// Binding builds and returns the RoleBinding API object from the builder
// object.
func (r *RoleBindingBuilder) Binding() (RoleBinding, error) {
	if len(r.RoleBinding.Subjects) == 0 {
		return RoleBinding{}, fmt.Errorf("subjects are required: %#v", r.RoleBinding)
	}

	return r.RoleBinding, nil
}

// SortableRuleSlice is the slice of PolicyRule.
type SortableRuleSlice []PolicyRule

func (s SortableRuleSlice) Len() int      { return len(s) }
func (s SortableRuleSlice) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s SortableRuleSlice) Less(i, j int) bool {
	return strings.Compare(s[i].String(), s[j].String()) < 0
}
