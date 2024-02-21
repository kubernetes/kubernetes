/*
Copyright 2024 The Kubernetes Authors.

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

package validating

import (
	"errors"
	"sort"
	"strings"

	"k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/klog/v2"
)

// TypeCheckingContext holds information about the policy being type-checked.
// The context can also provide information about referred types to the caller.
type TypeCheckingContext struct {
	gvks          []schema.GroupVersionKind
	declTypes     []*apiservercel.DeclType
	paramGVK      schema.GroupVersionKind
	paramDeclType *apiservercel.DeclType

	variables []v1.Variable
}

// CreateContext resolves all types and their schemas from a policy definition and creates the context.
func (c *TypeChecker) CreateContext(policy *v1.ValidatingAdmissionPolicy) *TypeCheckingContext {
	ctx := new(TypeCheckingContext)
	allGvks := c.typesToCheck(policy)
	gvks := make([]schema.GroupVersionKind, 0, len(allGvks))
	declTypes := make([]*apiservercel.DeclType, 0, len(allGvks))
	for _, gvk := range allGvks {
		declType, err := c.declType(gvk)
		if err != nil {
			// type checking errors MUST NOT alter the behavior of the policy
			// even if an error occurs.
			if !errors.Is(err, resolver.ErrSchemaNotFound) {
				// Anything except ErrSchemaNotFound is an internal error
				klog.V(2).ErrorS(err, "internal error: schema resolution failure", "gvk", gvk)
			}
			// skip for not found or internal error
			continue
		}
		gvks = append(gvks, gvk)
		declTypes = append(declTypes, declType)
	}
	ctx.gvks = gvks
	ctx.declTypes = declTypes

	paramsGVK := c.paramsGVK(policy) // maybe empty, correctly handled
	paramsDeclType, err := c.declType(paramsGVK)
	if err != nil {
		if !errors.Is(err, resolver.ErrSchemaNotFound) {
			klog.V(2).ErrorS(err, "internal error: cannot resolve schema for params", "gvk", paramsGVK)
		}
		paramsDeclType = nil
	}
	ctx.paramGVK = paramsGVK
	ctx.paramDeclType = paramsDeclType
	ctx.variables = policy.Spec.Variables
	return ctx
}

// GVKs returns GVKs that the policy refers to in its MatchConstraints.
func (c *TypeCheckingContext) GVKs() []schema.GroupVersionKind {
	result := make([]schema.GroupVersionKind, len(c.gvks))
	copy(result, c.gvks)
	return result
}

// ParamGVK returns the GVK of the param.
// It may be empty of the policy does not refer to a param.
func (c *TypeCheckingContext) ParamGVK() schema.GroupVersionKind {
	return c.paramGVK
}

func (c *TypeChecker) declType(gvk schema.GroupVersionKind) (*apiservercel.DeclType, error) {
	if gvk.Empty() {
		return nil, nil
	}
	s, err := c.SchemaResolver.ResolveSchema(gvk)
	if err != nil {
		return nil, err
	}
	return common.SchemaDeclType(&openapi.Schema{Schema: s}, true).MaybeAssignTypeName(generateUniqueTypeName(gvk.Kind)), nil
}

func (c *TypeChecker) paramsGVK(policy *v1.ValidatingAdmissionPolicy) schema.GroupVersionKind {
	if policy.Spec.ParamKind == nil {
		return schema.GroupVersionKind{}
	}
	gv, err := schema.ParseGroupVersion(policy.Spec.ParamKind.APIVersion)
	if err != nil {
		return schema.GroupVersionKind{}
	}
	return gv.WithKind(policy.Spec.ParamKind.Kind)
}

// typesToCheck extracts a list of GVKs that needs type checking from the policy
// the result is sorted in the order of Group, Version, and Kind
func (c *TypeChecker) typesToCheck(p *v1.ValidatingAdmissionPolicy) []schema.GroupVersionKind {
	gvks := sets.New[schema.GroupVersionKind]()
	if p.Spec.MatchConstraints == nil || len(p.Spec.MatchConstraints.ResourceRules) == 0 {
		return nil
	}
	restMapperRefreshAttempted := false // at most once per policy, refresh RESTMapper and retry resolution.
	for _, rule := range p.Spec.MatchConstraints.ResourceRules {
		groups := extractGroups(&rule.Rule)
		if len(groups) == 0 {
			continue
		}
		versions := extractVersions(&rule.Rule)
		if len(versions) == 0 {
			continue
		}
		resources := extractResources(&rule.Rule)
		if len(resources) == 0 {
			continue
		}
		// sort GVRs so that the loop below provides
		// consistent results.
		sort.Strings(groups)
		sort.Strings(versions)
		sort.Strings(resources)
		count := 0
		for _, group := range groups {
			for _, version := range versions {
				for _, resource := range resources {
					gvr := schema.GroupVersionResource{
						Group:    group,
						Version:  version,
						Resource: resource,
					}
					resolved, err := c.RestMapper.KindsFor(gvr)
					if err != nil {
						if restMapperRefreshAttempted {
							// RESTMapper refresh happens at most once per policy
							continue
						}
						c.tryRefreshRESTMapper()
						restMapperRefreshAttempted = true
						resolved, err = c.RestMapper.KindsFor(gvr)
						if err != nil {
							continue
						}
					}
					for _, r := range resolved {
						if !r.Empty() {
							gvks.Insert(r)
							count++
							// early return if maximum number of types are already
							// collected
							if count == maxTypesToCheck {
								if gvks.Len() == 0 {
									return nil
								}
								return sortGVKList(gvks.UnsortedList())
							}
						}
					}
				}
			}
		}
	}
	if gvks.Len() == 0 {
		return nil
	}
	return sortGVKList(gvks.UnsortedList())
}

func extractGroups(rule *v1.Rule) []string {
	groups := make([]string, 0, len(rule.APIGroups))
	for _, group := range rule.APIGroups {
		// give up if wildcard
		if strings.ContainsAny(group, "*") {
			return nil
		}
		groups = append(groups, group)
	}
	return groups
}

func extractVersions(rule *v1.Rule) []string {
	versions := make([]string, 0, len(rule.APIVersions))
	for _, version := range rule.APIVersions {
		if strings.ContainsAny(version, "*") {
			return nil
		}
		versions = append(versions, version)
	}
	return versions
}

func extractResources(rule *v1.Rule) []string {
	resources := make([]string, 0, len(rule.Resources))
	for _, resource := range rule.Resources {
		// skip wildcard and subresources
		if strings.ContainsAny(resource, "*/") {
			continue
		}
		resources = append(resources, resource)
	}
	return resources
}
