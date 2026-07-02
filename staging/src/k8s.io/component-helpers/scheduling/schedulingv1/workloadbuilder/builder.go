/*
Copyright The Kubernetes Authors.

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

package workloadbuilder

import (
	"fmt"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// BuildOptions carries the identity of the Workload object that
// Build produces.
type BuildOptions struct {
	Name      string
	Namespace string
	// optional owner reference for garbage collection
	Owner *metav1.OwnerReference
}

// Build resolves config, runs callbacks, validates, and compiles the tree into
// a Workload. Errors are returned as a field.ErrorList rooted at the field paths
// each WorkloadItem declares in its FieldPaths, so callers get controller-
// relative paths instead of the library's internal IR structure.
func Build(root *WorkloadItem, opts BuildOptions) (*schedulingv1alpha3.Workload, field.ErrorList) {
	spec, errs := compileWorkloadItemTree(root)
	if len(errs) > 0 {
		return nil, errs
	}

	wl := &schedulingv1alpha3.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      opts.Name,
			Namespace: opts.Namespace,
		},
		Spec: spec,
	}

	ownerPath := field.NewPath("ownerReference")
	if opts.Owner == nil {
		return nil, field.ErrorList{field.Required(ownerPath, "a Workload must be owned by its controller for garbage collection")}
	}

	gv, err := schema.ParseGroupVersion(opts.Owner.APIVersion)
	if err != nil {
		return nil, field.ErrorList{field.Invalid(ownerPath.Child("apiVersion"), opts.Owner.APIVersion, err.Error())}
	}

	wl.OwnerReferences = []metav1.OwnerReference{*opts.Owner}
	wl.Spec.ControllerRef = &schedulingv1alpha3.TypedLocalObjectReference{
		APIGroup: gv.Group,
		Kind:     opts.Owner.Kind,
		Name:     opts.Owner.Name,
	}

	return wl, nil
}

// Validate checks if the WorkloadItem tree can be compiled into a Workload,
// returning the field errors that would prevent compilation. It runs the same
// resolution, callbacks, and semantic checks as Build, but discards the
// compiled result.
//
// Errors are reported against the field paths each WorkloadItem declares in its
// FieldPaths, so the API server gets paths that match the calling controller's
// API surface. Where a path is not declared, a relative fallback is used.
func Validate(root *WorkloadItem) field.ErrorList {
	_, errs := compileWorkloadItemTree(root)
	return errs
}

// isComposite reports whether the item is a structural group (has children)
// rather than a leaf that compiles to a single PodGroupTemplate.
func isComposite(item *WorkloadItem) bool {
	return len(item.Children) > 0
}

// compileWorkloadItemTree compiles the root item into the WorkloadSpec it
// produces. For now, only single-level (leaf) items are supported, so the
// returned spec carries exactly one PodGroupTemplate; Build fills in the object
// identity separately.
func compileWorkloadItemTree(root *WorkloadItem) (schedulingv1alpha3.WorkloadSpec, field.ErrorList) {
	var spec schedulingv1alpha3.WorkloadSpec

	if root == nil {
		return spec, field.ErrorList{field.Required(field.NewPath("workloadItem"), "root WorkloadItem must not be nil")}
	}

	if isComposite(root) {
		// TODO: once CompositePodGroupTemplate lands, a composite item should compile to
		// a CompositePodGroupTemplate (nesting its children) instead of being rejected,
		// and parent-child conflict validation (plus leaf-name uniqueness and the
		// WorkloadMaxPodGroupTemplates cap) belongs here.
		return spec, field.ErrorList{field.Forbidden(field.NewPath("children"), fmt.Sprintf("composite WorkloadItem %q (with children) is not yet supported: the CompositePodGroup API has not landed", root.Name))}
	}

	pgTemplate, errs := compilePodGroupTemplate(root)
	if len(errs) > 0 {
		return spec, errs
	}
	spec.PodGroupTemplates = append(spec.PodGroupTemplates, pgTemplate)

	return spec, nil
}

// compilePodGroupTemplate resolves a leaf item's config, runs its callbacks,
// then compiles it into a single PodGroupTemplate, accumulating any errors.
func compilePodGroupTemplate(item *WorkloadItem) (schedulingv1alpha3.PodGroupTemplate, field.ErrorList) {
	fp := itemPaths(item)

	// Reject an empty name before resolution and callbacks run, so defaulting or
	// a callback cannot "recover" a name the caller never set.
	if item.Name == "" {
		return schedulingv1alpha3.PodGroupTemplate{}, field.ErrorList{field.Required(pathOrFallback(fp.Name, "name"), "workload item name cannot be empty")}
	}

	// resolveSchedulingConfig also runs the item's callbacks.
	item.ResolvedConfig = resolveSchedulingConfig(item)

	return buildLeafTemplate(item.Name, item.ResolvedConfig, fp)
}

// resolveSchedulingConfig merges DefaultConfig with UserConfig then runs the
// item's callbacks against the merged result to apply controller defaulting.
// Inputs are deep-copied so callbacks can't mutate the caller's configs. The
// resolved config is stored on item.ResolvedConfig so callbacks can read and
// mutate it, and is also returned for convenience.
func resolveSchedulingConfig(item *WorkloadItem) *SchedulingConfig {
	resolved := item.DefaultConfig.DeepCopy()
	if resolved == nil {
		resolved = &SchedulingConfig{}
	}

	if user := item.UserConfig.DeepCopy(); user != nil {
		if user.Policy != nil {
			resolved.Policy = user.Policy
		}
		if user.Constraints != nil {
			resolved.Constraints = user.Constraints
		}
		if user.DisruptionMode != nil {
			resolved.DisruptionMode = user.DisruptionMode
		}
		if len(user.ResourceClaims) > 0 {
			resolved.ResourceClaims = user.ResourceClaims
		}
	}

	item.ResolvedConfig = resolved
	for _, cb := range item.Callbacks {
		if cb != nil {
			cb(item)
		}
	}

	return item.ResolvedConfig
}

// buildLeafTemplate converts a resolved config into one PodGroupTemplate,
// accumulating semantic validation errors against the item's declared paths.
func buildLeafTemplate(name string, cfg *SchedulingConfig, fp *FieldPaths) (schedulingv1alpha3.PodGroupTemplate, field.ErrorList) {
	var allErrs field.ErrorList
	tmpl := schedulingv1alpha3.PodGroupTemplate{Name: name}

	if cfg == nil {
		cfg = &SchedulingConfig{}
	}

	policy, errs := compileSchedulingPolicy(cfg.Policy, fp)
	allErrs = append(allErrs, errs...)
	tmpl.SchedulingPolicy = policy

	if cfg.Constraints != nil && len(cfg.Constraints.Topology) > 0 {
		// Mirror the validation for maxItems=1 on PodGroupSchedulingConstraints.Topology
		// so we fail early.
		if len(cfg.Constraints.Topology) > 1 {
			allErrs = append(allErrs,
				field.TooMany(fp.Constraints.Child("topology"), len(cfg.Constraints.Topology), 1))
		}
		topology := make([]schedulingv1alpha3.TopologyConstraint, len(cfg.Constraints.Topology))
		copy(topology, cfg.Constraints.Topology)
		tmpl.SchedulingConstraints = &schedulingv1alpha3.PodGroupSchedulingConstraints{
			Topology: topology,
		}
	}

	if cfg.DisruptionMode != nil {
		dm, errs := compileDisruptionMode(cfg.DisruptionMode, fp)
		allErrs = append(allErrs, errs...)
		if len(errs) == 0 {
			tmpl.DisruptionMode = dm
		}
	}

	if len(cfg.ResourceClaims) > 0 {
		claimsPath := pathOrFallback(fp.ResourceClaims, "resourceClaims")
		if len(cfg.ResourceClaims) > schedulingv1alpha3.MaxPodGroupResourceClaims {
			allErrs = append(allErrs, field.TooMany(claimsPath, len(cfg.ResourceClaims), schedulingv1alpha3.MaxPodGroupResourceClaims))
		}

		seenClaims := make(map[string]struct{})
		for i := range cfg.ResourceClaims {
			rc := cfg.ResourceClaims[i]
			claimPath := claimsPath.Index(i)

			if rc.Name == "" {
				allErrs = append(allErrs, field.Required(claimPath.Child("name"), "resource claim name cannot be empty"))
			} else if _, ok := seenClaims[rc.Name]; ok {
				allErrs = append(allErrs, field.Duplicate(claimPath.Child("name"), rc.Name))
			} else {
				seenClaims[rc.Name] = struct{}{}
			}

			if (rc.ResourceClaimName == nil && rc.ResourceClaimTemplateName == nil) || (rc.ResourceClaimName != nil && rc.ResourceClaimTemplateName != nil) {
				allErrs = append(allErrs, field.Invalid(claimPath, "", "exactly one of resourceClaimName or resourceClaimTemplateName must be set"))
			}
		}

		tmpl.ResourceClaims = compileResourceClaims(cfg.ResourceClaims)
	}

	return tmpl, allErrs
}

// compileSchedulingPolicy maps the IR policy onto the API policy. A nil
// policy resolves to Basic, while Gang requires a positive MinCount
// resolved beforehand.
func compileSchedulingPolicy(policy *SchedulingPolicy, fp *FieldPaths) (schedulingv1alpha3.PodGroupSchedulingPolicy, field.ErrorList) {
	basic := schedulingv1alpha3.PodGroupSchedulingPolicy{
		Basic: &schedulingv1alpha3.BasicSchedulingPolicy{},
	}

	if policy == nil {
		return basic, nil
	}

	policyPath := pathOrFallback(fp.Policy, "policy")

	switch {
	case policy.Basic != nil && policy.Gang != nil:
		return schedulingv1alpha3.PodGroupSchedulingPolicy{}, field.ErrorList{field.Invalid(policyPath, "", "exactly one scheduling policy must be set")}
	case policy.Gang != nil:
		minCountPath := pathOrFallback(fp.GangMinCount, "policy", "gang", "minCount")
		if policy.Gang.MinCount == nil {
			return schedulingv1alpha3.PodGroupSchedulingPolicy{}, field.ErrorList{field.Required(minCountPath, "gang scheduling requires minCount to be set after resolution")}
		}
		if *policy.Gang.MinCount < 1 {
			return schedulingv1alpha3.PodGroupSchedulingPolicy{}, field.ErrorList{field.Invalid(minCountPath, *policy.Gang.MinCount, "must be at least 1")}
		}
		return schedulingv1alpha3.PodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: *policy.Gang.MinCount},
		}, nil
	default:
		// nil policy, Basic only, or an empty policy all resolve to Basic.
		return basic, nil
	}
}

func compileDisruptionMode(dm *DisruptionMode, fp *FieldPaths) (*schedulingv1alpha3.DisruptionMode, field.ErrorList) {
	dmPath := pathOrFallback(fp.DisruptionMode, "disruptionMode")

	switch {
	case dm.Single != nil && dm.All != nil:
		return nil, field.ErrorList{field.Invalid(dmPath, "", "exactly one disruption mode must be set")}
	case dm.All != nil:
		return &schedulingv1alpha3.DisruptionMode{All: &schedulingv1alpha3.AllDisruptionMode{}}, nil
	case dm.Single != nil:
		return &schedulingv1alpha3.DisruptionMode{Single: &schedulingv1alpha3.SingleDisruptionMode{}}, nil
	default:
		return nil, field.ErrorList{field.Required(dmPath, "exactly one disruption mode must be set")}
	}
}

func compileResourceClaims(claims []ResourceClaim) []schedulingv1alpha3.PodGroupResourceClaim {
	result := make([]schedulingv1alpha3.PodGroupResourceClaim, len(claims))
	for i := range claims {
		result[i] = schedulingv1alpha3.PodGroupResourceClaim{
			Name:                      claims[i].Name,
			ResourceClaimName:         claims[i].ResourceClaimName,
			ResourceClaimTemplateName: claims[i].ResourceClaimTemplateName,
		}
	}
	return result
}

// pathOrFallback returns p when it is non-nil, otherwise a relative path built
// from the fallback names. It lets the builder degrade gracefully when a
// controller did not declare a path for a given field.
func pathOrFallback(p *field.Path, fallback ...string) *field.Path {
	if p != nil {
		return p
	}
	return field.NewPath(fallback[0], fallback[1:]...)
}

// itemPaths returns the item's declared FieldPaths, or an empty mapping so
// callers can dereference fields without nil checks.
func itemPaths(item *WorkloadItem) *FieldPaths {
	if item.FieldPaths != nil {
		return item.FieldPaths
	}
	return &FieldPaths{}
}
