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
	"context"
	"fmt"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// WorkloadBuilder translates a WorkloadItem tree into a Workload.
type WorkloadBuilder interface {
	// Build resolves config, runs callbacks, validates, and compiles the tree.
	// A non-nil owner is recorded as both an OwnerReference (for GC) and the
	// Workload's ControllerRef.
	Build(ctx context.Context, name, namespace string, owner *metav1.OwnerReference) (*schedulingv1alpha3.Workload, error)

	// Validate runs Build's resolution and checks without compiling, returning
	// field errors rooted at fldPath. It reads only the tree (no cluster state),
	// so the API server can call it to reject configs Build couldn't compile.
	// Callbacks run, so pass the same ones used for Build.
	Validate(fldPath *field.Path) field.ErrorList
}

// NewBuilder initializes a builder rooted at the given item.
func NewBuilder(root *WorkloadItem) WorkloadBuilder {
	return &builderImpl{root: root}
}

type builderImpl struct {
	root *WorkloadItem
}

func (b *builderImpl) Build(_ context.Context, name, namespace string, owner *metav1.OwnerReference) (*schedulingv1alpha3.Workload, error) {
	if b.root == nil {
		return nil, fmt.Errorf("root WorkloadItem must not be nil")
	}

	templates, allErrs := b.compileWorkloadItemTree(field.NewPath("spec", "podGroupTemplates"))
	if err := allErrs.ToAggregate(); err != nil {
		return nil, fmt.Errorf("compiling workload %q: %w", name, err)
	}

	wl := &schedulingv1alpha3.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: schedulingv1alpha3.WorkloadSpec{
			PodGroupTemplates: templates,
		},
	}

	if owner != nil {
		gv, err := schema.ParseGroupVersion(owner.APIVersion)
		if err != nil {
			return nil, fmt.Errorf("parsing owner APIVersion %q: %w", owner.APIVersion, err)
		}

		wl.OwnerReferences = []metav1.OwnerReference{*owner}
		wl.Spec.ControllerRef = &schedulingv1alpha3.TypedLocalObjectReference{
			APIGroup: gv.Group,
			Kind:     owner.Kind,
			Name:     owner.Name,
		}
	}

	return wl, nil
}

func (b *builderImpl) Validate(fldPath *field.Path) field.ErrorList {
	if b.root == nil {
		return field.ErrorList{field.Required(fldPath, "root WorkloadItem must not be nil")}
	}
	_, allErrs := b.compileWorkloadItemTree(fldPath)
	return allErrs
}

// compileWorkloadItemTree resolves the tree into PodGroupTemplates and the
// errors found against templatesPath. Shared by Build and Validate so they agree.
func (b *builderImpl) compileWorkloadItemTree(templatesPath *field.Path) ([]schedulingv1alpha3.PodGroupTemplate, field.ErrorList) {
	templates := make([]schedulingv1alpha3.PodGroupTemplate, 0)
	seen := make(map[string]struct{})
	allErrs := field.ErrorList{}

	b.compileWorkloadItem(b.root, templatesPath, &templates, seen, &allErrs)

	if len(templates) > schedulingv1alpha3.WorkloadMaxPodGroupTemplates {
		allErrs = append(allErrs, field.TooMany(templatesPath, len(templates), schedulingv1alpha3.WorkloadMaxPodGroupTemplates))
	}
	return templates, allErrs
}

// compileWorkloadItem resolves the item, runs its callbacks, then emits a
// leaf template or recurses into children. Errors accumulate so Build
// reports them all at once.
func (b *builderImpl) compileWorkloadItem(item *WorkloadItem, path *field.Path, templates *[]schedulingv1alpha3.PodGroupTemplate,
	seen map[string]struct{}, allErrs *field.ErrorList) {
	item.ResolvedConfig = resolveSchedulingConfig(item)
	for _, cb := range item.Callbacks {
		if cb != nil {
			cb(item)
		}
	}

	// TODO: Implement Parent-Child Conflict Validation here once
	// CompositePodGroup semantics are fully defined (e.g. ensuring a nested leaf
	// group does not declare a conflicting disruption mode not supported by its parent).

	if len(item.Children) > 0 {
		for _, child := range item.Children {
			b.compileWorkloadItem(child, path, templates, seen, allErrs)
		}
		return
	}

	itemPath := path.Key(item.Name)
	if item.Name == "" {
		*allErrs = append(*allErrs, field.Required(itemPath.Child("name"), "workload item name cannot be empty"))
		return
	}
	if _, ok := seen[item.Name]; ok {
		*allErrs = append(*allErrs, field.Duplicate(itemPath.Child("name"), item.Name))
		return
	}
	seen[item.Name] = struct{}{}

	tmpl, errs := buildLeafTemplate(item.Name, item.ResolvedConfig, itemPath)
	*allErrs = append(*allErrs, errs...)
	*templates = append(*templates, tmpl)
}

// resolveSchedulingConfig merges DefaultConfig with UserConfig field-by-field, user
// winning. Inputs are deep-copied so callbacks can't mutate the caller's configs.
func resolveSchedulingConfig(item *WorkloadItem) *SchedulingConfig {
	resolved := item.DefaultConfig.DeepCopy()
	if resolved == nil {
		resolved = &SchedulingConfig{}
	}

	user := item.UserConfig.DeepCopy()
	if user == nil {
		return resolved
	}

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

	return resolved
}

// buildLeafTemplate converts a resolved config into one PodGroupTemplate,
// accumulating semantic validation errors.
func buildLeafTemplate(name string, cfg *SchedulingConfig, path *field.Path) (schedulingv1alpha3.PodGroupTemplate, field.ErrorList) {
	var allErrs field.ErrorList
	tmpl := schedulingv1alpha3.PodGroupTemplate{Name: name}

	if cfg == nil {
		cfg = &SchedulingConfig{}
	}

	policy, errs := compileSchedulingPolicy(cfg.Policy, path.Child("schedulingPolicy"))
	allErrs = append(allErrs, errs...)
	tmpl.SchedulingPolicy = policy

	if cfg.Constraints != nil && len(cfg.Constraints.Topology) > 0 {
		topology := make([]schedulingv1alpha3.TopologyConstraint, len(cfg.Constraints.Topology))
		copy(topology, cfg.Constraints.Topology)
		tmpl.SchedulingConstraints = &schedulingv1alpha3.PodGroupSchedulingConstraints{
			Topology: topology,
		}
	}

	if cfg.DisruptionMode != nil {
		dm, errs := compileDisruptionMode(cfg.DisruptionMode, path.Child("disruptionMode"))
		allErrs = append(allErrs, errs...)
		tmpl.DisruptionMode = dm
	}

	if len(cfg.ResourceClaims) > 0 {
		claimsPath := path.Child("resourceClaims")
		if len(cfg.ResourceClaims) > schedulingv1alpha3.MaxPodGroupResourceClaims {
			allErrs = append(allErrs, field.TooMany(claimsPath, len(cfg.ResourceClaims), schedulingv1alpha3.MaxPodGroupResourceClaims))
		}

		seenClaims := make(map[string]struct{})
		for i, rc := range cfg.ResourceClaims {
			idxPath := claimsPath.Index(i)
			if rc.Name == "" {
				allErrs = append(allErrs, field.Required(idxPath.Child("name"), "resource claim name cannot be empty"))
			} else if _, ok := seenClaims[rc.Name]; ok {
				allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), rc.Name))
			} else {
				seenClaims[rc.Name] = struct{}{}
			}

			if (rc.ResourceClaimName == nil && rc.ResourceClaimTemplateName == nil) || (rc.ResourceClaimName != nil && rc.ResourceClaimTemplateName != nil) {
				allErrs = append(allErrs, field.Invalid(idxPath, "", "exactly one of resourceClaimName or resourceClaimTemplateName must be set"))
			}
		}

		tmpl.ResourceClaims = compileResourceClaims(cfg.ResourceClaims)
	}

	return tmpl, allErrs
}

// compileSchedulingPolicy maps the IR policy onto the API policy. A nil
// policy resolves to Basic, while Gang requires a positive MinCount
// resolved beforehand.
func compileSchedulingPolicy(policy *SchedulingPolicy, path *field.Path) (schedulingv1alpha3.PodGroupSchedulingPolicy, field.ErrorList) {
	if policy == nil || (policy.Basic == nil && policy.Gang == nil) {
		return schedulingv1alpha3.PodGroupSchedulingPolicy{
			Basic: &schedulingv1alpha3.BasicSchedulingPolicy{},
		}, nil
	}

	if policy.Basic != nil && policy.Gang != nil {
		return schedulingv1alpha3.PodGroupSchedulingPolicy{}, field.ErrorList{
			field.Invalid(path, "", "exactly one scheduling policy must be set"),
		}
	}

	if policy.Gang != nil {
		if policy.Gang.MinCount == nil {
			return schedulingv1alpha3.PodGroupSchedulingPolicy{}, field.ErrorList{
				field.Required(path.Child("gang", "minCount"), "gang scheduling requires minCount to be set after resolution"),
			}
		}
		if *policy.Gang.MinCount < 1 {
			return schedulingv1alpha3.PodGroupSchedulingPolicy{}, field.ErrorList{
				field.Invalid(path.Child("gang", "minCount"), *policy.Gang.MinCount, "must be at least 1"),
			}
		}
		return schedulingv1alpha3.PodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: *policy.Gang.MinCount},
		}, nil
	}

	return schedulingv1alpha3.PodGroupSchedulingPolicy{
		Basic: &schedulingv1alpha3.BasicSchedulingPolicy{},
	}, nil
}

func compileDisruptionMode(dm *DisruptionMode, path *field.Path) (*schedulingv1alpha3.DisruptionMode, field.ErrorList) {
	switch {
	case dm.Single != nil && dm.All != nil:
		return nil, field.ErrorList{field.Invalid(path, "", "exactly one disruption mode must be set")}
	case dm.All != nil:
		return &schedulingv1alpha3.DisruptionMode{All: &schedulingv1alpha3.AllDisruptionMode{}}, nil
	case dm.Single != nil:
		return &schedulingv1alpha3.DisruptionMode{Single: &schedulingv1alpha3.SingleDisruptionMode{}}, nil
	default:
		return nil, field.ErrorList{field.Required(path, "exactly one disruption mode must be set")}
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
