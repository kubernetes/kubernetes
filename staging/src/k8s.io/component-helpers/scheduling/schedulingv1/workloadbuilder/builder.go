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
)

// SchedulingPolicyOption enumerates the scheduling policies a controller opts
// into. The allow-list rejects any policy not listed, so building-block
// additions stay denied until a controller extends its list.
type SchedulingPolicyOption int

const (
	// BasicPolicy allows the Basic scheduling policy.
	BasicPolicy SchedulingPolicyOption = iota
	// GangPolicy allows the Gang scheduling policy.
	GangPolicy
)

// DisruptionModeOption enumerates the disruption modes a controller opts into.
type DisruptionModeOption int

const (
	// SingleMode allows the Single disruption mode.
	SingleMode DisruptionModeOption = iota
	// AllMode allows the All disruption mode.
	AllMode
)

// BuildOptions carries the identity of the Workload object that BuildWorkload
// produces, plus the controller's scheduling allow-lists.
type BuildOptions struct {
	Name      string
	Namespace string
	// optional owner reference for garbage collection
	Owner *metav1.OwnerReference

	// AllowedPolicies is the set of scheduling policies the controller opts
	// into. Validate rejects any policy outside this set. An empty set allows
	// nothing.
	AllowedPolicies []SchedulingPolicyOption
	// AllowedDisruptionModes is the set of disruption modes the controller opts
	// into. Validate rejects any mode outside this set. An empty set allows
	// nothing.
	AllowedDisruptionModes []DisruptionModeOption
}

// Builder turns a controller's WorkloadItem tree into scheduler-facing objects.
// Construct it once with NewBuilder, then call Validate, BuildWorkload, and
// NewPodGroup as needed.
type Builder struct {
	root *WorkloadItem
	opts BuildOptions

	// workload holds the Workload NewPodGroup materializes from: either the
	// Workload compiled by BuildWorkload, or one supplied via SetExistingWorkload.
	workload *schedulingv1alpha3.Workload

	// externalWorkload records that workload was supplied via SetExistingWorkload
	// rather than compiled by this library. BuildWorkload refuses to run in this
	// mode so a caller can't recompile over a persisted or parent-authored Workload.
	externalWorkload bool
}

// build compiles the tree and sets the Workload's identity. It fails fast,
// returning the first error it encounters.
func (b *Builder) build() (*schedulingv1alpha3.Workload, error) {
	spec, err := compileWorkloadItemTree(b.root)
	if err != nil {
		return nil, err
	}

	wl := &schedulingv1alpha3.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      b.opts.Name,
			Namespace: b.opts.Namespace,
		},
		Spec: spec,
	}

	if b.opts.Owner == nil {
		return nil, fmt.Errorf("a Workload must be owned by its controller for garbage collection")
	}

	gv, err := schema.ParseGroupVersion(b.opts.Owner.APIVersion)
	if err != nil {
		return nil, fmt.Errorf("invalid owner apiVersion %q: %w", b.opts.Owner.APIVersion, err)
	}

	wl.OwnerReferences = []metav1.OwnerReference{*b.opts.Owner}
	wl.Spec.ControllerRef = &schedulingv1alpha3.TypedLocalObjectReference{
		APIGroup: gv.Group,
		Kind:     b.opts.Owner.Kind,
		Name:     b.opts.Owner.Name,
	}

	return wl, nil
}

// compileWorkloadItemTree compiles the root item into the WorkloadSpec it
// produces. Only single-level (leaf) items are supported, so the returned spec
// carries exactly one PodGroupTemplate; Build fills in the object identity
// separately.
func compileWorkloadItemTree(root *WorkloadItem) (schedulingv1alpha3.WorkloadSpec, error) {
	var spec schedulingv1alpha3.WorkloadSpec

	if root == nil {
		return spec, fmt.Errorf("root WorkloadItem must not be nil")
	}

	pgTemplate, err := compilePodGroupTemplate(root)
	if err != nil {
		return spec, err
	}
	spec.PodGroupTemplates = append(spec.PodGroupTemplates, pgTemplate)

	return spec, nil
}

// compilePodGroupTemplate resolves a leaf item's config, runs its callbacks,
// then compiles it into a single PodGroupTemplate.
func compilePodGroupTemplate(item *WorkloadItem) (schedulingv1alpha3.PodGroupTemplate, error) {
	// Reject an empty name before resolution and callbacks run, so defaulting or
	// a callback cannot "recover" a name the caller never set.
	if item.Name == "" {
		return schedulingv1alpha3.PodGroupTemplate{}, fmt.Errorf("workload item name cannot be empty")
	}

	// resolveSchedulingConfig also runs the item's callbacks.
	resolved := resolveSchedulingConfig(item)

	return buildLeafTemplate(item.Name, resolved)
}

// resolveSchedulingConfig merges DefaultConfig with UserConfig then runs the
// item's callbacks against the merged result to apply controller defaulting.
// Inputs are deep-copied so callbacks can't mutate the caller's configs. The
// resolved config is passed to each callback to read and mutate, and is
// returned once every callback has run.
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
		if user.PriorityClassName != "" {
			resolved.PriorityClassName = user.PriorityClassName
		}
	}

	for _, cb := range item.Callbacks {
		if cb != nil {
			cb(resolved)
		}
	}

	return resolved
}

// buildLeafTemplate converts a resolved config into one PodGroupTemplate.
func buildLeafTemplate(name string, cfg *SchedulingConfig) (schedulingv1alpha3.PodGroupTemplate, error) {
	tmpl := schedulingv1alpha3.PodGroupTemplate{Name: name}

	if cfg == nil {
		cfg = &SchedulingConfig{}
	}

	tmpl.PriorityClassName = cfg.PriorityClassName

	policy, err := compileSchedulingPolicy(cfg.Policy)
	if err != nil {
		return schedulingv1alpha3.PodGroupTemplate{}, err
	}
	tmpl.SchedulingPolicy = policy

	if cfg.Constraints != nil && len(cfg.Constraints.Topology) > 0 {
		topology := make([]schedulingv1alpha3.TopologyConstraint, len(cfg.Constraints.Topology))
		copy(topology, cfg.Constraints.Topology)
		tmpl.SchedulingConstraints = &schedulingv1alpha3.PodGroupSchedulingConstraints{
			Topology: topology,
		}
	}

	if cfg.DisruptionMode != nil {
		tmpl.DisruptionMode = compileDisruptionMode(cfg.DisruptionMode)
	}

	if len(cfg.ResourceClaims) > 0 {
		tmpl.ResourceClaims = compileResourceClaims(cfg.ResourceClaims)
	}

	return tmpl, nil
}

// compileSchedulingPolicy maps the IR policy onto the API policy. A nil
// policy resolves to Basic; Gang requires a MinCount resolved beforehand so it
// can populate the non-pointer API field. The minCount>=1 bound is a structural
// constraint enforced by declarative validation, so it is not repeated here.
func compileSchedulingPolicy(policy *SchedulingPolicy) (schedulingv1alpha3.PodGroupSchedulingPolicy, error) {
	basic := schedulingv1alpha3.PodGroupSchedulingPolicy{
		Basic: &schedulingv1alpha3.BasicSchedulingPolicy{},
	}

	if policy == nil {
		return basic, nil
	}

	// The basic/gang union is enforced by declarative validation when the
	// compiled Workload is submitted, only the compile-time invariant that a
	// resolved gang policy carries a minCount value is checked here.
	switch {
	case policy.Gang != nil:
		if policy.Gang.MinCount == nil {
			return schedulingv1alpha3.PodGroupSchedulingPolicy{}, fmt.Errorf("gang scheduling requires minCount to be set after resolution")
		}
		return schedulingv1alpha3.PodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: *policy.Gang.MinCount},
		}, nil
	default:
		// nil policy, Basic only, or an empty policy all resolve to Basic.
		return basic, nil
	}
}

// compileDisruptionMode maps the IR disruption mode onto the API mode. The
// exactly-one-of union is enforced by declarative validation when the compiled
// Workload is submitted, so a mode with neither member set compiles to nil.
func compileDisruptionMode(dm *DisruptionMode) *schedulingv1alpha3.DisruptionMode {
	switch {
	case dm.All != nil:
		return &schedulingv1alpha3.DisruptionMode{All: &schedulingv1alpha3.AllDisruptionMode{}}
	case dm.Single != nil:
		return &schedulingv1alpha3.DisruptionMode{Single: &schedulingv1alpha3.SingleDisruptionMode{}}
	default:
		return nil
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
