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

	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
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

	// DisableDeclarativeValidation skips running declarative validation on the
	// input building blocks. In-tree controllers like Job disable this because
	// their own APIServer validation already runs it.
	DisableDeclarativeValidation bool
}

// Builder turns a controller's WorkloadItem tree into scheduler-facing objects.
// Construct it once with NewBuilder, then call Validate, BuildWorkload, and
// NewPodGroup as needed.
type Builder struct {
	root *WorkloadItem
	opts BuildOptions

	// workload caches the Workload compiled by BuildWorkload so multiple
	// PodGroups can be materialized without recompiling.
	workload *schedulingv1beta1.Workload

	// existingWorkload is a persisted Workload supplied to
	// NewBuilderFromExistingWorkload. When set, NewPodGroup materializes from it,
	// BuildWorkload is refused so a caller can't recompile over a persisted or
	// parent-authored Workload, and Validate is refused too because the persisted
	// object already passed apiserver validation and there is no tree to check.
	existingWorkload *schedulingv1beta1.Workload

	// podGroupTemplateIndex and compositePodGroupTemplateIndex cache template
	// lookups by name. They are built lazily on the first matching
	// materialization call and reused so repeated NewPodGroup/NewCompositePodGroup
	// calls resolve in O(1) instead of rescanning the workload's template tree.
	podGroupTemplateIndex          map[string]*schedulingv1beta1.PodGroupTemplate
	compositePodGroupTemplateIndex map[string]*schedulingv1beta1.CompositePodGroupTemplate
}

// NewBuilder returns a Builder for the given WorkloadItem tree and options.
func NewBuilder(root *WorkloadItem, opts BuildOptions) *Builder {
	return &Builder{root: root, opts: opts}
}

// NewBuilderFromExistingWorkload returns a Builder that materializes PodGroups
// from an already-persisted Workload rather than compiling one from a
// WorkloadItem tree. Use it when a parent controller (or a hand-authored
// template) owns and compiled the Workload. BuildWorkload and Validate are both
// refused; only NewPodGroup is meaningful, using opts.Owner for the PodGroup's
// controller ownerRef.
func NewBuilderFromExistingWorkload(workload *schedulingv1beta1.Workload, opts BuildOptions) *Builder {
	return &Builder{opts: opts, existingWorkload: workload}
}

// BuildWorkload compiles the WorkloadItem tree into a Workload, sets its
// identity and controllerRef, and caches the result.
func (b *Builder) BuildWorkload() (*schedulingv1beta1.Workload, error) {
	if b.existingWorkload != nil {
		return nil, fmt.Errorf("BuildWorkload is not available on a Builder created from an existing Workload: it materializes from the supplied Workload")
	}
	if b.workload != nil {
		return b.workload, nil
	}
	wl, err := b.build()
	if err != nil {
		return nil, err
	}
	b.workload = wl
	return wl, nil
}

// NewPodGroup materializes a runtime PodGroup from the named PodGroupTemplate of
// the Builder's Workload. The template is resolved through a name index built
// lazily on first use and reused across calls.
func (b *Builder) NewPodGroup(podGroupName, templateName string) (*schedulingv1beta1.PodGroup, error) {
	workload, err := b.resolveWorkload()
	if err != nil {
		return nil, err
	}
	if b.podGroupTemplateIndex == nil {
		b.podGroupTemplateIndex = indexPodGroupTemplates(workload)
	}
	tmpl := b.podGroupTemplateIndex[templateName]
	if tmpl == nil {
		return nil, fmt.Errorf("podGroupTemplate %q not found in workload %q (have %v)", templateName, workload.Name, podGroupTemplateNames(workload))
	}
	return newPodGroup(workload, tmpl, podGroupName, b.resolveOwners()), nil
}

// NewCompositePodGroup materializes a runtime CompositePodGroup from the named
// CompositePodGroupTemplate of the Builder's Workload. The template is resolved
// through a name index built lazily on first use and reused across calls. Parent
// linkage (spec.parentCompositePodGroupName) and the child (Composite)PodGroups
// are left to the caller.
func (b *Builder) NewCompositePodGroup(compositePodGroupName, templateName string) (*schedulingv1beta1.CompositePodGroupTemplate, error) {
	workload, err := b.resolveWorkload()
	if err != nil {
		return nil, err
	}
	if b.compositePodGroupTemplateIndex == nil {
		b.compositePodGroupTemplateIndex = indexCompositePodGroupTemplates(workload)
	}
	tmpl := b.compositePodGroupTemplateIndex[templateName]
	if tmpl == nil {
		return nil, fmt.Errorf("compositePodGroupTemplate %q not found in workload %q (have %v)", templateName, workload.Name, compositePodGroupTemplateNames(workload))
	}
	return newCompositePodGroup(tmpl, compositePodGroupName), nil
}

// resolveWorkload returns the Workload that NewPodGroup and
// NewCompositePodGroup materialize from: the supplied existing Workload if the
// Builder was constructed from one, otherwise the Workload compiled by
// BuildWorkload. It errors when neither is available.
func (b *Builder) resolveWorkload() (*schedulingv1beta1.Workload, error) {
	if b.existingWorkload != nil {
		return b.existingWorkload, nil
	}
	if b.workload == nil {
		return nil, fmt.Errorf("no Workload available: call BuildWorkload or use NewBuilderFromExistingWorkload first")
	}
	return b.workload, nil
}

// resolveOwners returns the ownerReferences to stamp on a materialized
// (Composite)PodGroup, or nil when no controller owner is configured.
func (b *Builder) resolveOwners() []metav1.OwnerReference {
	if b.opts.Owner != nil {
		return []metav1.OwnerReference{*b.opts.Owner}
	}
	return nil
}

// build compiles the tree and sets the Workload's identity. It fails fast,
// returning the first error it encounters.
func (b *Builder) build() (*schedulingv1beta1.Workload, error) {
	spec, err := compileWorkloadItemTree(b.root)
	if err != nil {
		return nil, err
	}

	wl := &schedulingv1beta1.Workload{
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
	wl.Spec.ControllerRef = &schedulingv1beta1.TypedLocalObjectReference{
		APIGroup: gv.Group,
		Kind:     b.opts.Owner.Kind,
		Name:     b.opts.Owner.Name,
	}

	return wl, nil
}

// isComposite reports whether the item is a composite node (a group-of-groups
// compiled into a CompositePodGroupTemplate) rather than a leaf (compiled into a
// single PodGroupTemplate).
func isComposite(item *WorkloadItem) bool {
	return len(item.Children) > 0
}

// compileWorkloadItemTree compiles the root item into the WorkloadSpec it
// produces. A composite root yields a single CompositePodGroupTemplate (whose
// children are compiled recursively); a leaf root yields a single
// PodGroupTemplate. Build fills in the object identity separately.
func compileWorkloadItemTree(root *WorkloadItem) (schedulingv1beta1.WorkloadSpec, error) {
	var spec schedulingv1beta1.WorkloadSpec

	if err := validateWorkloadItemTree(root); err != nil {
		return spec, err
	}

	if isComposite(root) {
		cpgTemplate, err := compileCompositePodGroupTemplate(root)
		if err != nil {
			return spec, err
		}
		spec.CompositePodGroupTemplates = append(spec.CompositePodGroupTemplates, cpgTemplate)
		return spec, nil
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
func compilePodGroupTemplate(item *WorkloadItem) (schedulingv1beta1.PodGroupTemplate, error) {
	// resolveSchedulingConfig also runs the item's callbacks.
	resolved := resolveSchedulingConfig(item)

	return buildLeafTemplate(item.Name, resolved)
}

// resolveSchedulingConfig merges DefaultConfig with the mapped Input then runs
// the item's callbacks against the merged result to apply controller defaulting.
// It selects the composite or leaf input mapper based on whether the item is a
// composite node, so the same defaulting/callback machinery serves both paths.
// Inputs are deep-copied so callbacks can't mutate the caller's configs. The
// resolved config is passed to each callback to read and mutate, and is
// returned once every callback has run.
func resolveSchedulingConfig(item *WorkloadItem) *SchedulingConfig {
	resolved := item.DefaultConfig.DeepCopy()
	if resolved == nil {
		resolved = &SchedulingConfig{}
	}

	userConfig := mapWorkloadInput(item.Input)
	if isComposite(item) {
		userConfig = mapCompositeGroupInput(item.Input)
	}

	if userConfig.Policy != nil {
		resolved.Policy = userConfig.Policy
	}
	if userConfig.Constraints != nil {
		resolved.Constraints = userConfig.Constraints
	}
	if userConfig.DisruptionMode != nil {
		resolved.DisruptionMode = userConfig.DisruptionMode
	}
	if len(userConfig.ResourceClaims) > 0 {
		resolved.ResourceClaims = userConfig.ResourceClaims
	}
	if userConfig.PriorityClassName != "" {
		resolved.PriorityClassName = userConfig.PriorityClassName
	}

	for _, cb := range item.Callbacks {
		if cb != nil {
			cb(resolved)
		}
	}

	return resolved
}

// buildLeafTemplate converts a resolved config into one PodGroupTemplate.
func buildLeafTemplate(name string, cfg *SchedulingConfig) (schedulingv1beta1.PodGroupTemplate, error) {
	tmpl := schedulingv1beta1.PodGroupTemplate{Name: name}

	if cfg == nil {
		cfg = &SchedulingConfig{}
	}

	tmpl.PriorityClassName = cfg.PriorityClassName

	policy, err := compileSchedulingPolicy(cfg.Policy)
	if err != nil {
		return schedulingv1beta1.PodGroupTemplate{}, err
	}
	tmpl.SchedulingPolicy = policy

	if cfg.Constraints != nil && len(cfg.Constraints.Topology) > 0 {
		topology := make([]schedulingv1beta1.TopologyConstraint, len(cfg.Constraints.Topology))
		for i := range cfg.Constraints.Topology {
			topology[i] = schedulingv1beta1.TopologyConstraint{
				Key: cfg.Constraints.Topology[i].Key,
			}
		}
		tmpl.SchedulingConstraints = &schedulingv1beta1.PodGroupSchedulingConstraints{
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
func compileSchedulingPolicy(policy *SchedulingPolicy) (schedulingv1beta1.PodGroupSchedulingPolicy, error) {
	basic := schedulingv1beta1.PodGroupSchedulingPolicy{
		Basic: &schedulingv1beta1.BasicSchedulingPolicy{},
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
			return schedulingv1beta1.PodGroupSchedulingPolicy{}, fmt.Errorf("gang scheduling requires minCount to be set after resolution")
		}
		return schedulingv1beta1.PodGroupSchedulingPolicy{
			Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: *policy.Gang.MinCount},
		}, nil
	default:
		// nil policy, Basic only, or an empty policy all resolve to Basic.
		return basic, nil
	}
}

// compileCompositePodGroupTemplate resolves a composite item's config, runs its
// callbacks, and compiles it into a CompositePodGroupTemplate, recursing into
// each child (composite children become nested CompositePodGroupTemplates, leaf
// children become PodGroupTemplates).
func compileCompositePodGroupTemplate(item *WorkloadItem) (schedulingv1beta1.CompositePodGroupTemplate, error) {
	resolved := resolveSchedulingConfig(item)

	policy, err := compileCompositeSchedulingPolicy(resolved.Policy)
	if err != nil {
		return schedulingv1beta1.CompositePodGroupTemplate{}, err
	}

	tmpl := schedulingv1beta1.CompositePodGroupTemplate{
		Name:              item.Name,
		SchedulingPolicy:  policy,
		PriorityClassName: resolved.PriorityClassName,
	}

	if resolved.Constraints != nil && len(resolved.Constraints.Topology) > 0 {
		topology := make([]schedulingv1beta1.TopologyConstraint, len(resolved.Constraints.Topology))
		for i := range resolved.Constraints.Topology {
			topology[i] = schedulingv1beta1.TopologyConstraint{
				Key: resolved.Constraints.Topology[i].Key,
			}
		}
		tmpl.SchedulingConstraints = &schedulingv1beta1.CompositePodGroupSchedulingConstraints{
			Topology: topology,
		}
	}

	if resolved.DisruptionMode != nil {
		tmpl.DisruptionMode = compileCompositeDisruptionMode(resolved.DisruptionMode)
	}

	for _, child := range item.Children {
		if child == nil {
			return schedulingv1beta1.CompositePodGroupTemplate{}, fmt.Errorf("composite workload item %q has a nil child", item.Name)
		}
		if isComposite(child) {
			childTmpl, err := compileCompositePodGroupTemplate(child)
			if err != nil {
				return schedulingv1beta1.CompositePodGroupTemplate{}, err
			}
			tmpl.CompositePodGroupTemplates = append(tmpl.CompositePodGroupTemplates, childTmpl)
		} else {
			pgTmpl, err := compilePodGroupTemplate(child)
			if err != nil {
				return schedulingv1beta1.CompositePodGroupTemplate{}, err
			}
			tmpl.PodGroupTemplates = append(tmpl.PodGroupTemplates, pgTmpl)
		}
	}

	return tmpl, nil
}

// compileCompositeSchedulingPolicy maps the IR policy onto the composite API
// policy. A nil policy resolves to Basic; Gang requires a MinCount resolved
// beforehand so it can populate the required MinGroupCount field (the IR reuses
// GangSchedulingPolicy.MinCount to carry the composite minGroupCount value). The
// minGroupCount>=1 bound is a structural constraint enforced by declarative
// validation, so it is not repeated here.
func compileCompositeSchedulingPolicy(policy *SchedulingPolicy) (schedulingv1beta1.CompositePodGroupSchedulingPolicy, error) {
	basic := schedulingv1beta1.CompositePodGroupSchedulingPolicy{
		Basic: &schedulingv1beta1.CompositeBasicSchedulingPolicy{},
	}

	if policy == nil {
		return basic, nil
	}

	// The basic/gang union is enforced by declarative validation when the
	// compiled Workload is submitted, only the compile-time invariant that a
	// resolved gang policy carries a minGroupCount value is checked here.
	switch {
	case policy.Gang != nil:
		if policy.Gang.MinCount == nil {
			return schedulingv1beta1.CompositePodGroupSchedulingPolicy{},
				fmt.Errorf("composite gang scheduling requires minGroupCount to be set after resolution")
		}
		return schedulingv1beta1.CompositePodGroupSchedulingPolicy{
			Gang: &schedulingv1beta1.CompositeGangSchedulingPolicy{
				MinGroupCount: *policy.Gang.MinCount,
			},
		}, nil
	default:
		// nil policy, Basic only, or an empty policy all resolve to Basic.
		return basic, nil
	}
}

// compileDisruptionMode maps the IR disruption mode onto the leaf API mode. The
// exactly-one-of union is enforced by declarative validation when the compiled
// Workload is submitted, so a mode with neither member set compiles to nil.
func compileDisruptionMode(dm *DisruptionMode) *schedulingv1beta1.DisruptionMode {
	switch {
	case dm.All != nil:
		return &schedulingv1beta1.DisruptionMode{All: &schedulingv1beta1.AllDisruptionMode{}}
	case dm.Single != nil:
		return &schedulingv1beta1.DisruptionMode{Single: &schedulingv1beta1.SingleDisruptionMode{}}
	default:
		return nil
	}
}

func compileResourceClaims(claims []ResourceClaim) []schedulingv1beta1.PodGroupResourceClaim {
	result := make([]schedulingv1beta1.PodGroupResourceClaim, len(claims))
	for i := range claims {
		result[i] = schedulingv1beta1.PodGroupResourceClaim{
			Name:                      claims[i].Name,
			ResourceClaimName:         claims[i].ResourceClaimName,
			ResourceClaimTemplateName: claims[i].ResourceClaimTemplateName,
		}
	}
	return result
}

// compileCompositeDisruptionMode maps the IR disruption mode onto the composite
// API mode. The exactly-one-of union is enforced by declarative validation when
// the compiled Workload is submitted, so a mode with neither member set compiles
// to nil.
func compileCompositeDisruptionMode(dm *DisruptionMode) *schedulingv1beta1.CompositeDisruptionMode {
	switch {
	case dm.All != nil:
		return &schedulingv1beta1.CompositeDisruptionMode{
			All: &schedulingv1beta1.AllCompositeDisruptionMode{},
		}
	case dm.Single != nil:
		return &schedulingv1beta1.CompositeDisruptionMode{
			Single: &schedulingv1beta1.SingleCompositeDisruptionMode{},
		}
	default:
		return nil
	}
}
