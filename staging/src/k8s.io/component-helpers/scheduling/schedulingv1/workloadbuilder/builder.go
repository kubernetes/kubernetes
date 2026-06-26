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
	"errors"
	"fmt"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
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
// a Workload.
func Build(root *WorkloadItem, opts BuildOptions) (*schedulingv1alpha3.Workload, error) {
	spec, err := compileWorkloadItemTree(root)
	if err != nil {
		return nil, fmt.Errorf("compiling workload %q: %w", opts.Name, err)
	}

	wl := &schedulingv1alpha3.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      opts.Name,
			Namespace: opts.Namespace,
		},
		Spec: spec,
	}

	if opts.Owner == nil {
		return nil, fmt.Errorf("BuildOptions.Owner is required: a Workload must be owned by its controller for garbage collection")
	}

	gv, err := schema.ParseGroupVersion(opts.Owner.APIVersion)
	if err != nil {
		return nil, fmt.Errorf("parsing owner APIVersion %q: %w", opts.Owner.APIVersion, err)
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
// returning an error if it cannot. It runs the same resolution, callbacks,
// and semantic checks as Build, but discards the compiled result.
//
// Validate returns a standard error rather than a field.ErrorList because
// its internal field paths (e.g., "spec.podGroupTemplates") rarely match a
// calling controller's API surface. Controllers should wrap this error with
// their own appropriate field.Path.
func Validate(root *WorkloadItem) error {
	_, err := compileWorkloadItemTree(root)
	return err
}

// compileWorkloadItemTree compiles the root item into the WorkloadSpec it
// produces. For now, only single-level (leaf) items are supported, so the
// returned spec carries exactly one PodGroupTemplate; Build fills in the object
// identity separately.
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

// compilePodGroupTemplate resolves a leaf item's config, then compiles it into a
// single PodGroupTemplate, accumulating any errors.
func compilePodGroupTemplate(item *WorkloadItem) (schedulingv1alpha3.PodGroupTemplate, error) {
	if item.Name == "" {
		return schedulingv1alpha3.PodGroupTemplate{}, fmt.Errorf("workload item name cannot be empty")
	}

	// resolveSchedulingConfig also runs the item's callbacks.
	item.ResolvedConfig = resolveSchedulingConfig(item)

	return buildLeafTemplate(item.Name, item.ResolvedConfig)
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
// accumulating semantic validation errors.
func buildLeafTemplate(name string, cfg *SchedulingConfig) (schedulingv1alpha3.PodGroupTemplate, error) {
	var errs []error
	tmpl := schedulingv1alpha3.PodGroupTemplate{Name: name}

	if cfg == nil {
		cfg = &SchedulingConfig{}
	}

	policy, err := compileSchedulingPolicy(cfg.Policy)
	if err != nil {
		errs = append(errs, err)
	}
	tmpl.SchedulingPolicy = policy

	if cfg.Constraints != nil && len(cfg.Constraints.Topology) > 0 {
		// Mirror the validation for maxItems=1 on PodGroupSchedulingConstraints.Topology
		// so we fail early.
		if len(cfg.Constraints.Topology) > 1 {
			errs = append(errs,
				fmt.Errorf("too many topology constraints: %d, max is 1",
					len(cfg.Constraints.Topology)))
		}
		topology := make([]schedulingv1alpha3.TopologyConstraint, len(cfg.Constraints.Topology))
		copy(topology, cfg.Constraints.Topology)
		tmpl.SchedulingConstraints = &schedulingv1alpha3.PodGroupSchedulingConstraints{
			Topology: topology,
		}
	}

	if cfg.DisruptionMode != nil {
		dm, err := compileDisruptionMode(cfg.DisruptionMode)
		if err != nil {
			errs = append(errs, err)
		} else {
			tmpl.DisruptionMode = dm
		}
	}

	if len(cfg.ResourceClaims) > 0 {
		if len(cfg.ResourceClaims) > schedulingv1alpha3.MaxPodGroupResourceClaims {
			errs = append(errs, fmt.Errorf("too many resource claims: %d, max is %d", len(cfg.ResourceClaims), schedulingv1alpha3.MaxPodGroupResourceClaims))
		}

		seenClaims := make(map[string]bool)
		for _, rc := range cfg.ResourceClaims {
			if rc.Name == "" {
				errs = append(errs, fmt.Errorf("resource claim name cannot be empty"))
			} else if seenClaims[rc.Name] {
				errs = append(errs, fmt.Errorf("duplicate resource claim name: %s", rc.Name))
			} else {
				seenClaims[rc.Name] = true
			}

			if (rc.ResourceClaimName == nil && rc.ResourceClaimTemplateName == nil) || (rc.ResourceClaimName != nil && rc.ResourceClaimTemplateName != nil) {
				errs = append(errs, fmt.Errorf("exactly one of resourceClaimName or resourceClaimTemplateName must be set"))
			}
		}

		tmpl.ResourceClaims = compileResourceClaims(cfg.ResourceClaims)
	}

	if len(errs) > 0 {
		return tmpl, errors.Join(errs...)
	}
	return tmpl, nil
}

// compileSchedulingPolicy maps the IR policy onto the API policy. A nil
// policy resolves to Basic, while Gang requires a positive MinCount
// resolved beforehand.
func compileSchedulingPolicy(policy *SchedulingPolicy) (schedulingv1alpha3.PodGroupSchedulingPolicy, error) {
	basic := schedulingv1alpha3.PodGroupSchedulingPolicy{
		Basic: &schedulingv1alpha3.BasicSchedulingPolicy{},
	}

	if policy == nil {
		return basic, nil
	}

	switch {
	case policy.Basic != nil && policy.Gang != nil:
		return schedulingv1alpha3.PodGroupSchedulingPolicy{}, fmt.Errorf("exactly one scheduling policy must be set")
	case policy.Gang != nil:
		if policy.Gang.MinCount == nil {
			return schedulingv1alpha3.PodGroupSchedulingPolicy{}, fmt.Errorf("gang scheduling requires minCount to be set after resolution")
		}
		if *policy.Gang.MinCount < 1 {
			return schedulingv1alpha3.PodGroupSchedulingPolicy{}, fmt.Errorf("gang minCount must be at least 1, got %d", *policy.Gang.MinCount)
		}
		return schedulingv1alpha3.PodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: *policy.Gang.MinCount},
		}, nil
	default:
		// nil policy, Basic only, or an empty policy all resolve to Basic.
		return basic, nil
	}
}

func compileDisruptionMode(dm *DisruptionMode) (*schedulingv1alpha3.DisruptionMode, error) {
	switch {
	case dm.Single != nil && dm.All != nil:
		return nil, fmt.Errorf("exactly one disruption mode must be set")
	case dm.All != nil:
		return &schedulingv1alpha3.DisruptionMode{All: &schedulingv1alpha3.AllDisruptionMode{}}, nil
	case dm.Single != nil:
		return &schedulingv1alpha3.DisruptionMode{Single: &schedulingv1alpha3.SingleDisruptionMode{}}, nil
	default:
		return nil, fmt.Errorf("exactly one disruption mode must be set")
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
