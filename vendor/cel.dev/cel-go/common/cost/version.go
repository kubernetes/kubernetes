// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cost

import (
	"maps"
	"math"
	"slices"
	"sync"
)

const defaultModelVersion uint32 = math.MaxUint32

// ModelOption configures how OverloadModels, cost estimators, and cost trackers are built and
// evaluated.
//
// A ModelOption implements both Option (for Cost) and TrackerOption (for NewTracker), and may also
// be passed when compiling models directly via FunctionEstimatorWithOptions,
// FunctionTrackerWithOptions, StandardOverloadEstimatorsWithOptions, and
// StandardOverloadTrackersWithOptions.
type ModelOption func(*modelOptions)

// modelRules are the sizing strategy and model revision under which an estimator or a tracker
// prices calls. Both are fixed once the estimator or tracker has been configured.
type modelRules struct {
	sizingStrategy SizingStrategy
	modelVersion   uint32
}

func defaultModelRules() modelRules {
	return modelRules{sizingStrategy: defaultSizing, modelVersion: defaultModelVersion}
}

// isDefault reports whether the rules are those the package-level and per-model-set caches are
// compiled under, so that the cached functions may be shared rather than recompiled.
func (r modelRules) isDefault() bool {
	return r.sizingStrategy == defaultSizing && r.modelVersion == defaultModelVersion
}

func (r modelRules) modelOptions() []ModelOption {
	return []ModelOption{WithSizingStrategy(r.sizingStrategy), ModelVersion(r.modelVersion)}
}

// override replaces the rules the option sets, leaving the others in place.
func (r *modelRules) override(opt ModelOption) {
	o := newModelOptions(opt)
	if o.hasStrategy {
		r.sizingStrategy = o.strategy
	}
	if o.hasVersion {
		r.modelVersion = o.version
	}
}

// compiledModelSet is the immutable product of a Model option: the models, the result of
// validating them, and, built on first use and shared by every estimator and tracker configured
// with the option, the functions the models compile to under the default rules.
type compiledModelSet struct {
	models            []OverloadModel
	err               error
	estOnce           sync.Once
	defaultEstimators map[string]FunctionEstimator
	trkOnce           sync.Once
	defaultTrackers   map[string]FunctionTracker
}

func newCompiledModelSet(models []OverloadModel) *compiledModelSet {
	return &compiledModelSet{
		models: models,
		err:    ValidateOverloadModels(models...),
	}
}

func (cm *compiledModelSet) estimators() map[string]FunctionEstimator {
	cm.estOnce.Do(func() {
		var t overloadTable[FunctionEstimator]
		t.compile(cm.models, estimatorCompiler())
		cm.defaultEstimators = t.fns
	})
	return cm.defaultEstimators
}

func (cm *compiledModelSet) trackers() map[string]FunctionTracker {
	cm.trkOnce.Do(func() {
		var t overloadTable[FunctionTracker]
		t.compile(cm.models, trackerCompiler())
		cm.defaultTrackers = t.fns
	})
	return cm.defaultTrackers
}

// overloadTable maps overload IDs to the FunctionEstimators or FunctionTrackers which price them.
//
// Options populate a table while an estimator or tracker is configured; it is read-only after.
// A table configured with a single model set under the default rules refers to the set's cached
// functions rather than copying them, and is copied before its first write.
type overloadTable[F any] struct {
	fns    map[string]F
	shared bool
}

func (t *overloadTable[F]) mutable() map[string]F {
	if t.shared {
		t.fns, t.shared = maps.Clone(t.fns), false
	} else if t.fns == nil {
		t.fns = map[string]F{}
	}
	return t.fns
}

// compile binds each model's function in order, so a later model replaces an earlier one with the
// same ID.
func (t *overloadTable[F]) compile(models []OverloadModel, compile func(OverloadModel) F) {
	fns := t.mutable()
	for _, m := range models {
		fns[m.ID] = compile(m)
	}
}

// addModelSet binds the functions the models of cm compile to under the given rules.
func (t *overloadTable[F]) addModelSet(cm *compiledModelSet, rules modelRules,
	cached func() map[string]F, compile func(...ModelOption) func(OverloadModel) F) error {
	if cm.err != nil {
		return cm.err
	}
	if !rules.isDefault() {
		t.compile(cm.models, compile(rules.modelOptions()...))
		return nil
	}
	if t.fns == nil {
		t.fns, t.shared = cached(), true
		return nil
	}
	t.compile(cm.models, compile())
	return nil
}

func estimatorCompiler(opts ...ModelOption) func(OverloadModel) FunctionEstimator {
	return func(m OverloadModel) FunctionEstimator {
		return m.FunctionEstimatorWithOptions(opts...)
	}
}

func trackerCompiler(opts ...ModelOption) func(OverloadModel) FunctionTracker {
	return func(m OverloadModel) FunctionTracker {
		return m.FunctionTrackerWithOptions(opts...)
	}
}

// modelOptions is the resolved configuration for compiling an OverloadModel or configuring an
// estimator or tracker.
type modelOptions struct {
	strategy    SizingStrategy
	hasStrategy bool
	version     uint32
	hasVersion  bool
	modelSets   []*compiledModelSet
}

// A ModelOption is applied in two passes by Cost and NewTracker. The first, over every ModelOption
// before any other option, settles the rules, so that models are compiled under the final rules
// whatever the option order. The second registers the model sets in order with the other options,
// so that a later registration of an overload ID overrides an earlier one.

func (opt ModelOption) applyCoster(c *coster) error {
	for _, cm := range newModelOptions(opt).modelSets {
		if err := c.overloadEstimators.addModelSet(cm, c.modelRules, cm.estimators, estimatorCompiler); err != nil {
			return err
		}
	}
	return nil
}

func (opt ModelOption) applyTracker(b *trackerBuilder) error {
	for _, cm := range newModelOptions(opt).modelSets {
		if err := b.overloadTrackers.addModelSet(cm, b.modelRules, cm.trackers, trackerCompiler); err != nil {
			return err
		}
	}
	return nil
}

// newModelOptions resolves the supplied options over the defaults.
func newModelOptions(opts ...ModelOption) *modelOptions {
	resolved := &modelOptions{
		strategy: defaultSizing,
		version:  defaultModelVersion,
	}
	for _, opt := range opts {
		opt(resolved)
	}
	return resolved
}

// Model registers one or more OverloadModels for both cost estimation and runtime cost tracking.
//
// Models are validated when the option is applied to Cost or NewTracker and resolved after all
// options have run, so they observe the WithSizingStrategy and ModelVersion configured on the call
// regardless of option order.
func Model(models ...OverloadModel) ModelOption {
	cm := newCompiledModelSet(slices.Clone(models))
	return func(o *modelOptions) {
		o.modelSets = append(o.modelSets, cm)
	}
}

// WithSizingStrategy sets the SizingStrategy used to resolve sizes during both cost estimation and
// runtime cost tracking.
//
// A nil strategy is ignored, leaving the default in place.
func WithSizingStrategy(strategy SizingStrategy) ModelOption {
	return func(o *modelOptions) {
		o.hasStrategy = true
		if strategy != nil {
			o.strategy = strategy
		}
	}
}

// ModelVersion pins the revision of the cost model's estimation and runtime tracking rules.
//
// If this option is not set, the latest cost model version is used. Revisions exist so that
// corrections to the model can land without silently moving numbers that existing deployments
// budget against. A caller holding estimates recorded under an earlier release can pin to the
// revision those estimates were produced under, review the delta at leisure, and adopt the
// correction deliberately.
//
// Revisions apply to both estimation and runtime tracking, so that an estimate and the charge it
// predicts are always computed by the same rules. Pinning a version therefore holds both stable.
//
// Caveat: Pinning to version 0 preserves legacy behavior from v0.32.0 and earlier, which contains
// known soundness gaps where static estimates can underestimate actual runtime cost (for example,
// when a caller's Estimator overrides computed loop variable sizes via bare "@items" paths, when
// selecting or indexing into computed containers, or when evaluating non-literal integer bounds).
// Callers are strongly encouraged to switch to ModelVersion(1) or higher (or leave the version
// unset to use the latest revision) for improved soundness between estimated bounds and tracked
// runtime cost.
//
// Version 1:
//   - Min() is computed as the exact minimum of its operands' intervals, where v0.32.0 reported a
//     lower bound of 1 for any interval whose upper bound is non-zero. The change moves only the
//     lower bound of an estimate, never a charge, so it applies at every revision, including when
//     pinned to version 0.
//   - A value inside an optional keeps its size. optional.of, optional.ofNonZeroValue and value()
//     report the size they wrap or unwrap, the two `or` forms report the union of the receiver and
//     the alternative, and the optional index forms record the same `@items` and `@values` field
//     path their non-optional counterparts do.
//   - Optional field and index qualifiers (`m.?k`, `l[?0]`) charge 1 at runtime even when the
//     field or index is absent, matching static estimation and non-optional qualification.
//   - Selecting or indexing into a computed expression (such as `[1, 2][0]` or `{"a": 1}.a`)
//     charges the attribute traversal cost of 1 during estimation to match runtime tracking.
//   - Comprehension loop variables over computed (pathless) ranges retain their propagated element
//     or key sizes rather than recording bare `@items`, `@indices`, `@keys`, or `@values` paths
//     that a caller's Estimator could override with a smaller bound.
//   - The default sizing strategy skips `@items`, `@keys`, and `@values` child size hints when a
//     container's maximum size is 0.
//   - Extension estimators in ext/lists (distinct, sort, sortBy, slice, range, flatten),
//     ext/strings (join, split, substring, replace), and ext/regex (extract, extractAll, replace)
//     align their Min/Max cost and result-size bounds with their runtime cost trackers.
//   - Scale applies its factor as the simplest rational with the same float64 value (0.1 is 1/10)
//     and computes the product exactly; it keeps an unbounded quantity unbounded; and when a model
//     is compiled, factors are moved outward so that each product is rounded once. The standard
//     contains and matches costs become ⌈0.01·m·n⌉ and ⌈0.025·(m+1)·n⌉.
//   - Ranged never reports a lower bound above its upper bound.
//   - Combinators keep an unknown element or key size unknown, treating constants and integer
//     values as scalars; Sub carries only its minuend's metadata.
//   - The result size is estimated first and is available to the cost through Result().
//   - A non-literal IntArg or IntTarget estimates as unbounded rather than as its default.
//   - List and Map report the size the sizing strategy assigns to the container they describe.
//     Under the aggregate strategy, a comprehension iterates once per element of its range rather
//     than once per unit of content size, and sizes its result as an aggregate container.
//   - Runtime tracking honors the revision, as estimation does.
//   - Every ext overload with a cost estimator is priced by a model, which charges by Length what
//     its tracker charged and states result sizes in the units of the sizing strategy; flatten(depth)
//     follows the strategy in its estimator. An Arg or Target typed as a value without elements,
//     such as a string, carries no unknown element size.
//   - The value bound by cel.bind counts toward the minimum cost only when the result must read
//     it, since the binding is lazy.
//   - A list or map literal's minimum size counts only its required elements or entries, and an
//     optional element written as optional.none() or optional.of(x) contributes nothing or the
//     size of x.
func ModelVersion(version uint32) ModelOption {
	return func(o *modelOptions) {
		o.hasVersion = true
		o.version = version
	}
}
