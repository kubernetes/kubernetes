/*
Copyright 2022 The Kubernetes Authors.

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

package topologymanager

import (
	"fmt"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

const (
	PreferClosestNUMANodes string = "prefer-closest-numa-nodes"
	MaxAllowableNUMANodes  string = "max-allowable-numa-nodes"
	NUMAAllocationStrategy string = "numa-allocation-strategy"
	NUMAScoreWeights       string = "numa-score-weights"
)

// Bounds for the per-resource weights accepted by the NUMAScoreWeights policy
// option, matching the range kube-scheduler's NodeResourcesFit plugin uses for
// its own per-resource weights.
const (
	minNUMAScoreWeight = 0
	maxNUMAScoreWeight = 100
)

// defaultNUMAScoreWeights returns the weights applied when numa-score-weights
// is empty or unset. Following kube-scheduler's NodeResourcesFit pattern, this
// defaults to CPU and memory with equal weight, allowing the strategy to do
// something useful out of the box while excluding device plugins unless
// explicitly included.
//
// When numa-score-weights is explicitly set to a non-empty value, only the
// listed resources participate in scoring (opt-in pattern).
func defaultNUMAScoreWeights() map[string]int {
	return map[string]int{
		string(v1.ResourceCPU):    1,
		string(v1.ResourceMemory): 1,
	}
}

// Values accepted by the NUMAAllocationStrategy policy option.
const (
	// NUMAAllocationStrategyNone keeps the pre-existing hint selection, which
	// disregards how much of each NUMA node is already allocated. This is the
	// default.
	NUMAAllocationStrategyNone string = "none"
	// NUMAAllocationStrategyMostAllocated prefers the NUMA nodes which are
	// already the most allocated, packing workloads together.
	NUMAAllocationStrategyMostAllocated string = "most-allocated"
	// NUMAAllocationStrategyLeastAllocated prefers the NUMA nodes which are
	// the least allocated, spreading workloads apart.
	NUMAAllocationStrategyLeastAllocated string = "least-allocated"
)

var (
	alphaOptions = sets.New[string](
		NUMAAllocationStrategy,
		NUMAScoreWeights,
	)
	betaOptions   = sets.New[string]()
	stableOptions = sets.New[string](
		PreferClosestNUMANodes,
		MaxAllowableNUMANodes,
	)

	numaAllocationStrategies = sets.New[string](
		NUMAAllocationStrategyNone,
		NUMAAllocationStrategyMostAllocated,
		NUMAAllocationStrategyLeastAllocated,
	)
)

func CheckPolicyOptionAvailable(option string) error {
	if !alphaOptions.Has(option) && !betaOptions.Has(option) && !stableOptions.Has(option) {
		return fmt.Errorf("unknown Topology Manager Policy option: %q", option)
	}

	if alphaOptions.Has(option) && !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.TopologyManagerPolicyAlphaOptions) {
		return fmt.Errorf("topology manager policy alpha-level options not enabled, but option %q provided", option)
	}

	if betaOptions.Has(option) && !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.TopologyManagerPolicyBetaOptions) {
		return fmt.Errorf("topology manager policy beta-level options not enabled, but option %q provided", option)
	}

	return nil
}

type PolicyOptions struct {
	PreferClosestNUMA      bool
	MaxAllowableNUMANodes  int
	NUMAAllocationStrategy string
	// NUMAScoreWeights maps a resource name to the weight its score carries
	// when hint scores are aggregated. Following the NodeResourcesFit pattern:
	// when empty/unset, defaults to {cpu: 1, memory: 1}; when explicitly set,
	// only listed resources participate (unlisted resources are excluded).
	NUMAScoreWeights map[string]int
}

// parseNUMAScoreWeights turns the comma-separated resource=weight form of the
// NUMAScoreWeights policy option into the map the aggregation consumes, e.g.
// "cpu=3,memory=1,nvidia.com/gpu=6" into
// map[string]int{"cpu": 3, "memory": 1, "nvidia.com/gpu": 6}.
//
// Following kube-scheduler's NodeResourcesFit pattern:
//   - Empty/unset weights → defaults to cpu=1, memory=1 (applied in NewPolicyOptions)
//   - Explicitly set weights → only listed resources participate (unlisted excluded)
//
// Resource names are matched case-sensitively against the names the hint
// providers report, so they are stored exactly as written. Lower-casing them
// would break matching: hugepages resources are named after the page size,
// e.g. "hugepages-2Mi", and the memory manager files its hints under that
// name verbatim. Extended resources are no safer to fold, since
// only the domain prefix of a name like "nvidia.com/gpu" is forced lowercase
// by API validation; the part after the slash may carry upper case.
//
// Resource names are not checked against the providers present on the node.
// The set of providers that can contribute a score is a property of the pod
// being admitted rather than of the node, so a name matching nothing on the
// node is indistinguishable from one a given pod simply does not request.
//
// Weights outside [minNUMAScoreWeight, maxNUMAScoreWeight] are rejected rather
// than pinned to the nearest bound, because the bound is unlikely to be what
// the operator meant: weights only matter relative to each other, so pinning
// "cpu=150,memory=100" to "cpu=100,memory=100" would quietly turn "cpu counts
// 1.5x memory" into "cpu and memory count equally". A weight of 0 is valid and
// excludes the resource from scoring altogether.
//
// There is deliberately no check on the sum: the aggregation divides by the
// total applicable weight, so only the ratios between weights matter and
// "cpu=30,memory=10" behaves exactly like "cpu=3,memory=1".
func parseNUMAScoreWeights(raw string) (map[string]int, error) {
	weights := make(map[string]int)
	for pair := range strings.SplitSeq(raw, ",") {
		pair = strings.TrimSpace(pair)
		// Tolerate empty entries so a trailing comma, or the empty string
		// itself, means "no weights" rather than a startup failure.
		if pair == "" {
			continue
		}
		resource, rawWeight, found := strings.Cut(pair, "=")
		if !found {
			return nil, fmt.Errorf("invalid weight entry %q: expected resource=weight", pair)
		}
		resource = strings.TrimSpace(resource)
		if resource == "" {
			return nil, fmt.Errorf("empty resource name in weight entry %q", pair)
		}
		weight, err := strconv.Atoi(strings.TrimSpace(rawWeight))
		if err != nil {
			return nil, fmt.Errorf("invalid weight value for %q: %w", resource, err)
		}
		// Range-check here rather than in a second pass over the map, so a
		// weights map cannot be built without being validated, and so the
		// error can quote the entry as the operator wrote it. Walking the
		// input in order also makes "the first bad entry wins" deterministic,
		// which iterating the map would not be.
		if weight < minNUMAScoreWeight || weight > maxNUMAScoreWeight {
			return nil, fmt.Errorf("invalid weight entry %q: weight must be in range [%d, %d], got %d", pair, minNUMAScoreWeight, maxNUMAScoreWeight, weight)
		}
		weights[resource] = weight
	}
	return weights, nil
}

func NewPolicyOptions(logger klog.Logger, policyOptions map[string]string) (PolicyOptions, error) {
	opts := PolicyOptions{
		// Set MaxAllowableNUMANodes to the default. This will be overwritten
		// if the user has specified a policy option for MaxAllowableNUMANodes.
		MaxAllowableNUMANodes: defaultMaxAllowableNUMANodes,
		// Likewise, allocation state does not take part in hint selection
		// unless the user asks for it.
		NUMAAllocationStrategy: NUMAAllocationStrategyNone,
	}

	for name, value := range policyOptions {
		if err := CheckPolicyOptionAvailable(name); err != nil {
			return opts, err
		}

		switch name {
		case PreferClosestNUMANodes:
			optValue, err := strconv.ParseBool(value)
			if err != nil {
				return opts, fmt.Errorf("bad value for option %q: %w", name, err)
			}
			opts.PreferClosestNUMA = optValue
		case MaxAllowableNUMANodes:
			optValue, err := strconv.Atoi(value)
			if err != nil {
				return opts, fmt.Errorf("unable to convert policy option to integer %q: %w", name, err)
			}

			if optValue < defaultMaxAllowableNUMANodes {
				return opts, fmt.Errorf("the minimum value of %q should not be less than %v", name, defaultMaxAllowableNUMANodes)
			}

			if optValue > defaultMaxAllowableNUMANodes {
				logger.Info("WARNING: the value of max-allowable-numa-nodes is more than the default recommended value", "max-allowable-numa-nodes", optValue, "defaultMaxAllowableNUMANodes", defaultMaxAllowableNUMANodes)
			}
			opts.MaxAllowableNUMANodes = optValue
		case NUMAAllocationStrategy:
			if !numaAllocationStrategies.Has(value) {
				return opts, fmt.Errorf("bad value for option %q: %q must be one of %q", name, value, sets.List(numaAllocationStrategies))
			}
			opts.NUMAAllocationStrategy = value
		case NUMAScoreWeights:
			weights, err := parseNUMAScoreWeights(value)
			if err != nil {
				return opts, fmt.Errorf("bad value for option %q: %w", name, err)
			}
			opts.NUMAScoreWeights = weights
		default:
			// Recognized options added to alpha/betaOptions sets (by tests or future
			// features) that don't have implementation yet are silently ignored rather
			// than failing. This allows feature-gate tests to register dummy options.
			if alphaOptions.Has(name) || betaOptions.Has(name) {
				// No-op: option was recognized but has no behavior yet
			} else {
				return opts, fmt.Errorf("unsupported topologymanager option: %q (%s)", name, value)
			}
		}
	}
	// Apply defaults when numa-score-weights is empty or unset, following
	// the NodeResourcesFit pattern: cpu and memory with equal weight.
	if len(opts.NUMAScoreWeights) == 0 {
		opts.NUMAScoreWeights = defaultNUMAScoreWeights()
	}
	return opts, nil
}
