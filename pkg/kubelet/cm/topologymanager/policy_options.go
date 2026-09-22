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

// defaultNUMAScoreWeight is the weight given to a resource the weight string
// does not name. It is also the smallest weight an operator can explicitly
// assign, so naming a resource can only raise its influence relative to the
// others, never lower it. Dropping a resource from scoring therefore requires
// an explicit weight of 0.
const defaultNUMAScoreWeight = 1

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
	// when hint scores are aggregated. It is nil unless the user asked for
	// weights, and a resource missing from it sits at defaultNUMAScoreWeight.
	NUMAScoreWeights map[string]int
}

// parseNUMAScoreWeights turns the comma-separated resource=weight form of the
// NUMAScoreWeights policy option into the map the aggregation consumes, e.g.
// "cpu=3,memory=1,nvidia.com/gpu=6" into
// map[string]int{"cpu": 3, "memory": 1, "nvidia.com/gpu": 6}.
//
// Resource names are not checked against the providers present on the node.
// The set of providers that can contribute a score is a property of the pod
// being admitted rather than of the node, so a name matching nothing on the
// node is indistinguishable from one a given pod simply does not request.
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
		weights[resource] = weight
	}
	return weights, nil
}

// validateNUMAScoreWeights rejects weights outside the accepted range. A weight
// of 0 is valid and excludes the resource from scoring altogether.
//
// There is deliberately no check on the sum: the aggregation divides by the
// total applicable weight, so only the ratios between weights matter and
// "cpu=30,memory=10" behaves exactly like "cpu=3,memory=1".
func validateNUMAScoreWeights(weights map[string]int) error {
	for _, resource := range sets.List(sets.KeySet(weights)) {
		if weight := weights[resource]; weight < minNUMAScoreWeight || weight > maxNUMAScoreWeight {
			return fmt.Errorf("weight for %q must be in range [%d, %d], got %d", resource, minNUMAScoreWeight, maxNUMAScoreWeight, weight)
		}
	}
	return nil
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
			// an empty value means the same as "none", so the option can be
			// neutralized without removing the key.
			if value == "" {
				value = NUMAAllocationStrategyNone
			}
			if !numaAllocationStrategies.Has(value) {
				return opts, fmt.Errorf("bad value for option %q: %q must be one of %q", name, value, sets.List(numaAllocationStrategies))
			}
			opts.NUMAAllocationStrategy = value
		case NUMAScoreWeights:
			weights, err := parseNUMAScoreWeights(value)
			if err != nil {
				return opts, fmt.Errorf("bad value for option %q: %w", name, err)
			}
			if err := validateNUMAScoreWeights(weights); err != nil {
				return opts, fmt.Errorf("bad value for option %q: %w", name, err)
			}
			opts.NUMAScoreWeights = weights
		default:
			// this should never be reached, we already detect unknown options,
			// but we keep it as further safety.
			return opts, fmt.Errorf("unsupported topologymanager option: %q (%s)", name, value)
		}
	}
	return opts, nil
}
