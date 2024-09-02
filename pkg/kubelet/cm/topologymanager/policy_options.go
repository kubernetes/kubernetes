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

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

const (
	PreferClosestNUMANodes string = "prefer-closest-numa-nodes"
	MaxAllowableNUMANodes  string = "max-allowable-numa-nodes"
)

var (
	alphaOptions = sets.New[string]()
	betaOptions  = sets.New[string](
		PreferClosestNUMANodes,
		MaxAllowableNUMANodes,
	)
	stableOptions = sets.New[string]()
)

func CheckPolicyOptionAvailable(option string) error {
	if !alphaOptions.Has(option) && !betaOptions.Has(option) && !stableOptions.Has(option) {
		return fmt.Errorf("unknown Topology Manager Policy option: %q", option)
	}

	if alphaOptions.Has(option) && !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.TopologyManagerPolicyAlphaOptions) {
		return fmt.Errorf("Topology Manager Policy Alpha-level Options not enabled, but option %q provided", option)
	}

	if betaOptions.Has(option) && !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.TopologyManagerPolicyBetaOptions) {
		return fmt.Errorf("Topology Manager Policy Beta-level Options not enabled, but option %q provided", option)
	}

	return nil
}

type PolicyOptions struct {
	PreferClosestNUMA     bool
	MaxAllowableNUMANodes int
}

func NewPolicyOptions(policyOptions map[string]string) (PolicyOptions, error) {
	opts := PolicyOptions{
		// Set MaxAllowableNUMANodes to the default. This will be overwritten
		// if the user has specified a policy option for MaxAllowableNUMANodes.
		MaxAllowableNUMANodes: defaultMaxAllowableNUMANodes,
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
				klog.InfoS("WARNING: the value of max-allowable-numa-nodes is more than the default recommended value", "max-allowable-numa-nodes", optValue, "defaultMaxAllowableNUMANodes", defaultMaxAllowableNUMANodes)
			}
			opts.MaxAllowableNUMANodes = optValue
		default:
			// this should never be reached, we already detect unknown options,
			// but we keep it as further safety.
			return opts, fmt.Errorf("unsupported topologymanager option: %q (%s)", name, value)
		}
	}
	return opts, nil
}
