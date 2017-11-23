/*
Copyright 2014 The Kubernetes Authors.

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

package algorithmprovider

import (
	"testing"

	"github.com/stretchr/testify/require"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
)

var (
	algorithmProviderNames = []string{
		factory.DefaultProvider,
	}
)

func TestDefaultConfigExists(t *testing.T) {
	p, err := factory.GetAlgorithmProvider(factory.DefaultProvider)
	require.NoError(t, err, "error retrieving default provider: %v", err)
	require.NotNil(t, p, "algorithm provider config should not be nil")
	require.NotEqual(t, 0, len(p.FitPredicateKeys), "default algorithm provider shouldn't have 0 fit predicates")
}

func TestAlgorithmProviders(t *testing.T) {
	for _, pn := range algorithmProviderNames {
		p, err := factory.GetAlgorithmProvider(pn)
		require.NoError(t, err, "error retrieving '%s' provider: %v", pn, err)
		require.NotEqual(t, 0, len(p.PriorityFunctionKeys), "%s algorithm provider shouldn't have 0 priority functions", pn)
		for _, pf := range p.PriorityFunctionKeys.List() {
			require.True(t, factory.IsPriorityFunctionRegistered(pf), "priority function %s is not registered but is used in the %s algorithm provider", pf, pn)
		}
		for _, fp := range p.FitPredicateKeys.List() {
			require.True(t, factory.IsFitPredicateRegistered(fp), "fit predicate %s is not registered but is used in the %s algorithm provider", fp, pn)
		}
	}
}

func TestApplyFeatureGates(t *testing.T) {
	for _, pn := range algorithmProviderNames {
		p, err := factory.GetAlgorithmProvider(pn)
		require.NoError(t, err, "Error retrieving '%s' provider: %v", pn, err)
		require.Contains(t, p.FitPredicateKeys, "CheckNodeCondition", "Failed to find predicate: 'CheckNodeCondition'")
		require.Contains(t, p.FitPredicateKeys, "PodToleratesNodeTaints", "Failed to find predicate: 'PodToleratesNodeTaints'")
	}

	// Apply features for algorithm providers.
	utilfeature.DefaultFeatureGate.Set("TaintNodesByCondition=True")

	ApplyFeatureGates()

	for _, pn := range algorithmProviderNames {
		p, err := factory.GetAlgorithmProvider(pn)
		require.NoError(t, err, "Error retrieving '%s' provider: %v", pn, err)
		require.Contains(t, p.FitPredicateKeys, "PodToleratesNodeTaints", "Failed to find predicate: 'PodToleratesNodeTaints'")
		require.NotContains(t, p.FitPredicateKeys, "CheckNodeCondition", "Unexpected predicate: 'CheckNodeCondition'")
	}
}
