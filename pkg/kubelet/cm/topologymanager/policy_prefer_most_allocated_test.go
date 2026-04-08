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

package topologymanager

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// fakeNUMAScorer always resolves a 0-vs-1 single-NUMA tie in favor of the configured NUMA id.
type fakeNUMAScorer int

func (p fakeNUMAScorer) Score(current, candidate bitmask.BitMask) NUMAScoringResult {
	t := int(p)
	cb := candidate.GetBits()
	ob := current.GetBits()
	if len(cb) != 1 || len(ob) != 1 {
		return NUMAScoringResult{}
	}
	if cb[0] == t && ob[0] != t {
		return NUMAScoringResult{PreferCandidate: true, Ok: true}
	}
	if ob[0] == t && cb[0] != t {
		return NUMAScoringResult{PreferCandidate: false, Ok: true}
	}
	return NUMAScoringResult{}
}

func TestSingleNumaPolicyPreferMostAllocatedScoring(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	numaInfo := commonNUMAInfoTwoNodes()
	p := &singleNumaNodePolicy{
		numaInfo: numaInfo,
		opts:     PolicyOptions{PreferMostAllocatedNUMANode: true},
	}
	p.setNUMAScorer(fakeNUMAScorer(1))

	providersHints := []map[string][]TopologyHint{
		{
			"resource1": {
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
				{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
				{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			},
		},
		nil,
	}
	actual, admit := p.Merge(logger, providersHints)
	if !admit {
		t.Fatalf("expected admit true, got false for hint %#v", actual)
	}
	want := TopologyHint{NUMANodeAffinity: NewTestBitMask(1), Preferred: true}
	if !reflect.DeepEqual(actual, want) {
		t.Errorf("got %#v want %#v", actual, want)
	}
}

func TestSingleNumaPolicyPreferMostAllocatedDefaultNarrowest(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	numaInfo := commonNUMAInfoTwoNodes()
	p := &singleNumaNodePolicy{
		numaInfo: numaInfo,
		opts:     PolicyOptions{PreferMostAllocatedNUMANode: true},
	}

	providersHints := []map[string][]TopologyHint{
		{
			"resource1": {
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
				{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
				{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			},
		},
		nil,
	}
	actual, admit := p.Merge(logger, providersHints)
	if !admit {
		t.Fatalf("expected admit true, got false for hint %#v", actual)
	}
	want := TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true}
	if !reflect.DeepEqual(actual, want) {
		t.Errorf("got %#v want %#v", actual, want)
	}
}
