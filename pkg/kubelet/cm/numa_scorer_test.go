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

package cm

import (
	"testing"

	tmbitmask "k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

type stubNUMAScorer struct {
	curScore, candScore int64
}

func (s stubNUMAScorer) GetNUMAUtilizationScores(tmbitmask.BitMask, tmbitmask.BitMask) (int64, int64) {
	return s.curScore, s.candScore
}

func TestNUMAScorerAggregator(t *testing.T) {
	m0, err := tmbitmask.NewBitMask(0)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := tmbitmask.NewBitMask(1)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name      string
		cpu       stubNUMAScorer
		mem       stubNUMAScorer
		wantPick  bool
		wantOK    bool
		current   tmbitmask.BitMask
		candidate tmbitmask.BitMask
	}{
		{
			name:      "equal aggregates fallback",
			cpu:       stubNUMAScorer{curScore: 50, candScore: 50},
			mem:       stubNUMAScorer{curScore: 30, candScore: 30},
			wantOK:    false,
			current:   m0,
			candidate: m1,
		},
		{
			name:      "cpu dominated prefers candidate",
			cpu:       stubNUMAScorer{curScore: 20, candScore: 80},
			mem:       stubNUMAScorer{curScore: 0, candScore: 0},
			wantPick:  true,
			wantOK:    true,
			current:   m0,
			candidate: m1,
		},
		{
			name:      "memory dominated prefers current",
			cpu:       stubNUMAScorer{curScore: 0, candScore: 0},
			mem:       stubNUMAScorer{curScore: 90, candScore: 10},
			wantPick:  false,
			wantOK:    true,
			current:   m0,
			candidate: m1,
		},
		{
			name:      "both agree candidate wins",
			cpu:       stubNUMAScorer{curScore: 30, candScore: 70},
			mem:       stubNUMAScorer{curScore: 20, candScore: 60},
			wantPick:  true,
			wantOK:    true,
			current:   m0,
			candidate: m1,
		},
		{
			name:      "mixed scores candidate wins on aggregate",
			cpu:       stubNUMAScorer{curScore: 40, candScore: 80},
			mem:       stubNUMAScorer{curScore: 80, candScore: 79},
			wantPick:  true,
			wantOK:    true,
			current:   m0,
			candidate: m1,
		},
		{
			name:      "both zero fallback",
			cpu:       stubNUMAScorer{curScore: 0, candScore: 0},
			mem:       stubNUMAScorer{curScore: 0, candScore: 0},
			wantOK:    false,
			current:   m0,
			candidate: m1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := &numaScorerAggregator{cpu: tc.cpu, mem: tc.mem}
			result := a.Score(tc.current, tc.candidate)
			if result.Ok != tc.wantOK {
				t.Fatalf("ok: got %v want %v", result.Ok, tc.wantOK)
			}
			if result.Ok && result.PreferCandidate != tc.wantPick {
				t.Fatalf("pick: got %v want %v", result.PreferCandidate, tc.wantPick)
			}
		})
	}
}
