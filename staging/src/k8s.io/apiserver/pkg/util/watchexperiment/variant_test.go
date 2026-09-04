/*
Copyright 2026 The Kubernetes Authors.

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

package watchexperiment

import "testing"

func TestVariantForHostname(t *testing.T) {
	testCases := []struct {
		name     string
		hostname string
		want     Variant
	}{{
		// The three masters of a scale run, named as kops names them.
		name:     "zone b runs the raised grace budget",
		hostname: "control-plane-us-east1-b-crf7",
		want:     VariantGraceBudget,
	}, {
		name:     "zone c runs the raised chunk write size",
		hostname: "control-plane-us-east1-c-20k7",
		want:     VariantChunkWriteSize,
	}, {
		name:     "zone d is the control",
		hostname: "control-plane-us-east1-d-54qp",
		want:     VariantControl,
	}, {
		name:     "an unexpected zone falls back to control",
		hostname: "control-plane-us-east1-f-abcd",
		want:     VariantControl,
	}, {
		name:     "a hostname that is not kops-shaped falls back to control",
		hostname: "localhost",
		want:     VariantControl,
	}, {
		name:     "an empty hostname falls back to control",
		hostname: "",
		want:     VariantControl,
	}}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := variantForHostname(tc.hostname); got != tc.want {
				t.Errorf("variantForHostname(%q) = %q, want %q", tc.hostname, got, tc.want)
			}
		})
	}
}

// The three masters of a run must land in three different arms, otherwise the
// run is not a controlled experiment. This is the property that ruled out
// hashing the hostname.
func TestThreeMastersGetThreeDistinctVariants(t *testing.T) {
	hostnames := []string{
		"control-plane-us-east1-b-crf7",
		"control-plane-us-east1-c-20k7",
		"control-plane-us-east1-d-54qp",
	}
	seen := map[Variant]string{}
	for _, hostname := range hostnames {
		variant := variantForHostname(hostname)
		if other, duplicate := seen[variant]; duplicate {
			t.Fatalf("%q and %q both resolved to %q", other, hostname, variant)
		}
		seen[variant] = hostname
	}
	if len(seen) != 3 {
		t.Errorf("expected 3 distinct variants across 3 masters, got %d", len(seen))
	}
}

func TestMaxDispatchBudgetMatchesVariant(t *testing.T) {
	// Current() memoizes the real hostname, so assert the mapping rather than
	// the resolved value.
	if raisedMaxDispatchBudget <= defaultMaxDispatchBudget {
		t.Errorf("the grace budget arm must raise the ceiling, got %v vs %v", raisedMaxDispatchBudget, defaultMaxDispatchBudget)
	}
	if raisedChunkWriteSize <= defaultChunkWriteSize {
		t.Errorf("the chunk write arm must raise the buffer, got %v vs %v", raisedChunkWriteSize, defaultChunkWriteSize)
	}
}
