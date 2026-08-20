/*
Copyright 2024 The Kubernetes Authors.

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

package limitranger

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// A quantity past the int64 range wraps through Value()/MilliValue(): 2^64
// projects to 0 and 2^63 to a negative number. The constraints used to compare
// those projections, so an oversized limit or request slipped past the range
// and the LimitRange stopped being enforced. The exact comparison holds.
func TestConstraintsAreOverflowSafe(t *testing.T) {
	huge := resource.MustParse("18446744073709551616") // 2^64
	normal := resource.MustParse("1000")
	small := resource.MustParse("1")

	if err := maxConstraint("Container", "memory", normal, api.ResourceList{}, api.ResourceList{api.ResourceMemory: huge}); err == nil {
		t.Error("maxConstraint admitted a limit above the max")
	}
	if err := maxRequestConstraint("Container", "memory", normal, api.ResourceList{api.ResourceMemory: huge}); err == nil {
		t.Error("maxRequestConstraint admitted a request above the max")
	}
	if err := minConstraint("Container", "memory", huge, api.ResourceList{api.ResourceMemory: small}, api.ResourceList{}); err == nil {
		t.Error("minConstraint admitted a request below the min")
	}
	if err := limitRequestRatioConstraint("Container", "memory", normal, api.ResourceList{api.ResourceMemory: small}, api.ResourceList{api.ResourceMemory: huge}); err == nil {
		t.Error("limitRequestRatioConstraint admitted a limit-to-request ratio above the max")
	}

	// A within-limit value must still pass, so the exact comparison did not
	// start rejecting ordinary requests.
	if err := maxConstraint("Container", "memory", normal, api.ResourceList{}, api.ResourceList{api.ResourceMemory: small}); err != nil {
		t.Errorf("maxConstraint rejected a within-limit value: %v", err)
	}
}

func TestLimitRangerRatioIsExact(t *testing.T) {
	// The ratio limit/request is compared exactly, so a limit past
	// request*maxRatio is rejected even when both are large enough that an int64
	// or float projection would blur the boundary.
	ratioOf := func(reqStr, limStr, maxRatioStr string) error {
		enforced := resource.MustParse(maxRatioStr)
		req := api.ResourceList{api.ResourceMemory: resource.MustParse(reqStr)}
		lim := api.ResourceList{api.ResourceMemory: resource.MustParse(limStr)}
		return limitRequestRatioConstraint("Container", "memory", enforced, req, lim)
	}
	if err := ratioOf("100", "200", "2"); err != nil {
		t.Errorf("ratio exactly 2 with max 2 was rejected: %v", err)
	}
	if err := ratioOf("100", "300", "2"); err == nil {
		t.Error("ratio 3 admitted against max 2")
	}
	// The review's counterexample: under an int64 projection both collapse to
	// MaxInt64 and the ratio looks like 1.
	if err := ratioOf("18446744073709551616", "55340232221128654848", "2"); err == nil { // 2^64, 3*2^64
		t.Error("ratio 3 at 2^64 admitted against max 2")
	}
	// One unit over request*2. A float ratio rounds this to exactly 2 and would
	// admit it; the exact decimal comparison must reject.
	if err := ratioOf("18446744073709551616", "36893488147419103233", "2"); err == nil { // 2^64, 2^65+1
		t.Error("ratio just above 2 at 2^64 admitted against max 2 (float boundary)")
	}
}
