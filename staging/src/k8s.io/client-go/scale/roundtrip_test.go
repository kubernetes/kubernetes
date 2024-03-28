/*
Copyright 2017 The Kubernetes Authors.

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

package scale

import (
	"sort"
	"testing"

	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"

	fuzz "github.com/google/gofuzz"
)

// NB: this can't be in the scheme package, because importing'
// scheme/autoscalingv1 from scheme causes a dependency loop from
// conversions

func TestRoundTrip(t *testing.T) {
	scheme := NewScaleConverter().Scheme()
	roundtrip.RoundTripTestForScheme(t, scheme, func(codecs serializer.CodecFactory) []interface{} {
		// Find the existing metav1.LabelSelector fuzz func, to be applied first, before
		// applying the invariants needed to roundtrip through the textual label selector
		// format.
		lsfuzz := func(j *metav1.LabelSelector, c fuzz.Continue) {}
		for _, i := range metafuzzer.Funcs(codecs) {
			if fn, ok := i.(func(*metav1.LabelSelector, fuzz.Continue)); ok {
				lsfuzz = fn
			}
		}

		return []interface{}{
			// NB: the label selector parser code sorts match expressions by key, and
			// sorts and deduplicates the values, so we need to make sure ours are
			// sorted as well here to preserve round-trip comparison.  In practice, not
			// sorting doesn't hurt anything...
			func(j *metav1.LabelSelector, c fuzz.Continue) {
				lsfuzz(j, c)

				for i, req := range j.MatchExpressions {
					if req.Operator == metav1.LabelSelectorOpIn || req.Operator == metav1.LabelSelectorOpNotIn {
						req.Values = sets.List(sets.New(req.Values...)) // sort and deduplicate
					}
					j.MatchExpressions[i] = req
				}
				sort.Slice(j.MatchExpressions, func(a, b int) bool { return j.MatchExpressions[a].Key < j.MatchExpressions[b].Key })
			},
		}
	})
}
