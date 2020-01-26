/*
Copyright 2019 The Kubernetes Authors.

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

package filterconfig

import (
	"fmt"
	"math/rand"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	fcfmt "k8s.io/apiserver/pkg/util/flowcontrol/format"
)

func TestMatching(t *testing.T) {
	rngOuter := rand.New(rand.NewSource(42))
	goodPLNames := sets.NewString(genWords(rngOuter, true, "", 5)...)
	badPLNames := sets.NewString(genWords(rngOuter, false, "", 5)...)
	for i := 0; i < 1000; i++ {
		rng := rand.New(rand.NewSource(int64(rngOuter.Uint64())))
		t.Run(fmt.Sprintf("trial%d:", i), func(t *testing.T) {
			fs, valid, _, _, matchingDigests, skippingDigests := genFS(t, rng, fmt.Sprintf("fs%d", i), goodPLNames, badPLNames)
			if !valid {
				t.Logf("Not testing invalid %#+v", fcfmt.Fmt(fs))
				return
			}
			for _, digest := range matchingDigests {
				a := matchesFlowSchema(digest, fs)
				if !a {
					t.Errorf("Fail: Expected %#+v to match %#+v but it did not", fcfmt.Fmt(fs), digest)
				}
			}
			for _, digest := range skippingDigests {
				a := matchesFlowSchema(digest, fs)
				if a {
					t.Errorf("Fail: Expected %#+v to not match %#+v but it did", fcfmt.Fmt(fs), digest)
				}
			}
		})
	}
}

func TestPolicyRules(t *testing.T) {
	rngOuter := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		rng := rand.New(rand.NewSource(int64(rngOuter.Uint64())))
		t.Run(fmt.Sprintf("trial%d:", i), func(t *testing.T) {
			r := rng.Float32()
			n := rng.Float32()
			policyRule, valid, matchingDigests, skippingDigests := genPolicyRuleWithSubjects(t, rng, r < 0.15, n < 0.15, r < 0.05, n < 0.05)
			t.Logf("policyRule=%#+v, valid=%v, md=%#+v, sd=%#+v", fcfmt.Fmt(policyRule), valid, matchingDigests, skippingDigests)
			if !valid {
				return
			}
			for _, digest := range matchingDigests {
				if !matchesPolicyRule(digest, &policyRule) {
					t.Logf("Expected %#+v to match %#+v but it did not", fcfmt.Fmt(policyRule), digest)
				}
			}
			for _, digest := range skippingDigests {
				if matchesPolicyRule(digest, &policyRule) {
					t.Logf("Expected %#+v to not match %#+v but it did", fcfmt.Fmt(policyRule), digest)
				}
			}
		})
	}
}
