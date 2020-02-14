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

package flowcontrol

import (
	"fmt"
	"math/rand"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	fcfmt "k8s.io/apiserver/pkg/util/flowcontrol/format"
)

func TestMatching(t *testing.T) {
	checkFTR(t, mandFTRExempt)
	checkFTR(t, mandFTRCatchAll)
	rngOuter := rand.New(rand.NewSource(42))
	goodPLNames := sets.NewString("pl1", "pl2", "pl3", "pl4", "pl5")
	badPLNames := sets.NewString("ql1", "ql2", "ql3", "ql4", "ql5")
	for i := 0; i < 300; i++ {
		rng := rand.New(rand.NewSource(int64(rngOuter.Uint64())))
		t.Run(fmt.Sprintf("trial%d:", i), func(t *testing.T) {
			ftr := genFS(t, rng, fmt.Sprintf("fs%d", i), rng.Float32() < 0.2, goodPLNames, badPLNames)
			checkFTR(t, ftr)
		})
	}
}

func checkFTR(t *testing.T, ftr *fsTestingRecord) {
	for expectMatch, digests1 := range ftr.digests {
		t.Logf("%s.digests[%v] = %#+v", ftr.fs.Name, expectMatch, digests1)
		for _, digests2 := range digests1 {
			for _, digest := range digests2 {
				actualMatch := matchesFlowSchema(digest, ftr.fs)
				if expectMatch != actualMatch {
					t.Errorf("Fail for %s vs %#+v: expectedMatch=%v, actualMatch=%v", fcfmt.Fmt(ftr.fs), digest, expectMatch, actualMatch)
				}
			}
		}
	}
}

func TestPolicyRules(t *testing.T) {
	rngOuter := rand.New(rand.NewSource(42))
	for i := 0; i < 300; i++ {
		rng := rand.New(rand.NewSource(int64(rngOuter.Uint64())))
		t.Run(fmt.Sprintf("trial%d:", i), func(t *testing.T) {
			r := rng.Float32()
			n := rng.Float32()
			policyRule, matchingRDigests, matchingNDigests, skippingRDigests, skippingNDigests := genPolicyRuleWithSubjects(t, rng, fmt.Sprintf("t%d", i), rng.Float32() < 0.2, r < 0.10, n < 0.10, r < 0.05, n < 0.05)
			t.Logf("policyRule=%s, mrd=%#+v, mnd=%#+v, srd=%#+v, snd=%#+v", fcfmt.Fmt(policyRule), matchingRDigests, matchingNDigests, skippingRDigests, skippingNDigests)
			for _, digest := range append(matchingRDigests, matchingNDigests...) {
				if !matchesPolicyRule(digest, &policyRule) {
					t.Errorf("Fail: expected %s to match %#+v but it did not", fcfmt.Fmt(policyRule), digest)
				}
			}
			for _, digest := range append(skippingRDigests, skippingNDigests...) {
				if matchesPolicyRule(digest, &policyRule) {
					t.Errorf("Fail: expected %s to not match %#+v but it did", fcfmt.Fmt(policyRule), digest)
				}
			}
		})
	}
}
