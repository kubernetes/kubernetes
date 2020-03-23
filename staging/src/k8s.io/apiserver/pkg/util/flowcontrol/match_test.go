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

	fcv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
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

func TestLiterals(t *testing.T) {
	ui := &user.DefaultInfo{Name: "goodu", UID: "1",
		Groups: []string{"goodg1", "goodg2"}}
	reqRN := RequestDigest{
		&request.RequestInfo{
			IsResourceRequest: true,
			Path:              "/apis/goodapig/v1/namespaces/goodns/goodrscs",
			Verb:              "goodverb",
			APIPrefix:         "apis",
			APIGroup:          "goodapig",
			APIVersion:        "v1",
			Namespace:         "goodns",
			Resource:          "goodrscs",
			Name:              "eman",
			Parts:             []string{"goodrscs", "eman"}},
		ui}
	reqRU := RequestDigest{
		&request.RequestInfo{
			IsResourceRequest: true,
			Path:              "/apis/goodapig/v1/goodrscs",
			Verb:              "goodverb",
			APIPrefix:         "apis",
			APIGroup:          "goodapig",
			APIVersion:        "v1",
			Namespace:         "",
			Resource:          "goodrscs",
			Name:              "eman",
			Parts:             []string{"goodrscs", "eman"}},
		ui}
	reqN := RequestDigest{
		&request.RequestInfo{
			IsResourceRequest: false,
			Path:              "/openapi/v2",
			Verb:              "goodverb"},
		ui}
	checkRules(t, true, reqRN, []fcv1a1.PolicyRulesWithSubjects{{
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindGroup,
			Group: &fcv1a1.GroupSubject{"goodg1"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"*"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindGroup,
			Group: &fcv1a1.GroupSubject{"*"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"*"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"*"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"*"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"*"}}}},
	})
	checkRules(t, false, reqRN, []fcv1a1.PolicyRulesWithSubjects{{
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"badu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindGroup,
			Group: &fcv1a1.GroupSubject{"badg"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"badverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"badapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"badrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"badns"}}}},
	})
	checkRules(t, true, reqRU, []fcv1a1.PolicyRulesWithSubjects{{
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:        []string{"*"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"*"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"*"},
			ClusterScope: true}}}})
	checkRules(t, false, reqRU, []fcv1a1.PolicyRulesWithSubjects{{
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:        []string{"badverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"badapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"badrscs"},
			ClusterScope: true}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		ResourceRules: []fcv1a1.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: false}}},
	})
	checkRules(t, true, reqN, []fcv1a1.PolicyRulesWithSubjects{{
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		NonResourceRules: []fcv1a1.NonResourcePolicyRule{{
			Verbs:           []string{"goodverb"},
			NonResourceURLs: []string{"/openapi/v2"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		NonResourceRules: []fcv1a1.NonResourcePolicyRule{{
			Verbs:           []string{"*"},
			NonResourceURLs: []string{"/openapi/v2"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		NonResourceRules: []fcv1a1.NonResourcePolicyRule{{
			Verbs:           []string{"goodverb"},
			NonResourceURLs: []string{"*"}}}},
	})
	checkRules(t, false, reqN, []fcv1a1.PolicyRulesWithSubjects{{
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		NonResourceRules: []fcv1a1.NonResourcePolicyRule{{
			Verbs:           []string{"badverb"},
			NonResourceURLs: []string{"/openapi/v2"}}}}, {
		Subjects: []fcv1a1.Subject{{Kind: fcv1a1.SubjectKindUser,
			User: &fcv1a1.UserSubject{"goodu"}}},
		NonResourceRules: []fcv1a1.NonResourcePolicyRule{{
			Verbs:           []string{"goodverb"},
			NonResourceURLs: []string{"/closedapi/v2"}}}},
	})
}

func checkRules(t *testing.T, expectMatch bool, digest RequestDigest, rules []fcv1a1.PolicyRulesWithSubjects) {
	for idx, rule := range rules {
		fs := &fcv1a1.FlowSchema{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("rule%d", idx)},
			Spec: fcv1a1.FlowSchemaSpec{
				Rules: []fcv1a1.PolicyRulesWithSubjects{rule}}}
		actualMatch := matchesFlowSchema(digest, fs)
		if expectMatch != actualMatch {
			t.Errorf("expectMatch=%v, actualMatch=%v, digest=%#+v, fs=%s", expectMatch, actualMatch, digest, fcfmt.Fmt(fs))
		}
	}
}
