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

	flowcontrol "k8s.io/api/flowcontrol/v1"
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
		if testDebugLogs {
			t.Logf("%s.digests[%v] = %#+v", ftr.fs.Name, expectMatch, digests1)
		}
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
			if testDebugLogs {
				t.Logf("policyRule=%s, mrd=%#+v, mnd=%#+v, srd=%#+v, snd=%#+v", fcfmt.Fmt(policyRule), matchingRDigests, matchingNDigests, skippingRDigests, skippingNDigests)
			}
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
		RequestInfo: &request.RequestInfo{
			IsResourceRequest: true,
			Path:              "/apis/goodapig/v1/namespaces/goodns/goodrscs",
			Verb:              "goodverb",
			APIPrefix:         "apis",
			APIGroup:          "goodapig",
			APIVersion:        "v1",
			Namespace:         "goodns",
			Resource:          "goodrscs",
			Name:              "eman",
			Parts:             []string{"goodrscs", "eman"},
		},
		User: ui,
	}
	reqRU := RequestDigest{
		RequestInfo: &request.RequestInfo{
			IsResourceRequest: true,
			Path:              "/apis/goodapig/v1/goodrscs",
			Verb:              "goodverb",
			APIPrefix:         "apis",
			APIGroup:          "goodapig",
			APIVersion:        "v1",
			Namespace:         "",
			Resource:          "goodrscs",
			Name:              "eman",
			Parts:             []string{"goodrscs", "eman"},
		},
		User: ui,
	}
	reqN := RequestDigest{
		RequestInfo: &request.RequestInfo{
			IsResourceRequest: false,
			Path:              "/openapi/v2",
			Verb:              "goodverb",
		},
		User: ui,
	}
	checkRules(t, true, reqRN, []flowcontrol.PolicyRulesWithSubjects{{
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindGroup,
			Group: &flowcontrol.GroupSubject{Name: "goodg1"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "*"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindGroup,
			Group: &flowcontrol.GroupSubject{Name: "*"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"*"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"*"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"*"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"*"}}}},
	})
	checkRules(t, false, reqRN, []flowcontrol.PolicyRulesWithSubjects{{
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "badu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindGroup,
			Group: &flowcontrol.GroupSubject{Name: "badg"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"badverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"badapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"badrscs"},
			Namespaces: []string{"goodns"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:      []string{"goodverb"},
			APIGroups:  []string{"goodapig"},
			Resources:  []string{"goodrscs"},
			Namespaces: []string{"badns"}}}},
	})
	checkRules(t, true, reqRU, []flowcontrol.PolicyRulesWithSubjects{{
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:        []string{"*"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"*"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"*"},
			ClusterScope: true}}}})
	checkRules(t, false, reqRU, []flowcontrol.PolicyRulesWithSubjects{{
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:        []string{"badverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"badapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: true}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"badrscs"},
			ClusterScope: true}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		ResourceRules: []flowcontrol.ResourcePolicyRule{{
			Verbs:        []string{"goodverb"},
			APIGroups:    []string{"goodapig"},
			Resources:    []string{"goodrscs"},
			ClusterScope: false}}},
	})
	checkRules(t, true, reqN, []flowcontrol.PolicyRulesWithSubjects{{
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		NonResourceRules: []flowcontrol.NonResourcePolicyRule{{
			Verbs:           []string{"goodverb"},
			NonResourceURLs: []string{"/openapi/v2"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		NonResourceRules: []flowcontrol.NonResourcePolicyRule{{
			Verbs:           []string{"*"},
			NonResourceURLs: []string{"/openapi/v2"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		NonResourceRules: []flowcontrol.NonResourcePolicyRule{{
			Verbs:           []string{"goodverb"},
			NonResourceURLs: []string{"*"}}}},
	})
	checkRules(t, false, reqN, []flowcontrol.PolicyRulesWithSubjects{{
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		NonResourceRules: []flowcontrol.NonResourcePolicyRule{{
			Verbs:           []string{"badverb"},
			NonResourceURLs: []string{"/openapi/v2"}}}}, {
		Subjects: []flowcontrol.Subject{{Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{Name: "goodu"}}},
		NonResourceRules: []flowcontrol.NonResourcePolicyRule{{
			Verbs:           []string{"goodverb"},
			NonResourceURLs: []string{"/closedapi/v2"}}}},
	})
}

func checkRules(t *testing.T, expectMatch bool, digest RequestDigest, rules []flowcontrol.PolicyRulesWithSubjects) {
	for idx, rule := range rules {
		fs := &flowcontrol.FlowSchema{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("rule%d", idx)},
			Spec: flowcontrol.FlowSchemaSpec{
				Rules: []flowcontrol.PolicyRulesWithSubjects{rule}}}
		actualMatch := matchesFlowSchema(digest, fs)
		if expectMatch != actualMatch {
			t.Errorf("expectMatch=%v, actualMatch=%v, digest=%#+v, fs=%s", expectMatch, actualMatch, digest, fcfmt.Fmt(fs))
		}
	}
}
