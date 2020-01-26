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
	"time"

	fcv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	fqtesting "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing"
	fcfmt "k8s.io/apiserver/pkg/util/flowcontrol/format"
)

var noRestraintQSF = fqtesting.NewNoRestraintFactory()

// genPL creates a PriorityLevelConfiguration with the given name and
// randomly generated spec.  The spec is relatively likely to be valid
// but might be not be valid.  The returned boolean indicates whether
// the returned object is acceptable to this package; this may be a
// more liberal test than the official validation test.
func genPL(rng *rand.Rand, name string) (*fcv1a1.PriorityLevelConfiguration, bool) {
	plc := &fcv1a1.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: fcv1a1.PriorityLevelConfigurationSpec{
			Type: fcv1a1.PriorityLevelEnablementExempt}}
	if rng.Float32() < 0.98 {
		plc.Spec.Type = fcv1a1.PriorityLevelEnablementLimited
	}
	if (plc.Spec.Type == fcv1a1.PriorityLevelEnablementLimited) == (rng.Float32() < 0.95) {
		plc.Spec.Limited = &fcv1a1.LimitedPriorityLevelConfiguration{
			AssuredConcurrencyShares: rng.Int31n(100) - 5,
			LimitResponse: fcv1a1.LimitResponse{
				Type: fcv1a1.LimitResponseTypeReject}}
		if rng.Float32() < 0.95 {
			plc.Spec.Limited.LimitResponse.Type = fcv1a1.LimitResponseTypeQueue
		}
		if (plc.Spec.Limited.LimitResponse.Type == fcv1a1.LimitResponseTypeQueue) == (rng.Float32() < 0.98) {
			qc := &fcv1a1.QueuingConfiguration{
				Queues:           rng.Int31n(256) - 25,
				HandSize:         rng.Int31n(96) - 16,
				QueueLengthLimit: rng.Int31n(20) - 2}
			plc.Spec.Limited.LimitResponse.Queuing = qc
		}
	}
	_, err := qscOfPL(noRestraintQSF, nil, name, &plc.Spec, time.Minute)
	return plc, err == nil
}

// genFS creates a FlowSchema with the given name and randomly
// generated spec, along with example matching and non-matching
// digests.  The spec is relatively likely to be valid but might be
// not be valid.  goodPLNames may be empty, but badPLNames may not.
// The first returned bool indicates whether the returned object is
// acceptable to this package; this may be a more liberal test than
// official validation.  The second returned bool indicates whether
// the flow schema matches every RequestDigest for a resource request.
// The third returned bool indicates whether the flow schema matches
// every RequestDigest for a non-resource request.
func genFS(t *testing.T, rng *rand.Rand, name string, goodPLNames, badPLNames sets.String) (*fcv1a1.FlowSchema, bool, bool, bool, []RequestDigest, []RequestDigest) {
	fs := &fcv1a1.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       fcv1a1.FlowSchemaSpec{}}
	valid := true
	if rng.Float32() < 0.9 && len(goodPLNames) > 0 {
		fs.Spec.PriorityLevelConfiguration = fcv1a1.PriorityLevelConfigurationReference{pickSetString(rng, goodPLNames)}
	} else {
		fs.Spec.PriorityLevelConfiguration = fcv1a1.PriorityLevelConfigurationReference{pickSetString(rng, badPLNames)}
		valid = false
	}
	fs.Spec.MatchingPrecedence = rng.Int31n(15000) - 2500
	if rng.Float32() < 0.8 {
		fdmt := fcv1a1.FlowDistinguisherMethodType(
			pickString(rng,
				stringp{s: string(fcv1a1.FlowDistinguisherMethodByUserType), p: 1},
				stringp{s: string(fcv1a1.FlowDistinguisherMethodByNamespaceType), p: 1},
				stringp{s: "fubar", p: 0.1}))
		fs.Spec.DistinguisherMethod = &fcv1a1.FlowDistinguisherMethod{fdmt}
	}
	fs.Spec.Rules = []fcv1a1.PolicyRulesWithSubjects{}
	matchingDigests := []RequestDigest{}
	skippingDigests := []RequestDigest{}

	// We are going to independently generate each rule and its
	// associated matching and non-matching ("skipping") digests, and
	// then simply put them all together in the result.  The rule
	// generator takes special care to produce digest lists for which
	// this is correct (so that one rule does not match any of another
	// rule's skipping digests).

	if rng.Float32() < 0.05 {
		// Produce a FlowSchema with zero rules.
		// Gen some digests to test against.
		_, _, matchingDigests, _ := genPolicyRuleWithSubjects(t, rng, false, false, false, false)
		return fs, valid, false, false, nil, matchingDigests
	}
	matchEveryResourceRequest := rng.Float32() < 0.1
	matchEveryNonResourceRequest := rng.Float32() < 0.1
	nRules := (1 + rng.Intn(3)) * (1 + rng.Intn(3))
	everyResourceMatcher := -1
	if matchEveryResourceRequest {
		everyResourceMatcher = rng.Intn(nRules)
	}
	everyNonResourceMatcher := -1
	if matchEveryNonResourceRequest {
		everyNonResourceMatcher = rng.Intn(nRules)
	}
	for i := 0; i < nRules; i++ {
		rule, ruleValid, ruleMatchingDigests, ruleSkippingDigests := genPolicyRuleWithSubjects(t, rng, matchEveryResourceRequest, matchEveryNonResourceRequest, i == everyResourceMatcher, i == everyNonResourceMatcher)
		valid = valid && ruleValid
		fs.Spec.Rules = append(fs.Spec.Rules, rule)
		matchingDigests = append(matchingDigests, ruleMatchingDigests...)
		skippingDigests = append(skippingDigests, ruleSkippingDigests...)
		if rng.Float32() < 0.5 {
			break
		}
	}
	t.Logf("Returning name=%s, plRef=%q, valid=%v, matchEveryResourceRequest=%v, matchEveryNonResourceRequest=%v", fs.Name, fs.Spec.PriorityLevelConfiguration.Name, valid, matchEveryResourceRequest, matchEveryNonResourceRequest)
	return fs, valid, matchEveryResourceRequest, matchEveryNonResourceRequest, matchingDigests, skippingDigests
}

var noextra = make(map[string][]string)

// Generate one PolicyRulesWithSubjects.  Also returns: `valid bool`,
// matchingDigests, skippingDigests.  So that the returned lists can
// simply be concatenated, this generator makes two considerations.
// One is to take parameters indicating (a) whether any rule will
// match all resource requests and whether any rule will match all
// non-resource requests (so it can avoid generating allegedly
// non-matching digests that will actually match another match-all
// rule) and (b) whether this generated rule should match all of
// either style of request.  The other consideration is that for rules
// that are not intended to match all, the generator ensures that each
// generated non-matching user.Info and and each generated
// non-matching `*request.RequestInfo` fails to match not only the
// local rule but all non-match-all rules.
func genPolicyRuleWithSubjects(t *testing.T, rng *rand.Rand, someMatchesAllResourceRequests, someMatchesAllNonResourceRequests, matchAllResourceRequests, matchAllNonResourceRequests bool) (fcv1a1.PolicyRulesWithSubjects, bool, []RequestDigest, []RequestDigest) {
	matchAllUsers := matchAllResourceRequests || matchAllNonResourceRequests
	subjects := []fcv1a1.Subject{}
	matchingUsers := []user.Info{}
	skippingUsers := []user.Info{}
	resourceRules := []fcv1a1.ResourcePolicyRule{}
	nonResourceRules := []fcv1a1.NonResourcePolicyRule{}
	matchingReqs := []*request.RequestInfo{}
	skippingReqs := []*request.RequestInfo{}
	if !someMatchesAllResourceRequests {
		skippingReqs = append(skippingReqs, genSkippingRsc(rng))
	}
	if !someMatchesAllNonResourceRequests {
		skippingReqs = append(skippingReqs, genSkippingNonRsc(rng))
	}
	valid := true
	nSubj := rng.Intn(4)
	for i := 0; i < nSubj; i++ {
		subject, subjectValid, smus, ssus := genSubject(rng)
		subjects = append(subjects, subject)
		valid = valid && subjectValid
		matchingUsers = append(matchingUsers, smus...)
		skippingUsers = append(skippingUsers, ssus...)
	}
	if matchAllUsers {
		subjects = append(subjects, mkGroupSubject("system:authenticated"), mkGroupSubject("system:unauthenticated"))
		matchingUsers = append(matchingUsers, skippingUsers...)
		skippingUsers = nil
	} else {
		skippingUsers = append(skippingUsers, &user.DefaultInfo{"nobody", "no id", []string{"notme", mg(rng)}, noextra})
	}
	var nRR, nNRR int
	for nRR+nNRR == 0 || matchAllResourceRequests && nRR == 0 || matchAllNonResourceRequests && nNRR == 0 {
		nRR = rng.Intn(4)
		nNRR = rng.Intn(4)
	}
	allResourceMatcher := -1
	if matchAllResourceRequests {
		allResourceMatcher = rng.Intn(nRR)
	}
	for i := 0; i < nRR; i++ {
		rr, rrValid, rmrs, rsrs := genResourceRule(rng, i == allResourceMatcher, someMatchesAllResourceRequests, someMatchesAllNonResourceRequests)
		resourceRules = append(resourceRules, rr)
		valid = valid && rrValid
		matchingReqs = append(matchingReqs, rmrs...)
		skippingReqs = append(skippingReqs, rsrs...)
	}
	allNonResourceMatcher := -1
	if matchAllNonResourceRequests {
		allNonResourceMatcher = rng.Intn(nNRR)
	}
	for i := 0; i < nNRR; i++ {
		nrr, nrrValid, nmrs, nsrs := genNonResourceRule(rng, i == allNonResourceMatcher, someMatchesAllResourceRequests, someMatchesAllNonResourceRequests)
		nonResourceRules = append(nonResourceRules, nrr)
		valid = valid && nrrValid
		matchingReqs = append(matchingReqs, nmrs...)
		skippingReqs = append(skippingReqs, nsrs...)
	}
	rule := fcv1a1.PolicyRulesWithSubjects{subjects, resourceRules, nonResourceRules}
	t.Logf("For someMatchesAllResourceRequests=%v, someMatchesAllNonResourceRequests=%v: generated prws=%#+v, valid=%v, marr=%v, manrr=%v, mu=%#+v, su=%#+v, mr=%#+v, sr=%#+v", someMatchesAllResourceRequests, someMatchesAllNonResourceRequests, fcfmt.Fmt(rule), valid, matchAllResourceRequests, matchAllNonResourceRequests, fcfmt.Fmt(matchingUsers), fcfmt.Fmt(skippingUsers), fcfmt.Fmt(matchingReqs), fcfmt.Fmt(skippingReqs))
	var matchingDigests, skippingDigests []RequestDigest
	sgnMatchingUsers := sgn(len(matchingUsers))
	sgnMatchingReqs := sgn(len(matchingReqs))
	nMatch := (1 + rng.Intn(3)) * (1 + rng.Intn(3)) * sgnMatchingUsers * sgnMatchingReqs
	for i := 0; i < nMatch; i++ {
		rd := RequestDigest{
			matchingReqs[rng.Intn(len(matchingReqs))],
			matchingUsers[rng.Intn(len(matchingUsers))]}
		matchingDigests = append(matchingDigests, rd)
		t.Logf("Added matching digest %#+v", rd)
		if valid && !matchesPolicyRule(rd, &rule) {
			t.Logf("Check failure: rule %#+v does not match digest %#+v", fcfmt.Fmt(rule), rd)
		}
	}
	sgnSkippingUsers := sgn(len(skippingUsers))
	sgnSkippingReqs := sgn(len(skippingReqs))
	nSkip := (1 + rng.Intn(3)) * (1 + rng.Intn(3)) * sgnSkippingUsers * sgnSkippingReqs
	for i := 0; i < nSkip; i++ {
		rd := RequestDigest{
			skippingReqs[rng.Intn(len(skippingReqs))],
			skippingUsers[rng.Intn(len(skippingUsers))],
		}
		skippingDigests = append(skippingDigests, rd)
		t.Logf("Added skipping digest %#+v", rd)
		if valid && matchesPolicyRule(rd, &rule) {
			t.Logf("Check failure: rule %#+v matches digest %#+v", fcfmt.Fmt(rule), rd)
		}
	}
	return rule, valid, matchingDigests, skippingDigests
}

// genSubject returns a randomly generated Subject, a bool indicating
// whether that Subject is valid for this package, a list of matching
// user.Info, and a list of non-matching user.Info.
func genSubject(rng *rand.Rand) (fcv1a1.Subject, bool, []user.Info, []user.Info) {
	subject := fcv1a1.Subject{}
	valid := true
	var matchingUsers, skippingUsers []user.Info
	x := rng.Float32()
	switch {
	case x < 0.32:
		subject.Kind = fcv1a1.SubjectKindUser
		if rng.Float32() < 0.98 {
			subject.User, matchingUsers, skippingUsers = genUser(rng)
		}
		if rng.Float32() < 0.02 {
			subject.Group, _, _ = genGroup(rng)
		}
		if rng.Float32() < 0.02 {
			subject.ServiceAccount, _, _ = genServiceAccount(rng)
		}
	case x < 0.64:
		subject.Kind = fcv1a1.SubjectKindGroup
		if rng.Float32() < 0.98 {
			subject.Group, matchingUsers, skippingUsers = genGroup(rng)
		}
		if rng.Float32() < 0.02 {
			subject.User, _, _ = genUser(rng)
		}
		if rng.Float32() < 0.02 {
			subject.ServiceAccount, _, _ = genServiceAccount(rng)
		}
	case x < 0.96:
		subject.Kind = fcv1a1.SubjectKindServiceAccount
		if rng.Float32() < 0.98 {
			subject.ServiceAccount, matchingUsers, skippingUsers = genServiceAccount(rng)
		}
		if rng.Float32() < 0.02 {
			subject.User, _, _ = genUser(rng)
		}
		if rng.Float32() < 0.02 {
			subject.Group, _, _ = genGroup(rng)
		}
	default:
		if rng.Float32() < 0.5 {
			subject.Kind = fcv1a1.SubjectKind(fmt.Sprintf("%d", rng.Intn(100)))
		}
		_, _, skippingUsers = genUser(rng)
	}
	return subject, valid, matchingUsers, skippingUsers
}

func genUser(rng *rand.Rand) (*fcv1a1.UserSubject, []user.Info, []user.Info) {
	name := fmt.Sprintf("u%d", rng.Uint64())
	matches := []user.Info{&user.DefaultInfo{
		Name:   name,
		UID:    "good-id",
		Groups: []string{"b4", mg(rng), "after"},
		Extra:  noextra}}
	skips := []user.Info{}
	nSkips := 1 + rng.Intn(3)
	for i := 0; i < nSkips; i++ {
		skips = append(skips, &user.DefaultInfo{
			Name:   fmt.Sprintf("v%d", rng.Uint64()),
			UID:    "some-id",
			Groups: []string{"b4", mg(rng), "after"},
			Extra:  noextra})
	}
	return &fcv1a1.UserSubject{name}, matches, skips
}

func mg(rng *rand.Rand) string {
	return ([]string{"system:authenticated", "system:unauthenticated"})[rng.Intn(2)]
}

func mkGroupSubject(group string) fcv1a1.Subject {
	return fcv1a1.Subject{
		Kind:  fcv1a1.SubjectKindGroup,
		Group: &fcv1a1.GroupSubject{group},
	}
}

func genGroup(rng *rand.Rand) (*fcv1a1.GroupSubject, []user.Info, []user.Info) {
	name := fmt.Sprintf("g%d", rng.Uint64())
	nMatches := 1 + rng.Intn(3)
	matches := []user.Info{}
	for i := 0; i < nMatches; i++ {
		matches = append(matches, &user.DefaultInfo{
			Name:   fmt.Sprintf("v%d", rng.Uint64()),
			UID:    "good-id",
			Groups: []string{"b4", name, mg(rng)},
			Extra:  noextra})
	}
	skips := []user.Info{}
	nSkips := 1 + rng.Intn(3)
	for i := 0; i < nSkips; i++ {
		skips = append(skips, &user.DefaultInfo{
			Name:   fmt.Sprintf("v%d", rng.Uint64()),
			UID:    "some-id",
			Groups: []string{"b4", fmt.Sprintf("h%d", rng.Uint64()), mg(rng)},
			Extra:  noextra})
	}
	return &fcv1a1.GroupSubject{name}, matches, skips
}

func genServiceAccount(rng *rand.Rand) (*fcv1a1.ServiceAccountSubject, []user.Info, []user.Info) {
	ns := fmt.Sprintf("ns%d", rng.Uint64())
	name := fmt.Sprintf("n%d", rng.Uint64())
	matches := []user.Info{&user.DefaultInfo{
		Name:   fmt.Sprintf("system:serviceaccount:%s:%s", ns, name),
		UID:    "good-id",
		Groups: []string{"b4", mg(rng), "after"},
		Extra:  noextra}}
	skips := []user.Info{}
	nSkips := 1 + rng.Intn(3)
	for i := 0; i < nSkips; i++ {
		skips = append(skips, &user.DefaultInfo{
			Name:   fmt.Sprintf("v%d", rng.Uint64()),
			UID:    "some-id",
			Groups: []string{"b4", fmt.Sprintf("h%d", rng.Uint64()), mg(rng)},
			Extra:  noextra})
	}
	return &fcv1a1.ServiceAccountSubject{Namespace: ns, Name: name}, matches, skips
}

// genResourceRule randomly generates a ResourcePolicyRule, a bool
// indicating whether than rule is valid for this package, and lists
// of matching and non-matching `*request.RequestInfo`.
func genResourceRule(rng *rand.Rand, matchAllResources, someMatchesAllResources, otherMatchesAllNonResources bool) (fcv1a1.ResourcePolicyRule, bool, []*request.RequestInfo, []*request.RequestInfo) {
	nns := rng.Intn(4)
	if matchAllResources {
		nns = 1 + rng.Intn(3)
	}
	rr := fcv1a1.ResourcePolicyRule{
		Verbs:        genWords(rng, true, "verb", 1+rng.Intn(3)),
		APIGroups:    genWords(rng, true, "apigrp", 1+rng.Intn(3)),
		Resources:    genWords(rng, true, "plural", 1+rng.Intn(3)),
		ClusterScope: matchAllResources || rng.Intn(2) == 0,
		Namespaces:   genNums(rng, "ns", 4)[:nns],
	}
	// choose a proper subset of fields to wildcard; only matters if not matching all
	starMask := rng.Intn(15)
	if matchAllResources || starMask&1 == 1 && rng.Float32() < 0.1 {
		rr.Verbs = []string{fcv1a1.VerbAll}
	}
	if matchAllResources || starMask&2 == 2 && rng.Float32() < 0.1 {
		rr.APIGroups = []string{fcv1a1.APIGroupAll}
	}
	if matchAllResources || starMask&4 == 4 && rng.Float32() < 0.1 {
		rr.Resources = []string{fcv1a1.ResourceAll}
	}
	if matchAllResources || starMask&8 == 8 && rng.Float32() < 0.1 {
		rr.Namespaces = []string{fcv1a1.NamespaceEvery}
	}
	valid := true
	matchingReqs := []*request.RequestInfo{}
	skippingReqs := []*request.RequestInfo{}
	nSkip := 1 + rng.Intn(3)
	for i := 0; i < nSkip; i++ {
		if rng.Float32() < 0.67 && !someMatchesAllResources {
			// Add a resource request that mismatches every generated resource
			// rule on every field, so that wildcards on a subset of
			// fields will not defeat the mismatch
			ri := genSkippingRsc(rng)
			skippingReqs = append(skippingReqs, ri)
		} else if !otherMatchesAllNonResources {
			// Add a non-resource request that mismatches every
			// generated non-resource rule on every field, so that
			// wildcards on a subset of fields will not defeat the
			// mismatch
			skippingReqs = append(skippingReqs, genSkippingNonRsc(rng))
		}
	}
	if len(rr.Namespaces) > 0 || rr.ClusterScope {
		nMatch := 1 + rng.Intn(3)
		for i := 0; i < nMatch; i++ {
			apig := rr.APIGroups[rng.Intn(len(rr.APIGroups))]
			apiv := fmt.Sprintf("v%d", 1+rng.Intn(9))
			rsc := rr.Resources[rng.Intn(len(rr.Resources))]
			ri := &request.RequestInfo{
				IsResourceRequest: true,
				Path:              fmt.Sprintf("/apis/%s/%s/%s", apig, apiv, rsc),
				Verb:              rr.Verbs[rng.Intn(len(rr.Verbs))],
				APIPrefix:         "apis",
				APIGroup:          apig,
				APIVersion:        apiv,
				Resource:          rsc,
				Parts:             []string{rsc}}
			if rr.Verbs[0] == fcv1a1.VerbAll {
				ri.Verb = genWord(rng, false, "verb")
			}
			if rr.APIGroups[0] == fcv1a1.APIGroupAll {
				ri.APIGroup = genWord(rng, false, "apigrp")
			}
			if rr.Resources[0] == fcv1a1.ResourceAll {
				ri.Resource = genWord(rng, false, "plural")
			}
			if len(rr.Namespaces) == 0 {
				// can only match non-namespaced requests
			} else {
				if rr.Namespaces[0] == fcv1a1.NamespaceEvery {
					ri.Namespace = fmt.Sprintf("ns%d", rng.Uint64())
				} else {
					ri.Namespace = rr.Namespaces[rng.Intn(len(rr.Namespaces))]
				}
				ri.Path = ri.Path + "/" + ri.Namespace
			}
			matchingReqs = append(matchingReqs, ri)
		}
	}
	return rr, valid, matchingReqs, skippingReqs
}

func genSkippingRsc(rng *rand.Rand) *request.RequestInfo {
	rsc := genWord(rng, false, "plural")
	apig := genWord(rng, false, "apigrp")
	apiv := fmt.Sprintf("v%d", 1+rng.Intn(9))
	ns := fmt.Sprintf("ms%d", rng.Int())
	ri := &request.RequestInfo{
		IsResourceRequest: true,
		Path:              fmt.Sprintf("/apis/%s/%s/%s/%s", apig, apiv, rsc, ns),
		Verb:              genWord(rng, false, "verb"),
		APIPrefix:         "apis",
		APIGroup:          apig,
		APIVersion:        apiv,
		Namespace:         ns,
		Resource:          rsc,
		Parts:             []string{rsc}}
	if rng.Float32() < 0.3 {
		name := genWord(rng, false, "") + genWord(rng, true, "")
		ri.Subresource = "status"
		ri.Parts = append(ri.Parts, name, "status")
		ri.Path = fmt.Sprintf("%s/%s/%s", ri.Path, name, ri.Subresource)
	}
	return ri
}

func genSkippingNonRsc(rng *rand.Rand) *request.RequestInfo {
	return &request.RequestInfo{
		IsResourceRequest: false,
		Path:              genWord(rng, false, "path"),
		Verb:              genWord(rng, false, "verb"),
	}
}

// genNonResourceRule returns a randomly generated
// NonResourcePolicyRule, a bool indicating whether that rule is valid
// for this package, and lists of matching and non-matching
// `*request.RequestInfo`.
func genNonResourceRule(rng *rand.Rand, matchAllNonResources, otherMatchesAllResources, someMatchesAllNonResources bool) (fcv1a1.NonResourcePolicyRule, bool, []*request.RequestInfo, []*request.RequestInfo) {
	nrr := fcv1a1.NonResourcePolicyRule{
		Verbs:           genWords(rng, true, "verb", 1+rng.Intn(4)),
		NonResourceURLs: genWords(rng, true, "path", 1+rng.Intn(4)),
	}
	// choose a proper subset of fields to consider wildcarding; only matters if not matching all
	starMask := rng.Intn(3)
	if matchAllNonResources || starMask&1 == 1 && rng.Float32() < 0.1 {
		nrr.Verbs = []string{fcv1a1.VerbAll}
	}
	if matchAllNonResources || starMask&2 == 2 && rng.Float32() < 0.1 {
		nrr.NonResourceURLs = []string{"*"}
	}
	valid := true
	matchingReqs := []*request.RequestInfo{}
	skippingReqs := []*request.RequestInfo{}
	nSkip := 1 + rng.Intn(3)
	for i := 0; i < nSkip; i++ {
		if rng.Float32() < 0.67 && !someMatchesAllNonResources {
			// Add a non-resource request that mismatches every
			// generated non-resource rule on every field, so that
			// wildcards on a subset of fields will not defeat the
			// mismatch
			skippingReqs = append(skippingReqs, genSkippingNonRsc(rng))
		} else if !otherMatchesAllResources {
			// Add a resource request that mismatches every generated resource
			// rule on every field, so that wildcards on a subset of
			// fields will not defeat the mismatch
			skippingReqs = append(skippingReqs, genSkippingRsc(rng))
		}
	}
	nMatch := 1 + rng.Intn(3)
	for i := 0; i < nMatch; i++ {
		ri := &request.RequestInfo{
			IsResourceRequest: false,
			Path:              nrr.NonResourceURLs[rng.Intn(len(nrr.NonResourceURLs))],
			Verb:              nrr.Verbs[rng.Intn(len(nrr.Verbs))],
		}
		if nrr.Verbs[0] == fcv1a1.VerbAll {
			ri.Verb = genWord(rng, false, "verb")
		}
		if nrr.NonResourceURLs[0] == "*" {
			ri.Path = genWord(rng, false, "path")
		}
		matchingReqs = append(matchingReqs, ri)
	}
	return nrr, valid, matchingReqs, skippingReqs
}

func sgn(x int) int {
	if x == 0 {
		return 0
	} else if x < 0 {
		return -1
	}
	return 1
}

type stringp struct {
	s string
	p float32
}

func pickString(rng *rand.Rand, choices ...stringp) string {
	var sum float32
	for _, sp := range choices {
		sum += sp.p
	}
	x := rng.Float32() * sum
	for _, sp := range choices {
		x -= sp.p
		if x <= 0 {
			return sp.s
		}
	}
	return choices[len(choices)-1].s
}

func pickSetString(rng *rand.Rand, set sets.String) string {
	i, n := 0, rng.Intn(len(set))
	for s := range set {
		if i == n {
			return s
		}
		i++
	}
	panic("empty set")
}

const (
	loInitials = "bcdfghjklm"
	hiInitials = "npqrstvwyz"
	finals     = "bcdfghjklmnpqrtvwy"
	vowels     = "aeiou"
)

var verbEndings = []string{"o", "u", "en", "er"}

// Returns a random string that looks like it could be a word root,
// and is suitable for appending "s" or a vowel.  The result is drawn
// from one of two populations as chosen by the given bool.
func genRoot(rng *rand.Rand, sel bool) string {
	letters1 := loInitials
	if sel {
		letters1 = hiInitials
	}
	return string([]byte{
		letters1[rng.Intn(len(letters1))],
		vowels[rng.Intn(len(vowels))],
		finals[rng.Intn(len(finals))],
	})
}

// Returns a random word-like string of the given sort.
func genWord(rng *rand.Rand, sel bool, sort string) string {
	switch sort {
	case "plural":
		return genRoot(rng, sel) + "s"
	case "verb":
		return genRoot(rng, sel) + verbEndings[rng.Intn(len(verbEndings))]
	case "path":
		return "/" + genRoot(rng, sel) + "/" + genRoot(rng, sel)
	case "apigrp":
		return genRoot(rng, sel) + "." + genRoot(rng, sel)
	default:
		return genRoot(rng, sel)
	}
}

func genWords(rng *rand.Rand, sel bool, sort string, n int) []string {
	ans := make([]string, 0, n)
	for i := 0; i < n; i++ {
		ans = append(ans, genWord(rng, sel, sort))
	}
	return ans
}

func genNums(rng *rand.Rand, prefix string, n int) []string {
	ans := make([]string, 0, n)
	for i := 0; i < n; i++ {
		word := fmt.Sprintf("%s%d", prefix, rng.Int())
		ans = append(ans, word)
	}
	return ans
}
