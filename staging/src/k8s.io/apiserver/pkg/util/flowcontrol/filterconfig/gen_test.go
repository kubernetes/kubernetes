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
	if (plc.Spec.Type == fcv1a1.PriorityLevelEnablementLimited) == (rng.Float32() < 0.98) {
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

// A FlowSchema together with characteristics relevant to testing
type fsTestingRecord struct {
	fs                            *fcv1a1.FlowSchema
	matchesAllResourceRequests    bool
	matchesAllNonResourceRequests bool
	matchingRDigests              []RequestDigest
	matchingNDigests              []RequestDigest
	skippingRDigests              []RequestDigest
	skippingNDigests              []RequestDigest
}

func (ftr *fsTestingRecord) addDigest(digest RequestDigest, matches bool) {
	if matches {
		if digest.RequestInfo.IsResourceRequest {
			ftr.matchingRDigests = append(ftr.matchingRDigests, digest)
		} else {
			ftr.matchingNDigests = append(ftr.matchingNDigests, digest)
		}
	} else {
		if digest.RequestInfo.IsResourceRequest {
			ftr.skippingRDigests = append(ftr.skippingRDigests, digest)
		} else {
			ftr.skippingNDigests = append(ftr.skippingNDigests, digest)
		}
	}
}

func (ftr *fsTestingRecord) addDigests(digests []RequestDigest, matches bool) {
	for _, digest := range digests {
		ftr.addDigest(digest, matches)
	}
}

// genFS creates a FlowSchema with the given name and randomly
// generated spec, along with four lists of digests: matching
// resource-style, matching non-resource-style, non-matching
// resource-style, and non-matching non-resource-style.  When all the
// FlowSchemas in a collection are generated with different names, the
// non-matching digests do not match any schema in the collection.
// The generated spec is relatively likely to be valid but might be
// not be valid.  goodPLNames may be empty, but badPLNames may not.
// The first returned bool indicates whether the returned object is
// acceptable to this package; this may be a more liberal test than
// official validation.  The second returned bool indicates whether
// the flow schema matches every RequestDigest for a resource request.
// The third returned bool indicates whether the flow schema matches
// every RequestDigest for a non-resource request.
func genFS(t *testing.T, rng *rand.Rand, name string, mayMatchClusterScope bool, goodPLNames, badPLNames sets.String) (*fsTestingRecord, bool) {
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
	// 5% chance of zero rules, otherwise draw from 1--6 biased low
	nRules := (1 + rng.Intn(3)) * (1 + rng.Intn(2)) * ((19 + rng.Intn(20)) / 20)
	ftr := &fsTestingRecord{fs: fs,
		matchesAllResourceRequests:    nRules > 0 && rng.Float32() < 0.1,
		matchesAllNonResourceRequests: nRules > 0 && rng.Float32() < 0.1}
	fs.Spec.Rules = []fcv1a1.PolicyRulesWithSubjects{}
	everyResourceMatcher := -1
	if ftr.matchesAllResourceRequests {
		if mayMatchClusterScope {
			everyResourceMatcher = 0
		} else {
			everyResourceMatcher = rng.Intn(nRules)
		}
	}
	everyNonResourceMatcher := -1
	if ftr.matchesAllNonResourceRequests {
		if mayMatchClusterScope {
			everyNonResourceMatcher = 0
		} else {
			everyNonResourceMatcher = rng.Intn(nRules)
		}
	}
	// Allow only one rule if mayMatchClusterScope because that breaks
	// cross-rule exclusion.
	for i := 0; i < nRules && (i == 0 || !mayMatchClusterScope); i++ {
		rule, ruleValid, ruleMatchingRDigests, ruleMatchingNDigests, ruleSkippingRDigests, ruleSkippingNDigests := genPolicyRuleWithSubjects(t, rng, fmt.Sprintf("%s-%d", name, i+1), mayMatchClusterScope, ftr.matchesAllResourceRequests, ftr.matchesAllNonResourceRequests, i == everyResourceMatcher, i == everyNonResourceMatcher)
		valid = valid && ruleValid
		fs.Spec.Rules = append(fs.Spec.Rules, rule)
		ftr.addDigests(ruleMatchingRDigests, true)
		ftr.addDigests(ruleMatchingNDigests, true)
		ftr.addDigests(ruleSkippingRDigests, false)
		ftr.addDigests(ruleSkippingNDigests, false)
	}
	if nRules == 0 {
		_, _, _, _, ftr.skippingRDigests, ftr.skippingNDigests = genPolicyRuleWithSubjects(t, rng, name+"-1", false, false, false, false, false)
	}
	t.Logf("Returning name=%s, plRef=%q, valid=%v, matchesAllResourceRequests=%v, matchesAllNonResourceRequests=%v for mayMatchClusterScope=%v", fs.Name, fs.Spec.PriorityLevelConfiguration.Name, valid, ftr.matchesAllResourceRequests, ftr.matchesAllNonResourceRequests, mayMatchClusterScope)
	return ftr, valid
}

var noextra = make(map[string][]string)

// Generate one PolicyRulesWithSubjects.  Also returns: `valid bool`,
// matching resource-style digests, matching non-resource-style
// digests, skipping (i.e., not matching the generated rule)
// resource-style digests, skipping non-resource-style digests.  When
// a collection of rules is generated with unique prefixes, the
// skipping digests for each rule match no rules in the collection.
// The someMatchesAllResourceRequests and
// someMatchesAllNonResourceRequests parameters indicate whether any
// rule in the collection matches all of the relevant sort of request;
// these imply the respective returned slice of counterexamples will
// be empty.  The matchAllResourceRequests and
// matchAllNonResourceRequests parameters indicate whether the
// generated rule should match all of the relevant sort.  The
// cross-rule exclusion is based on using names that start with the
// given prefix --- which can not be done for the namespace of a
// cluster-scoped request.  Thus, these are normally excluded.  When
// mayMatchClusterScope==true the generated rule may be cluster-scoped
// and there is no promise of cross-rule exclusion.
func genPolicyRuleWithSubjects(t *testing.T, rng *rand.Rand, pfx string, mayMatchClusterScope, someMatchesAllResourceRequests, someMatchesAllNonResourceRequests, matchAllResourceRequests, matchAllNonResourceRequests bool) (fcv1a1.PolicyRulesWithSubjects, bool, []RequestDigest, []RequestDigest, []RequestDigest, []RequestDigest) {
	subjects := []fcv1a1.Subject{}
	matchingUIs := []user.Info{}
	skippingUIs := []user.Info{}
	resourceRules := []fcv1a1.ResourcePolicyRule{}
	nonResourceRules := []fcv1a1.NonResourcePolicyRule{}
	matchingRRIs := []*request.RequestInfo{}
	skippingRRIs := []*request.RequestInfo{}
	matchingNRIs := []*request.RequestInfo{}
	skippingNRIs := []*request.RequestInfo{}
	valid := true
	nSubj := rng.Intn(4)
	for i := 0; i < nSubj; i++ {
		subject, subjectValid, smus, ssus := genSubject(rng, fmt.Sprintf("%s-%d", pfx, i+1))
		subjects = append(subjects, subject)
		valid = valid && subjectValid
		matchingUIs = append(matchingUIs, smus...)
		skippingUIs = append(skippingUIs, ssus...)
	}
	if matchAllResourceRequests || matchAllNonResourceRequests {
		subjects = append(subjects, mkGroupSubject("system:authenticated"), mkGroupSubject("system:unauthenticated"))
		matchingUIs = append(matchingUIs, skippingUIs...)
	}
	if someMatchesAllResourceRequests || someMatchesAllNonResourceRequests {
		skippingUIs = []user.Info{}
	} else if nSubj == 0 {
		_, _, _, skippingUIs = genSubject(rng, pfx+"-o")
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
	// Allow only one resource rule if mayMatchClusterScope because
	// that breaks cross-rule exclusion.
	for i := 0; i < nRR && (i == 0 || !mayMatchClusterScope); i++ {
		rr, rrValid, rmrs, rsrs := genResourceRule(rng, fmt.Sprintf("%s-%d", pfx, i+1), mayMatchClusterScope, i == allResourceMatcher, someMatchesAllResourceRequests)
		resourceRules = append(resourceRules, rr)
		valid = valid && rrValid
		matchingRRIs = append(matchingRRIs, rmrs...)
		skippingRRIs = append(skippingRRIs, rsrs...)
	}
	if nRR == 0 {
		_, _, _, skippingRRIs = genResourceRule(rng, pfx+"-o", mayMatchClusterScope, false, someMatchesAllResourceRequests)
	}
	allNonResourceMatcher := -1
	if matchAllNonResourceRequests {
		allNonResourceMatcher = rng.Intn(nNRR)
	}
	for i := 0; i < nNRR; i++ {
		nrr, nrrValid, nmrs, nsrs := genNonResourceRule(rng, fmt.Sprintf("%s-%d", pfx, i+1), i == allNonResourceMatcher, someMatchesAllNonResourceRequests)
		nonResourceRules = append(nonResourceRules, nrr)
		valid = valid && nrrValid
		matchingNRIs = append(matchingNRIs, nmrs...)
		skippingNRIs = append(skippingNRIs, nsrs...)
	}
	if nRR == 0 {
		_, _, _, skippingNRIs = genNonResourceRule(rng, pfx+"-o", false, someMatchesAllNonResourceRequests)
	}
	rule := fcv1a1.PolicyRulesWithSubjects{subjects, resourceRules, nonResourceRules}
	t.Logf("For pfx=%s, mayMatchClusterScope=%v, someMatchesAllResourceRequests=%v, someMatchesAllNonResourceRequests=%v, marr=%v, manrr=%v: generated prws=%#+v, valid=%v, mu=%#+v, su=%#+v, mrr=%#+v, mnr=%#+v, srr=%#+v, snr=%#+v", pfx, mayMatchClusterScope, someMatchesAllResourceRequests, someMatchesAllNonResourceRequests, matchAllResourceRequests, matchAllNonResourceRequests, fcfmt.Fmt(rule), valid, fcfmt.Fmt(matchingUIs), fcfmt.Fmt(skippingUIs), fcfmt.Fmt(matchingRRIs), fcfmt.Fmt(matchingNRIs), fcfmt.Fmt(skippingRRIs), fcfmt.Fmt(skippingNRIs))
	matchingRDigests := cross(matchingUIs, matchingRRIs)
	skippingRDigests := append(append(cross(matchingUIs, skippingRRIs),
		cross(skippingUIs, matchingRRIs)...),
		cross(skippingUIs, skippingRRIs)...)
	matchingNDigests := cross(matchingUIs, matchingNRIs)
	skippingNDigests := append(append(cross(matchingUIs, skippingNRIs),
		cross(skippingUIs, matchingNRIs)...),
		cross(skippingUIs, skippingNRIs)...)
	matchingRDigests = shuffleAndTakeDigests(t, rng, &rule, valid, true, matchingRDigests, (1+rng.Intn(2))*(1+rng.Intn(2)))
	skippingRDigests = shuffleAndTakeDigests(t, rng, &rule, valid, false, skippingRDigests, (1+rng.Intn(2))*(1+rng.Intn(2)))
	matchingNDigests = shuffleAndTakeDigests(t, rng, &rule, valid, true, matchingNDigests, (1+rng.Intn(2))*(1+rng.Intn(2)))
	skippingNDigests = shuffleAndTakeDigests(t, rng, &rule, valid, false, skippingNDigests, (1+rng.Intn(2))*(1+rng.Intn(2)))
	return rule, valid, matchingRDigests, matchingNDigests, skippingRDigests, skippingNDigests
}

func cross(uis []user.Info, ris []*request.RequestInfo) []RequestDigest {
	ans := make([]RequestDigest, 0, len(uis)*len(ris))
	for _, ui := range uis {
		for _, ri := range ris {
			ans = append(ans, RequestDigest{RequestInfo: ri, User: ui})
		}
	}
	return ans
}

func shuffleAndTakeDigests(t *testing.T, rng *rand.Rand, rule *fcv1a1.PolicyRulesWithSubjects, valid, toMatch bool, digests []RequestDigest, n int) []RequestDigest {
	ans := make([]RequestDigest, 0, n)
	for len(ans) < n && len(digests) > 0 {
		i := rng.Intn(len(digests))
		digest := digests[i]
		ans = append(ans, digest)
		digests[i] = digests[len(digests)-1]
		digests = digests[:len(digests)-1]
		if rule != nil {
			thisMatches := matchesPolicyRule(digest, rule)
			if toMatch {
				t.Logf("Added matching digest %#+v", digest)
				if valid && !thisMatches {
					t.Logf("Check failure: rule %#+v does not match digest %#+v", fcfmt.Fmt(rule), digest)
				}
			} else {
				t.Logf("Added skipping digest %#+v", digest)
				if valid && thisMatches {
					t.Logf("Check failure: rule %#+v matches digest %#+v", fcfmt.Fmt(rule), digest)
				}
			}
		}
	}
	return ans
}

// genSubject returns a randomly generated Subject that matches on
// some name(s) starting with the given prefix.  The returned bool
// indicates whether the Subject is valid for this package.  The first
// returned list contains members that match the generated Subject and
// involve names that begin with the given prefix.  The second
// returned list contains members that mismatch the generated Subject
// and involve names that begin with the given prefix.
func genSubject(rng *rand.Rand, pfx string) (fcv1a1.Subject, bool, []user.Info, []user.Info) {
	subject := fcv1a1.Subject{}
	valid := true
	var matchingUIs, skippingUIs []user.Info
	x := rng.Float32()
	switch {
	case x < 0.32:
		subject.Kind = fcv1a1.SubjectKindUser
		if rng.Float32() < 0.98 {
			subject.User, matchingUIs, skippingUIs = genUser(rng, pfx)
		}
		if rng.Float32() < 0.02 {
			subject.Group, _, _ = genGroup(rng, pfx)
		}
		if rng.Float32() < 0.02 {
			subject.ServiceAccount, _, _ = genServiceAccount(rng, pfx)
		}
	case x < 0.64:
		subject.Kind = fcv1a1.SubjectKindGroup
		if rng.Float32() < 0.98 {
			subject.Group, matchingUIs, skippingUIs = genGroup(rng, pfx)
		}
		if rng.Float32() < 0.02 {
			subject.User, _, _ = genUser(rng, pfx)
		}
		if rng.Float32() < 0.02 {
			subject.ServiceAccount, _, _ = genServiceAccount(rng, pfx)
		}
	case x < 0.96:
		subject.Kind = fcv1a1.SubjectKindServiceAccount
		if rng.Float32() < 0.98 {
			subject.ServiceAccount, matchingUIs, skippingUIs = genServiceAccount(rng, pfx)
		}
		if rng.Float32() < 0.02 {
			subject.User, _, _ = genUser(rng, pfx)
		}
		if rng.Float32() < 0.02 {
			subject.Group, _, _ = genGroup(rng, pfx)
		}
	default:
		if rng.Float32() < 0.5 {
			subject.Kind = fcv1a1.SubjectKind(pfx + "-k")
		}
		_, _, skippingUIs = genUser(rng, pfx)
	}
	return subject, valid, matchingUIs, skippingUIs
}

func genUser(rng *rand.Rand, pfx string) (*fcv1a1.UserSubject, []user.Info, []user.Info) {
	mui := &user.DefaultInfo{
		Name:   pfx + "-u",
		UID:    "good-id",
		Groups: []string{pfx + "-g1", mg(rng), pfx + "-g2"},
		Extra:  noextra}
	skips := []user.Info{&user.DefaultInfo{
		Name:   mui.Name + "x",
		UID:    mui.UID,
		Groups: mui.Groups,
		Extra:  mui.Extra}}
	return &fcv1a1.UserSubject{mui.Name}, []user.Info{mui}, skips
}

var groupCover = []string{"system:authenticated", "system:unauthenticated"}

func mg(rng *rand.Rand) string {
	return groupCover[rng.Intn(len(groupCover))]
}

func mkGroupSubject(group string) fcv1a1.Subject {
	return fcv1a1.Subject{
		Kind:  fcv1a1.SubjectKindGroup,
		Group: &fcv1a1.GroupSubject{group},
	}
}

func genGroup(rng *rand.Rand, pfx string) (*fcv1a1.GroupSubject, []user.Info, []user.Info) {
	name := pfx + "-g"
	ui := &user.DefaultInfo{
		Name:   pfx + "-u",
		UID:    "good-id",
		Groups: []string{name},
		Extra:  noextra}
	if rng.Intn(2) == 0 {
		ui.Groups = append([]string{mg(rng)}, ui.Groups...)
	} else {
		ui.Groups = append(ui.Groups, mg(rng))
	}
	if rng.Intn(3) == 0 {
		ui.Groups = append([]string{pfx + "-h"}, ui.Groups...)
	}
	if rng.Intn(3) == 0 {
		ui.Groups = append(ui.Groups, pfx+"-i")
	}
	skipper := &user.DefaultInfo{
		Name:   pfx + "-u",
		UID:    "bad-id",
		Groups: []string{pfx + "-j", mg(rng)},
		Extra:  noextra}
	if rng.Intn(2) == 0 {
		skipper.Groups = append(skipper.Groups, pfx+"-k")
	}
	return &fcv1a1.GroupSubject{name}, []user.Info{ui}, []user.Info{skipper}
}

func genServiceAccount(rng *rand.Rand, pfx string) (*fcv1a1.ServiceAccountSubject, []user.Info, []user.Info) {
	ns := pfx + "-ns"
	name := pfx + "-n"
	mname := name
	if rng.Float32() < 0.05 {
		mname = "*"
	}
	mui := &user.DefaultInfo{
		Name:   fmt.Sprintf("system:serviceaccount:%s:%s", ns, name),
		UID:    "good-id",
		Groups: []string{pfx + "-g1", mg(rng), pfx + "-g2"},
		Extra:  noextra}
	var skips []user.Info
	if mname == "*" || rng.Intn(2) == 0 {
		skips = []user.Info{&user.DefaultInfo{
			Name:   fmt.Sprintf("system:serviceaccount:%sx:%s", ns, name),
			UID:    "bad-id",
			Groups: mui.Groups,
			Extra:  mui.Extra}}
	} else {
		skips = []user.Info{&user.DefaultInfo{
			Name:   fmt.Sprintf("system:serviceaccount:%s:%sx", ns, name),
			UID:    "bad-id",
			Groups: mui.Groups,
			Extra:  mui.Extra}}
	}
	return &fcv1a1.ServiceAccountSubject{Namespace: ns, Name: mname}, []user.Info{mui}, skips
}

// genResourceRule randomly generates a ResourcePolicyRule, a bool
// indicating whether than rule is valid for this package, and lists
// of matching and non-matching `*request.RequestInfo`.
func genResourceRule(rng *rand.Rand, pfx string, mayMatchClusterScope, matchAllResources, someMatchesAllResources bool) (fcv1a1.ResourcePolicyRule, bool, []*request.RequestInfo, []*request.RequestInfo) {
	valid := true
	namespaces := []string{pfx + "-n1", pfx + "-n2", pfx + "-n3"}
	rnamespaces := namespaces
	if mayMatchClusterScope && rng.Float32() < 0.1 {
		namespaces[0] = ""
		rnamespaces = namespaces[1:]
	}
	rr := fcv1a1.ResourcePolicyRule{
		Verbs:        []string{pfx + "-v1", pfx + "-v2", pfx + "-v3"},
		APIGroups:    []string{pfx + ".g1", pfx + ".g2", pfx + ".g3"},
		Resources:    []string{pfx + "-r1s", pfx + "-r2s", pfx + "-r3s"},
		ClusterScope: namespaces[0] == "",
		Namespaces:   rnamespaces}
	matchingRIs := genRRIs(rng, 3, rr.Verbs, rr.APIGroups, rr.Resources, namespaces)
	var skippingRIs []*request.RequestInfo
	if !someMatchesAllResources {
		skipNSs := []string{pfx + "-n4", pfx + "-n5", pfx + "-n6"}
		if mayMatchClusterScope && rr.Namespaces[0] != "" && rng.Float32() < 0.1 {
			skipNSs[0] = ""
		}
		skippingRIs = genRRIs(rng, 3,
			[]string{pfx + "-v4", pfx + "-v5", pfx + "-v6"},
			[]string{pfx + ".g4", pfx + ".g5", pfx + ".g6"},
			[]string{pfx + "-r4s", pfx + "-r5s", pfx + "-r6s"},
			skipNSs)
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
		rr.ClusterScope = true
		rr.Namespaces = []string{fcv1a1.NamespaceEvery}
	}
	return rr, valid, matchingRIs, skippingRIs
}

func genRRIs(rng *rand.Rand, m int, verbs, apiGroups, resources, namespaces []string) []*request.RequestInfo {
	nv := len(verbs)
	ng := len(apiGroups)
	nr := len(resources)
	nn := len(namespaces)
	coords := chooseInts(rng, nv*ng*nr*nn, m)
	ans := make([]*request.RequestInfo, 0, m)
	for _, coord := range coords {
		ans = append(ans, &request.RequestInfo{
			IsResourceRequest: true,
			Verb:              verbs[coord%nv],
			APIGroup:          apiGroups[coord/nv%ng],
			Resource:          resources[coord/nv/ng%nr],
			Namespace:         namespaces[coord/nv/ng/nr]})
	}
	return ans
}

func genNRRIs(rng *rand.Rand, m int, verbs, urls []string) []*request.RequestInfo {
	nv := len(verbs)
	nu := len(urls)
	coords := chooseInts(rng, nv*nu, m)
	ans := make([]*request.RequestInfo, 0, m)
	for _, coord := range coords {
		ans = append(ans, &request.RequestInfo{
			IsResourceRequest: false,
			Verb:              verbs[coord%nv],
			Path:              urls[coord/nv]})
	}
	return ans
}

func chooseInts(rng *rand.Rand, n, m int) []int {
	ans := sets.NewInt()
	for len(ans) < m {
		i := rng.Intn(n)
		if ans.Has(i) {
			continue
		}
		ans.Insert(i)
	}
	return ans.List()
}

// genNonResourceRule returns a randomly generated
// NonResourcePolicyRule, a bool indicating whether that rule is valid
// for this package, and lists of matching and non-matching
// `*request.RequestInfo`.
func genNonResourceRule(rng *rand.Rand, pfx string, matchAllNonResources, someMatchesAllNonResources bool) (fcv1a1.NonResourcePolicyRule, bool, []*request.RequestInfo, []*request.RequestInfo) {
	nrr := fcv1a1.NonResourcePolicyRule{
		Verbs:           []string{pfx + "-v1", pfx + "-v2", pfx + "-v3"},
		NonResourceURLs: []string{"/" + pfx + "/p1", "/" + pfx + "/p2", "/" + pfx + "/p3"},
	}
	valid := true
	matchingRIs := genNRRIs(rng, 3, nrr.Verbs, nrr.NonResourceURLs)
	var skippingRIs []*request.RequestInfo
	if !someMatchesAllNonResources {
		skippingRIs = genNRRIs(rng, 3,
			[]string{pfx + "-v4", pfx + "-v5", pfx + "-v6"},
			[]string{"/" + pfx + "/p4", "/" + pfx + "/p5", "/" + pfx + "/p6"})
	}
	// choose a proper subset of fields to consider wildcarding; only matters if not matching all
	starMask := rng.Intn(3)
	if matchAllNonResources || starMask&1 == 1 && rng.Float32() < 0.1 {
		nrr.Verbs = []string{fcv1a1.VerbAll}
	}
	if matchAllNonResources || starMask&2 == 2 && rng.Float32() < 0.1 {
		nrr.NonResourceURLs = []string{"*"}
	}
	return nrr, valid, matchingRIs, skippingRIs
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
