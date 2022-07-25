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
	"sync/atomic"
	"testing"
	"time"

	flowcontrol "k8s.io/api/flowcontrol/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	fqtesting "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing"
	fcfmt "k8s.io/apiserver/pkg/util/flowcontrol/format"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
)

var noRestraintQSF = fqtesting.NewNoRestraintFactory()

// genPL creates a valid PriorityLevelConfiguration with the given
// name and randomly generated spec.  The given name must not be one
// of the mandatory ones.
func genPL(rng *rand.Rand, name string) *flowcontrol.PriorityLevelConfiguration {
	plc := &flowcontrol.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: flowcontrol.PriorityLevelConfigurationSpec{
			Type: flowcontrol.PriorityLevelEnablementLimited,
			Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: rng.Int31n(100) + 1,
				LimitResponse: flowcontrol.LimitResponse{
					Type: flowcontrol.LimitResponseTypeReject}}}}
	if rng.Float32() < 0.95 {
		plc.Spec.Limited.LimitResponse.Type = flowcontrol.LimitResponseTypeQueue
		hs := rng.Int31n(5) + 1
		plc.Spec.Limited.LimitResponse.Queuing = &flowcontrol.QueuingConfiguration{
			Queues:           hs + rng.Int31n(20),
			HandSize:         hs,
			QueueLengthLimit: 5}
	}
	labelVals := []string{"test"}
	_, err := queueSetCompleterForPL(noRestraintQSF, nil, plc, time.Minute, metrics.RatioedGaugeVecPhasedElementPair(metrics.PriorityLevelConcurrencyGaugeVec, 1, 1, labelVals), metrics.PriorityLevelExecutionSeatsGaugeVec.NewForLabelValuesSafe(0, 1, labelVals))
	if err != nil {
		panic(err)
	}
	return plc
}

// A FlowSchema together with characteristics relevant to testing
type fsTestingRecord struct {
	fs *flowcontrol.FlowSchema
	// Does this reference an existing priority level?
	wellFormed                    bool
	matchesAllResourceRequests    bool
	matchesAllNonResourceRequests bool
	// maps `matches bool` to `isResourceRequest bool` to digests
	digests map[bool]map[bool][]RequestDigest
}

func (ftr *fsTestingRecord) addDigest(digest RequestDigest, matches bool) {
	ftr.digests[matches][digest.RequestInfo.IsResourceRequest] = append(ftr.digests[matches][digest.RequestInfo.IsResourceRequest], digest)
}

func (ftr *fsTestingRecord) addDigests(digests []RequestDigest, matches bool) {
	for _, digest := range digests {
		ftr.addDigest(digest, matches)
	}
}

var flowDistinguisherMethodTypes = sets.NewString(
	string(flowcontrol.FlowDistinguisherMethodByUserType),
	string(flowcontrol.FlowDistinguisherMethodByNamespaceType),
)

var mandFTRExempt = &fsTestingRecord{
	fs:         fcboot.MandatoryFlowSchemaExempt,
	wellFormed: true,
	digests: map[bool]map[bool][]RequestDigest{
		false: {
			false: {{
				RequestInfo: &request.RequestInfo{
					IsResourceRequest: false,
					Path:              "/foo/bar",
					Verb:              "frobulate"},
				User: &user.DefaultInfo{
					Name:   "nobody",
					Groups: []string{user.AllAuthenticated, "nogroup"},
				},
			}},
			true: {{
				RequestInfo: &request.RequestInfo{
					IsResourceRequest: true,
					Verb:              "mandate",
					APIGroup:          "nogroup",
					Namespace:         "nospace",
					Resource:          "nons",
				},
				User: &user.DefaultInfo{
					Name:   "nobody",
					Groups: []string{user.AllAuthenticated, "nogroup"},
				},
			}},
		},
		true: {
			false: {{
				RequestInfo: &request.RequestInfo{
					IsResourceRequest: false,
					Path:              "/foo/bar",
					Verb:              "frobulate"},
				User: &user.DefaultInfo{
					Name:   "nobody",
					Groups: []string{user.AllAuthenticated, user.SystemPrivilegedGroup},
				},
			}},
			true: {{
				RequestInfo: &request.RequestInfo{
					IsResourceRequest: true,
					Verb:              "mandate",
					APIGroup:          "nogroup",
					Namespace:         "nospace",
					Resource:          "nons",
				},
				User: &user.DefaultInfo{
					Name:   "nobody",
					Groups: []string{user.AllAuthenticated, user.SystemPrivilegedGroup},
				},
			}},
		},
	},
}

var mandFTRCatchAll = &fsTestingRecord{
	fs:         fcboot.MandatoryFlowSchemaCatchAll,
	wellFormed: true,
	digests: map[bool]map[bool][]RequestDigest{
		false: {},
		true: {
			false: {{
				RequestInfo: &request.RequestInfo{
					IsResourceRequest: false,
					Path:              "/foo/bar",
					Verb:              "frobulate"},
				User: &user.DefaultInfo{
					Name:   "nobody",
					Groups: []string{user.AllAuthenticated, "nogroup"},
				},
			}},
			true: {{
				RequestInfo: &request.RequestInfo{
					IsResourceRequest: true,
					Verb:              "mandate",
					APIGroup:          "nogroup",
					Namespace:         "nospace",
					Resource:          "nons",
				},
				User: &user.DefaultInfo{
					Name:   "nobody",
					Groups: []string{user.AllAuthenticated, "nogroup"},
				},
			}},
		},
	},
}

// genFS creates a valid FlowSchema with the given name and randomly
// generated spec, along with characteristics relevant to testing.
// When all the FlowSchemas in a collection are generated with
// different names: (a) the matching digests match only the schema for
// which they were generated, and (b) the non-matching digests do not
// match any schema in the collection.  The generated spec is
// relatively likely to be well formed but might not be.  An ill
// formed spec references a priority level drawn from badPLNames.
// goodPLNames may be empty, but badPLNames may not.
func genFS(t *testing.T, rng *rand.Rand, name string, mayMatchClusterScope bool, goodPLNames, badPLNames sets.String) *fsTestingRecord {
	fs := &flowcontrol.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       flowcontrol.FlowSchemaSpec{}}
	// 5% chance of zero rules, otherwise draw from 1--6 biased low
	nRules := (1 + rng.Intn(3)) * (1 + rng.Intn(2)) * ((19 + rng.Intn(20)) / 20)
	ftr := &fsTestingRecord{fs: fs,
		wellFormed:                    true,
		matchesAllResourceRequests:    nRules > 0 && rng.Float32() < 0.1,
		matchesAllNonResourceRequests: nRules > 0 && rng.Float32() < 0.1,
		digests: map[bool]map[bool][]RequestDigest{
			false: {false: {}, true: {}},
			true:  {false: {}, true: {}}},
	}
	dangleStatus := flowcontrol.ConditionFalse
	if rng.Float32() < 0.9 && len(goodPLNames) > 0 {
		fs.Spec.PriorityLevelConfiguration = flowcontrol.PriorityLevelConfigurationReference{pickSetString(rng, goodPLNames)}
	} else {
		fs.Spec.PriorityLevelConfiguration = flowcontrol.PriorityLevelConfigurationReference{pickSetString(rng, badPLNames)}
		ftr.wellFormed = false
		dangleStatus = flowcontrol.ConditionTrue
	}
	fs.Status.Conditions = []flowcontrol.FlowSchemaCondition{{
		Type:   flowcontrol.FlowSchemaConditionDangling,
		Status: dangleStatus}}
	fs.Spec.MatchingPrecedence = rng.Int31n(9997) + 2
	if rng.Float32() < 0.8 {
		fdmt := flowcontrol.FlowDistinguisherMethodType(pickSetString(rng, flowDistinguisherMethodTypes))
		fs.Spec.DistinguisherMethod = &flowcontrol.FlowDistinguisherMethod{fdmt}
	}
	fs.Spec.Rules = []flowcontrol.PolicyRulesWithSubjects{}
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
		rule, ruleMatchingRDigests, ruleMatchingNDigests, ruleSkippingRDigests, ruleSkippingNDigests := genPolicyRuleWithSubjects(t, rng, fmt.Sprintf("%s-%d", name, i+1), mayMatchClusterScope, ftr.matchesAllResourceRequests, ftr.matchesAllNonResourceRequests, i == everyResourceMatcher, i == everyNonResourceMatcher)
		fs.Spec.Rules = append(fs.Spec.Rules, rule)
		ftr.addDigests(ruleMatchingRDigests, true)
		ftr.addDigests(ruleMatchingNDigests, true)
		ftr.addDigests(ruleSkippingRDigests, false)
		ftr.addDigests(ruleSkippingNDigests, false)
	}
	if nRules == 0 {
		var skippingRDigests, skippingNDigests []RequestDigest
		_, _, _, skippingRDigests, skippingNDigests = genPolicyRuleWithSubjects(t, rng, name+"-1", false, false, false, false, false)
		ftr.addDigests(skippingRDigests, false)
		ftr.addDigests(skippingNDigests, false)
	}
	if testDebugLogs {
		t.Logf("Returning name=%s, plRef=%q, wellFormed=%v, matchesAllResourceRequests=%v, matchesAllNonResourceRequests=%v for mayMatchClusterScope=%v", fs.Name, fs.Spec.PriorityLevelConfiguration.Name, ftr.wellFormed, ftr.matchesAllResourceRequests, ftr.matchesAllNonResourceRequests, mayMatchClusterScope)
	}
	return ftr
}

var noextra = make(map[string][]string)

// Generate one valid PolicyRulesWithSubjects.  Also returns: matching
// resource-style digests, matching non-resource-style digests,
// skipping (i.e., not matching the generated rule) resource-style
// digests, skipping non-resource-style digests.  When a collection of
// rules is generated with unique prefixes, the skipping digests for
// each rule match no rules in the collection.  The
// someMatchesAllResourceRequests and
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
func genPolicyRuleWithSubjects(t *testing.T, rng *rand.Rand, pfx string, mayMatchClusterScope, someMatchesAllResourceRequests, someMatchesAllNonResourceRequests, matchAllResourceRequests, matchAllNonResourceRequests bool) (flowcontrol.PolicyRulesWithSubjects, []RequestDigest, []RequestDigest, []RequestDigest, []RequestDigest) {
	subjects := []flowcontrol.Subject{}
	matchingUIs := []user.Info{}
	skippingUIs := []user.Info{}
	resourceRules := []flowcontrol.ResourcePolicyRule{}
	nonResourceRules := []flowcontrol.NonResourcePolicyRule{}
	matchingRRIs := []*request.RequestInfo{}
	skippingRRIs := []*request.RequestInfo{}
	matchingNRIs := []*request.RequestInfo{}
	skippingNRIs := []*request.RequestInfo{}
	nSubj := rng.Intn(4)
	for i := 0; i < nSubj; i++ {
		subject, smus, ssus := genSubject(rng, fmt.Sprintf("%s-%d", pfx, i+1))
		subjects = append(subjects, subject)
		matchingUIs = append(matchingUIs, smus...)
		skippingUIs = append(skippingUIs, ssus...)
	}
	if matchAllResourceRequests || matchAllNonResourceRequests {
		switch rng.Intn(3) {
		case 0:
			subjects = append(subjects, mkUserSubject("*"))
		case 1:
			subjects = append(subjects, mkGroupSubject("*"))
		default:
			subjects = append(subjects, mkGroupSubject("system:authenticated"), mkGroupSubject("system:unauthenticated"))
		}
		matchingUIs = append(matchingUIs, skippingUIs...)
	}
	if someMatchesAllResourceRequests || someMatchesAllNonResourceRequests {
		skippingUIs = []user.Info{}
	} else if nSubj == 0 {
		_, _, skippingUIs = genSubject(rng, pfx+"-o")
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
		rr, rmrs, rsrs := genResourceRule(rng, fmt.Sprintf("%s-%d", pfx, i+1), mayMatchClusterScope, i == allResourceMatcher, someMatchesAllResourceRequests)
		resourceRules = append(resourceRules, rr)
		matchingRRIs = append(matchingRRIs, rmrs...)
		skippingRRIs = append(skippingRRIs, rsrs...)
	}
	if nRR == 0 {
		_, _, skippingRRIs = genResourceRule(rng, pfx+"-o", mayMatchClusterScope, false, someMatchesAllResourceRequests)
	}
	allNonResourceMatcher := -1
	if matchAllNonResourceRequests {
		allNonResourceMatcher = rng.Intn(nNRR)
	}
	for i := 0; i < nNRR; i++ {
		nrr, nmrs, nsrs := genNonResourceRule(rng, fmt.Sprintf("%s-%d", pfx, i+1), i == allNonResourceMatcher, someMatchesAllNonResourceRequests)
		nonResourceRules = append(nonResourceRules, nrr)
		matchingNRIs = append(matchingNRIs, nmrs...)
		skippingNRIs = append(skippingNRIs, nsrs...)
	}
	if nRR == 0 {
		_, _, skippingNRIs = genNonResourceRule(rng, pfx+"-o", false, someMatchesAllNonResourceRequests)
	}
	rule := flowcontrol.PolicyRulesWithSubjects{subjects, resourceRules, nonResourceRules}
	if testDebugLogs {
		t.Logf("For pfx=%s, mayMatchClusterScope=%v, someMatchesAllResourceRequests=%v, someMatchesAllNonResourceRequests=%v, marr=%v, manrr=%v: generated prws=%s, mu=%s, su=%s, mrr=%s, mnr=%s, srr=%s, snr=%s", pfx, mayMatchClusterScope, someMatchesAllResourceRequests, someMatchesAllNonResourceRequests, matchAllResourceRequests, matchAllNonResourceRequests, fcfmt.Fmt(rule), fcfmt.Fmt(matchingUIs), fcfmt.Fmt(skippingUIs), fcfmt.Fmt(matchingRRIs), fcfmt.Fmt(matchingNRIs), fcfmt.Fmt(skippingRRIs), fcfmt.Fmt(skippingNRIs))
	}
	matchingRDigests := cross(matchingUIs, matchingRRIs)
	skippingRDigests := append(append(cross(matchingUIs, skippingRRIs),
		cross(skippingUIs, matchingRRIs)...),
		cross(skippingUIs, skippingRRIs)...)
	matchingNDigests := cross(matchingUIs, matchingNRIs)
	skippingNDigests := append(append(cross(matchingUIs, skippingNRIs),
		cross(skippingUIs, matchingNRIs)...),
		cross(skippingUIs, skippingNRIs)...)
	matchingRDigests = shuffleAndTakeDigests(t, rng, &rule, true, matchingRDigests, (1+rng.Intn(2))*(1+rng.Intn(2)))
	skippingRDigests = shuffleAndTakeDigests(t, rng, &rule, false, skippingRDigests, (1+rng.Intn(2))*(1+rng.Intn(2)))
	matchingNDigests = shuffleAndTakeDigests(t, rng, &rule, true, matchingNDigests, (1+rng.Intn(2))*(1+rng.Intn(2)))
	skippingNDigests = shuffleAndTakeDigests(t, rng, &rule, false, skippingNDigests, (1+rng.Intn(2))*(1+rng.Intn(2)))
	return rule, matchingRDigests, matchingNDigests, skippingRDigests, skippingNDigests
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

func shuffleAndTakeDigests(t *testing.T, rng *rand.Rand, rule *flowcontrol.PolicyRulesWithSubjects, toMatch bool, digests []RequestDigest, n int) []RequestDigest {
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
				if testDebugLogs {
					t.Logf("Added matching digest %#+v", digest)
				}
				if !thisMatches {
					t.Errorf("Fail in check: rule %s does not match digest %#+v", fcfmt.Fmt(rule), digest)
				}
			} else {
				if testDebugLogs {
					t.Logf("Added skipping digest %#+v", digest)
				}
				if thisMatches {
					t.Errorf("Fail in check: rule %s matches digest %#+v", fcfmt.Fmt(rule), digest)
				}
			}
		}
	}
	return ans
}

var uCounter uint32 = 1

func uniqify(in RequestDigest) RequestDigest {
	u1 := in.User.(*user.DefaultInfo)
	u2 := *u1
	u2.Extra = map[string][]string{"u": {fmt.Sprintf("z%d", atomic.AddUint32(&uCounter, 1))}}
	return RequestDigest{User: &u2, RequestInfo: in.RequestInfo}
}

// genSubject returns a randomly generated valid Subject that matches
// on some name(s) starting with the given prefix.  The first returned
// list contains members that match the generated Subject and involve
// names that begin with the given prefix.  The second returned list
// contains members that mismatch the generated Subject and involve
// names that begin with the given prefix.
func genSubject(rng *rand.Rand, pfx string) (flowcontrol.Subject, []user.Info, []user.Info) {
	subject := flowcontrol.Subject{}
	var matchingUIs, skippingUIs []user.Info
	x := rng.Float32()
	switch {
	case x < 0.33:
		subject.Kind = flowcontrol.SubjectKindUser
		subject.User, matchingUIs, skippingUIs = genUser(rng, pfx)
	case x < 0.67:
		subject.Kind = flowcontrol.SubjectKindGroup
		subject.Group, matchingUIs, skippingUIs = genGroup(rng, pfx)
	default:
		subject.Kind = flowcontrol.SubjectKindServiceAccount
		subject.ServiceAccount, matchingUIs, skippingUIs = genServiceAccount(rng, pfx)
	}
	return subject, matchingUIs, skippingUIs
}

func genUser(rng *rand.Rand, pfx string) (*flowcontrol.UserSubject, []user.Info, []user.Info) {
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
	return &flowcontrol.UserSubject{mui.Name}, []user.Info{mui}, skips
}

var groupCover = []string{"system:authenticated", "system:unauthenticated"}

func mg(rng *rand.Rand) string {
	return groupCover[rng.Intn(len(groupCover))]
}

func mkUserSubject(username string) flowcontrol.Subject {
	return flowcontrol.Subject{
		Kind: flowcontrol.SubjectKindUser,
		User: &flowcontrol.UserSubject{username},
	}
}

func mkGroupSubject(group string) flowcontrol.Subject {
	return flowcontrol.Subject{
		Kind:  flowcontrol.SubjectKindGroup,
		Group: &flowcontrol.GroupSubject{group},
	}
}

func genGroup(rng *rand.Rand, pfx string) (*flowcontrol.GroupSubject, []user.Info, []user.Info) {
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
	return &flowcontrol.GroupSubject{name}, []user.Info{ui}, []user.Info{skipper}
}

func genServiceAccount(rng *rand.Rand, pfx string) (*flowcontrol.ServiceAccountSubject, []user.Info, []user.Info) {
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
	return &flowcontrol.ServiceAccountSubject{Namespace: ns, Name: mname}, []user.Info{mui}, skips
}

// genResourceRule randomly generates a valid ResourcePolicyRule and lists
// of matching and non-matching `*request.RequestInfo`.
func genResourceRule(rng *rand.Rand, pfx string, mayMatchClusterScope, matchAllResources, someMatchesAllResources bool) (flowcontrol.ResourcePolicyRule, []*request.RequestInfo, []*request.RequestInfo) {
	namespaces := []string{pfx + "-n1", pfx + "-n2", pfx + "-n3"}
	rnamespaces := namespaces
	if mayMatchClusterScope && rng.Float32() < 0.1 {
		namespaces[0] = ""
		rnamespaces = namespaces[1:]
	}
	rr := flowcontrol.ResourcePolicyRule{
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
		rr.Verbs = []string{flowcontrol.VerbAll}
	}
	if matchAllResources || starMask&2 == 2 && rng.Float32() < 0.1 {
		rr.APIGroups = []string{flowcontrol.APIGroupAll}
	}
	if matchAllResources || starMask&4 == 4 && rng.Float32() < 0.1 {
		rr.Resources = []string{flowcontrol.ResourceAll}
	}
	if matchAllResources || starMask&8 == 8 && rng.Float32() < 0.1 {
		rr.ClusterScope = true
		rr.Namespaces = []string{flowcontrol.NamespaceEvery}
	}
	return rr, matchingRIs, skippingRIs
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
		ri := &request.RequestInfo{
			IsResourceRequest: false,
			Verb:              verbs[coord%nv],
			Path:              urls[coord/nv]}
		if rng.Intn(2) == 1 {
			ri.Path = ri.Path + "/more"
		}
		ans = append(ans, ri)
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

// genNonResourceRule returns a randomly generated valid
// NonResourcePolicyRule and lists of matching and non-matching
// `*request.RequestInfo`.
func genNonResourceRule(rng *rand.Rand, pfx string, matchAllNonResources, someMatchesAllNonResources bool) (flowcontrol.NonResourcePolicyRule, []*request.RequestInfo, []*request.RequestInfo) {
	nrr := flowcontrol.NonResourcePolicyRule{
		Verbs:           []string{pfx + "-v1", pfx + "-v2", pfx + "-v3"},
		NonResourceURLs: []string{"/" + pfx + "/g/p1", "/" + pfx + "/g/p2", "/" + pfx + "/g/p3"},
	}
	matchingRIs := genNRRIs(rng, 3, nrr.Verbs, nrr.NonResourceURLs)
	var skippingRIs []*request.RequestInfo
	if !someMatchesAllNonResources {
		skippingRIs = genNRRIs(rng, 3,
			[]string{pfx + "-v4", pfx + "-v5", pfx + "-v6"},
			[]string{"/" + pfx + "/b/p1", "/" + pfx + "/b/p2", "/" + pfx + "/b/p3"})
	}
	// choose a proper subset of fields to consider wildcarding; only matters if not matching all
	starMask := rng.Intn(3)
	if matchAllNonResources || starMask&1 == 1 && rng.Float32() < 0.1 {
		nrr.Verbs = []string{flowcontrol.VerbAll}
	}
	if matchAllNonResources || starMask&2 == 2 && rng.Float32() < 0.1 {
		nrr.NonResourceURLs = []string{"*"}
	} else {
		nrr.NonResourceURLs[rng.Intn(3)] = "/" + pfx + "/g/*"
		nrr.NonResourceURLs[rng.Intn(3)] = "/" + pfx + "/g"
	}
	return nrr, matchingRIs, skippingRIs
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
