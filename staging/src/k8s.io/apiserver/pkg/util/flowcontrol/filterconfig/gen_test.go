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

	fcv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
)

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
				string(fcv1a1.FlowDistinguisherMethodByUserType),
				string(fcv1a1.FlowDistinguisherMethodByUserType),
				string(fcv1a1.FlowDistinguisherMethodByUserType),
				string(fcv1a1.FlowDistinguisherMethodByUserType),
				string(fcv1a1.FlowDistinguisherMethodByNamespaceType),
				string(fcv1a1.FlowDistinguisherMethodByNamespaceType),
				string(fcv1a1.FlowDistinguisherMethodByNamespaceType),
				string(fcv1a1.FlowDistinguisherMethodByNamespaceType),
				"fubar"))
		valid = valid && string(fdmt) != "fubar"
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
	t.Logf("Returning valid=%v, matchEveryResourceRequest=%v, matchEveryNonResourceRequest=%v", valid, matchEveryResourceRequest, matchEveryNonResourceRequest)
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
		skippingReqs = append(skippingReqs, &request.RequestInfo{IsResourceRequest: true, Path: "/api/v1/namespaces/example-com/pets", Verb: "mash", APIPrefix: "api", APIVersion: "v1", Namespace: "fredz", Resource: "pets", Parts: []string{"pets"}})
	}
	if !someMatchesAllNonResourceRequests {
		skippingReqs = append(skippingReqs, &request.RequestInfo{IsResourceRequest: false, Path: "/bog/us", Verb: "mash", Parts: nil})
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
	t.Logf("For someMatchesAllResourceRequests=%v, someMatchesAllNonResourceRequests=%v: generated prws=%s, valid=%v, marr=%v, manrr=%v, mu=%s, su=%s, mr=%s, sr=%s", someMatchesAllResourceRequests, someMatchesAllNonResourceRequests, FmtPolicyRule(rule), valid, matchAllResourceRequests, matchAllNonResourceRequests, FmtUsers(matchingUsers), FmtUsers(skippingUsers), FmtRequests(matchingReqs), FmtRequests(skippingReqs))
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
			t.Logf("Check failure: rule %s does not match digest %#+v", FmtPolicyRule(rule), rd)
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
			t.Logf("Check failure: rule %s matches digest %#+v", FmtPolicyRule(rule), rd)
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
		Verbs:        ([]string{"create", "get", "list"})[:1+rng.Intn(3)],
		APIGroups:    ([]string{"", "apps", "storage.k8s.io", "etcd.database.coreos.com"})[:1+rng.Intn(4)],
		Resources:    ([]string{"apples", "bananas", "cats"})[:1+rng.Intn(3)],
		ClusterScope: matchAllResources || rng.Intn(2) == 0,
		Namespaces: ([]string{fmt.Sprintf("ns%d", rng.Uint64()),
			fmt.Sprintf("ns%d", rng.Uint64()),
			fmt.Sprintf("ns%d", rng.Uint64())})[:nns],
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
			ri := &request.RequestInfo{
				IsResourceRequest: true,
				Path:              ([]string{"/api/v1/services", "/openapi/v2", "/apis/authentication.k8s.io/v1/tokenreviews"})[rng.Intn(3)],
				Verb:              ([]string{"update", "frob", "delete"})[rng.Intn(3)],
				APIPrefix:         "apis",
				APIGroup:          ([]string{"authentication.k8s.io", "fubar", "blogs", "foo.com"})[rng.Intn(4)],
				APIVersion:        "v1",
				Resource:          ([]string{"dogs", "eels", "fish", "goats"})[rng.Intn(4)],
				Subresource:       "", Name: "", Parts: []string{""}}
			ri.Parts[0] = ri.Resource
			if rng.Float32() < 0.3 {
				ri.Subresource = "status"
				ri.Parts = append(ri.Parts, "status")
			}
			if rng.Float32() < 0.5 || rr.ClusterScope {
				ri.Namespace = fmt.Sprintf("ms%d", rng.Uint64())
			}
			skippingReqs = append(skippingReqs, ri)
		} else if !otherMatchesAllNonResources {
			// Add a non-resource request that mismatches every
			// generated non-resource rule on every field, so that
			// wildcards on a subset of fields will not defeat the
			// mismatch
			skippingReqs = append(skippingReqs, &request.RequestInfo{
				IsResourceRequest: false,
				Path:              ([]string{"/api", "/openapi/v2", "/apis/coordination.k8s.io/v1beta1"})[rng.Intn(3)],
				Verb:              ([]string{"patch", "post", "delete"})[rng.Intn(3)],
				APIPrefix:         "", APIGroup: "",
				APIVersion: "", Namespace: "",
				Resource: "", Subresource: "", Name: "", Parts: nil})
		}
	}
	if len(rr.Namespaces) > 0 || rr.ClusterScope {
		nMatch := 1 + rng.Intn(3)
		for i := 0; i < nMatch; i++ {
			ri := &request.RequestInfo{
				IsResourceRequest: true,
				Path:              "/api/v1" + rr.Resources[0],
				Verb:              rr.Verbs[rng.Intn(len(rr.Verbs))],
				APIPrefix:         "api",
				APIGroup:          rr.APIGroups[rng.Intn(len(rr.APIGroups))],
				APIVersion:        "v1",
				Resource:          rr.Resources[rng.Intn(len(rr.Resources))],
				Subresource:       "", Name: "", Parts: []string{""}}
			if rr.Verbs[0] == fcv1a1.VerbAll {
				ri.Verb = ([]string{"update", "frob", "delete"})[rng.Intn(3)]
			}
			if rr.APIGroups[0] == fcv1a1.APIGroupAll {
				ri.APIGroup = ([]string{"authentication.k8s.io", "fubar", "blogs", "foo.com"})[rng.Intn(4)]
			}
			if rr.Resources[0] == fcv1a1.ResourceAll {
				ri.Resource = ([]string{"dogs", "eels", "fish", "goats"})[rng.Intn(4)]
			}
			if len(rr.Namespaces) == 0 {
				// can only match non-namespaced requests
			} else if rr.Namespaces[0] == fcv1a1.NamespaceEvery {
				ri.Namespace = fmt.Sprintf("ns%d", rng.Uint64())
			} else {
				ri.Namespace = rr.Namespaces[rng.Intn(len(rr.Namespaces))]
			}
			matchingReqs = append(matchingReqs, ri)
		}
	}
	return rr, valid, matchingReqs, skippingReqs
}

// genNonResourceRule returns a randomly generated
// NonResourcePolicyRule, a bool indicating whether that rule is valid
// for this package, and lists of matching and non-matching
// `*request.RequestInfo`.
func genNonResourceRule(rng *rand.Rand, matchAllNonResources, otherMatchesAllResources, someMatchesAllNonResources bool) (fcv1a1.NonResourcePolicyRule, bool, []*request.RequestInfo, []*request.RequestInfo) {
	nrr := fcv1a1.NonResourcePolicyRule{
		Verbs:           ([]string{"get", "head", "put"})[:1+rng.Intn(3)],
		NonResourceURLs: ([]string{"/healthz", "/metrics", "/go"})[:1+rng.Intn(3)],
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
			skippingReqs = append(skippingReqs, &request.RequestInfo{
				IsResourceRequest: false,
				Path:              ([]string{"/api", "/openapi/v2", "/apis/coordination.k8s.io/v1beta1"})[rng.Intn(3)],
				Verb:              ([]string{"patch", "post", "delete"})[rng.Intn(3)],
				APIPrefix:         "", APIGroup: "",
				APIVersion: "", Namespace: "",
				Resource: "", Subresource: "", Name: "", Parts: nil})
		} else if !otherMatchesAllResources {
			// Add a resource request that mismatches every generated resource
			// rule on every field, so that wildcards on a subset of
			// fields will not defeat the mismatch
			ri := &request.RequestInfo{
				IsResourceRequest: true,
				Path:              ([]string{"/api/v1/services", "/openapi/v2", "/apis/authentication.k8s.io/v1/tokenreviews"})[rng.Intn(3)],
				Verb:              ([]string{"update", "frob", "delete"})[rng.Intn(3)],
				APIPrefix:         "apis",
				APIGroup:          ([]string{"authentication.k8s.io", "fubar", "blogs", "foo.com"})[rng.Intn(4)],
				APIVersion:        "v1",
				Resource:          ([]string{"dogs", "eels", "fish", "goats"})[rng.Intn(4)],
				Subresource:       "", Name: "", Parts: []string{""}}
			ri.Parts[0] = ri.Resource
			if rng.Float32() < 0.3 {
				ri.Subresource = "status"
				ri.Parts = append(ri.Parts, "status")
			}
			ri.Namespace = fmt.Sprintf("ms%d", rng.Uint64())
			skippingReqs = append(skippingReqs, ri)
		}
	}
	nMatch := 1 + rng.Intn(3)
	for i := 0; i < nMatch; i++ {
		ri := &request.RequestInfo{
			IsResourceRequest: false,
			Path:              nrr.NonResourceURLs[rng.Intn(len(nrr.NonResourceURLs))],
			Verb:              nrr.Verbs[rng.Intn(len(nrr.Verbs))],
			APIPrefix:         "", APIGroup: "",
			APIVersion: "", Resource: "",
			Subresource: "", Name: "", Parts: []string{""}}
		if nrr.Verbs[0] == fcv1a1.VerbAll {
			ri.Verb = ([]string{"patch", "post", "delete"})[rng.Intn(3)]
		}
		if nrr.NonResourceURLs[0] == "*" {
			ri.Path = ([]string{"/api", "/openapi/v2", "/apis/coordination.k8s.io/v1beta1"})[rng.Intn(3)]
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

func pickString(rng *rand.Rand, strings ...string) string {
	return strings[rng.Intn(len(strings))]
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
