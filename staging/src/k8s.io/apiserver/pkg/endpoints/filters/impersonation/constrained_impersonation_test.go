/*
Copyright 2025 The Kubernetes Authors.

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

package impersonation

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/transport"
)

type constrainedImpersonationTest struct {
	t *testing.T

	constrainedImpersonationHandler *constrainedImpersonationHandler
	authzChecks                     []authzCheck
	echoCalled                      bool
}

type authzCheck struct {
	attributes authorizer.Attributes
	decision   authorizer.Decision
	reason     string
	err        error
}

func (c *constrainedImpersonationTest) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	u := a.GetUser()

	var decision authorizer.Decision
	var reason string
	var err error

	defer func() {
		c.authzChecks = append(c.authzChecks, authzCheck{
			attributes: a,
			decision:   decision,
			reason:     reason,
			err:        err,
		})
	}()

	if u.GetName() == "sa-impersonater" && a.GetVerb() == "impersonate:serviceaccount" && a.GetResource() == "serviceaccounts" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	if u.GetName() == "system:serviceaccount:default:node" && a.GetVerb() == "impersonate:arbitrary-node" && a.GetResource() == "nodes" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	if u.GetName() == "node-impersonater" && a.GetVerb() == "impersonate:arbitrary-node" && a.GetResource() == "nodes" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	if len(u.GetGroups()) > 0 && u.GetGroups()[0] == "associate-node-impersonater" && a.GetVerb() == "impersonate:associated-node" && a.GetResource() == "nodes" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	if u.GetName() == "user-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "users" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	if len(u.GetGroups()) > 0 && u.GetGroups()[0] == "group-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "groups" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	if len(u.GetGroups()) > 0 && u.GetGroups()[0] == "extra-setter-scopes" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "userextras" && a.GetSubresource() == "pandas.io/scopes" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	if u.GetName() == "legacy-impersonater" && a.GetVerb() == "impersonate" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	if u.GetName() != "legacy-impersonator" &&
		strings.HasPrefix(a.GetVerb(), "impersonate-on:") &&
		(strings.HasSuffix(a.GetVerb(), "list") || strings.HasSuffix(a.GetVerb(), "get")) {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	// many-groups-impersonater: can impersonate users and any groups via wildcard
	if u.GetName() == "many-groups-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "users" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}
	if u.GetName() == "many-groups-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "groups" && a.GetName() == "*" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	// many-extras-impersonater: can impersonate users and any extras via wildcard
	if u.GetName() == "many-extras-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "users" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}
	if u.GetName() == "many-extras-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "userextras" && a.GetSubresource() == "*" && a.GetName() == "*" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	// mixed-impersonater: can impersonate users, specific groups, and specific extras (but NOT via wildcards)
	if u.GetName() == "mixed-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "users" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}
	// Allow specific group names, but NOT wildcard
	if u.GetName() == "mixed-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "groups" && a.GetName() != "*" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}
	// Allow specific extras for tags.io/env, but NOT wildcard
	if u.GetName() == "mixed-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "userextras" &&
		a.GetSubresource() == "tags.io/env" && a.GetName() != "*" {
		decision, reason, err = authorizer.DecisionAllow, "", nil
		return decision, reason, err
	}

	decision, reason, err = authorizer.DecisionNoOpinion, "deny by default", nil
	return decision, reason, err
}

func (c *constrainedImpersonationTest) echoUserInfoHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		c.echoCalled = true

		u, ok := request.UserFrom(req.Context())
		if !ok {
			c.t.Fatal("user not found in request")
		}

		_ = json.NewEncoder(w).Encode(comparableUser(u))

		if _, ok := req.Header[authenticationv1.ImpersonateUserHeader]; ok {
			c.t.Fatal("user header still present")
		}
		if _, ok := req.Header[authenticationv1.ImpersonateUIDHeader]; ok {
			c.t.Fatal("uid header still present")
		}
		if _, ok := req.Header[authenticationv1.ImpersonateGroupHeader]; ok {
			c.t.Fatal("group header still present")
		}
		for key := range req.Header {
			if strings.HasPrefix(key, authenticationv1.ImpersonateUserExtraHeaderPrefix) {
				c.t.Fatalf("extra header still present: %v", key)
			}
		}
	}
}

func (c *constrainedImpersonationTest) authenticationHandler(handler http.Handler) http.Handler {
	return filters.WithAuthentication(handler, authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
		userData := req.Header.Get(testUserHeader)
		if len(userData) == 0 {
			c.t.Fatal("missing user header")
		}
		var u user.DefaultInfo
		if err := json.Unmarshal([]byte(userData), &u); err != nil {
			c.t.Fatal(err)
		}
		return &authenticator.Response{User: &u}, true, nil
	}), nil, nil, nil)
}

func (c *constrainedImpersonationTest) requestInfoHandler(handler http.Handler) http.Handler {
	return filters.WithRequestInfo(handler, requestInfoFunc(func(req *http.Request) (*request.RequestInfo, error) {
		requestInfoData := req.Header.Get(testRequestInfoHeader)
		if len(requestInfoData) == 0 {
			c.t.Fatal("missing request info header")
		}
		var r request.RequestInfo
		if err := json.Unmarshal([]byte(requestInfoData), &r); err != nil {
			c.t.Fatal(err)
		}
		return &r, nil
	}))
}

type requestInfoFunc func(*http.Request) (*request.RequestInfo, error)

func (f requestInfoFunc) NewRequestInfo(req *http.Request) (*request.RequestInfo, error) {
	return f(req)
}

const (
	testUserHeader        = "insecure-test-user-json"
	testRequestInfoHeader = "insecure-test-request-info-json"
)

func (c *constrainedImpersonationTest) handler() http.Handler {
	s := runtime.NewScheme()
	metav1.AddToGroupVersion(s, metav1.SchemeGroupVersion)
	addImpersonation := WithConstrainedImpersonation(c.echoUserInfoHandler(), c, serializer.NewCodecFactory(s))
	c.constrainedImpersonationHandler = addImpersonation.(*constrainedImpersonationHandler)

	addAuthentication := c.authenticationHandler(addImpersonation)
	addRequestInfo := c.requestInfoHandler(addAuthentication)
	return addRequestInfo
}

type testRoundTripper struct {
	user        *user.DefaultInfo
	requestInfo *request.RequestInfo
	delegate    http.RoundTripper
}

func (t *testRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	userData, err := json.Marshal(t.user)
	if err != nil {
		return nil, err
	}
	requestInfoData, err := json.Marshal(t.requestInfo)
	if err != nil {
		return nil, err
	}
	r.Header.Set(testUserHeader, string(userData))
	r.Header.Set(testRequestInfoHeader, string(requestInfoData))
	return t.delegate.RoundTrip(r)
}

func (c *constrainedImpersonationTest) assertAttributes(r testRequest) {
	authzChecks := c.authzChecks
	c.authzChecks = nil

	require.Len(c.t, authzChecks, len(r.expectedAuthzChecks))

	// normally all authorization checks are done against the requestor
	// but in some cases such as associated-node, we check against a slightly different user
	expectedAttributesUser := r.expectedAttributesUser
	if expectedAttributesUser == nil {
		expectedAttributesUser = r.requestor
	}

	for i := range len(r.expectedAuthzChecks) {
		expectedAttributes := withUser(r.expectedAuthzChecks[i].attributes, expectedAttributesUser)
		require.Equal(c.t, expectedAttributes, comparableAttributes(authzChecks[i].attributes), "authz check %d: attributes mismatch", i)
		require.Equal(c.t, r.expectedAuthzChecks[i].decision, authzChecks[i].decision, "authz check %d: decision mismatch", i)
	}
}

func comparableAttributes(attributes authorizer.Attributes) authorizer.AttributesRecord {
	fs, errFS := attributes.GetFieldSelector()
	ls, errLS := attributes.GetLabelSelector()
	return authorizer.AttributesRecord{
		User:                      comparableUser(attributes.GetUser()),
		Verb:                      attributes.GetVerb(),
		Namespace:                 attributes.GetNamespace(),
		APIGroup:                  attributes.GetAPIGroup(),
		APIVersion:                attributes.GetAPIVersion(),
		Resource:                  attributes.GetResource(),
		Subresource:               attributes.GetSubresource(),
		Name:                      attributes.GetName(),
		ResourceRequest:           attributes.IsResourceRequest(),
		Path:                      attributes.GetPath(),
		FieldSelectorRequirements: fs,
		FieldSelectorParsingErr:   errFS,
		LabelSelectorRequirements: ls,
		LabelSelectorParsingErr:   errLS,
	}
}

func comparableUser(u user.Info) *user.DefaultInfo {
	return &user.DefaultInfo{
		Name:   u.GetName(),
		UID:    u.GetUID(),
		Groups: u.GetGroups(),
		Extra:  u.GetExtra(),
	}
}

func (c *constrainedImpersonationTest) assertEchoCalled(expectedCalled bool) {
	called := c.echoCalled
	c.echoCalled = false
	require.Equal(c.t, expectedCalled, called)
}

func (c *constrainedImpersonationTest) assertCache(r testRequest) {
	rr := require.New(c.t)

	tracker := c.constrainedImpersonationHandler.tracker
	idxCacheInternals := tracker.idxCache.cache

	if r.expectedCache == nil {
		rr.Zero(idxCacheInternals.Len())
		for _, mode := range tracker.modes {
			outer, inner := mode.cachesForTests()
			rr.Zero(outer.cache.Len())
			rr.Zero(inner.cache.Len())
		}
		return
	}

	rr.Equal(len(r.expectedCache.modeIdx), idxCacheInternals.Len())
	for username, expectedVerb := range r.expectedCache.modeIdx {
		modeIdx, ok := idxCacheInternals.Get(usernameHash(username))
		rr.True(ok)
		actualVerb := tracker.modes[modeIdx.(int)].verbForTests()
		rr.Equal(expectedVerb, actualVerb)
	}

	for _, mode := range tracker.modes {
		verb := mode.verbForTests()
		outer, inner := mode.cachesForTests()
		expectedMode, ok := r.expectedCache.modes[verb]
		if !ok {
			rr.Zero(outer.cache.Len())
			rr.Zero(inner.cache.Len())
			continue
		}
		rr.Equal(len(expectedMode.outer), outer.cache.Len())
		for expectedKey, expectedUser := range expectedMode.outer {
			c.checkCacheEntry(outer, expectedKey.wantedUser, expectedKey.attributes, expectedUser, verb)
		}

		rr.Equal(len(expectedMode.inner), inner.cache.Len())
		for expectedKey, expectedUser := range expectedMode.inner {
			c.checkCacheEntry(inner, expectedKey.wantedUser, authorizer.AttributesRecord{User: expectedKey.requestor}, expectedUser, verb)
		}
	}
}

func usernameHash(username string) string {
	hash := fnvSum128a([]byte(username))
	return fmt.Sprintf("%x", hash)
}

func (c *constrainedImpersonationTest) checkCacheEntry(cache *impersonationCache, wantedUser *user.DefaultInfo, attributes authorizer.Attributes, expectedUser *user.DefaultInfo, expectedVerb string) {
	rr := require.New(c.t)
	keyString, err := buildKey(wantedUser, attributes)
	rr.NoError(err)
	val, ok := cache.cache.Get(keyString)
	rr.True(ok)
	impersonatedUser := val.(*impersonatedUserInfo)
	rr.Equal(expectedVerb, impersonatedUser.constraint)
	rr.Equal(expectedUser, impersonatedUser.user)
	rr.True(strings.HasSuffix(keyString, "/"+attributes.GetUser().GetName()))
}

func withConstrainedImpersonationAttributes(a authorizer.AttributesRecord, mode string) authorizer.AttributesRecord {
	a.Verb = "impersonate:" + mode
	a.APIGroup = "authentication.k8s.io"
	a.APIVersion = "v1"
	a.ResourceRequest = true
	return a
}

func withImpersonateOnAttributes(requestInfo *request.RequestInfo, mode string) authorizer.AttributesRecord {
	a := requestInfoToAttributes(requestInfo)
	a.Verb = "impersonate-on:" + mode + ":" + a.Verb
	return a
}

func requestInfoToAttributes(requestInfo *request.RequestInfo) authorizer.AttributesRecord {
	requestCtx := request.WithRequestInfo(context.Background(), requestInfo)
	attrs, err := filters.GetAuthorizerAttributes(requestCtx)
	if err != nil {
		panic(err)
	}
	return *(attrs.(*authorizer.AttributesRecord))
}

func withLegacyImpersonateAttributes(a authorizer.AttributesRecord) authorizer.AttributesRecord {
	a.Verb = "impersonate"
	a.APIVersion = "v1"
	a.ResourceRequest = true
	return a
}

func withUser(a authorizer.AttributesRecord, u *user.DefaultInfo) authorizer.AttributesRecord {
	a.User = u
	return a
}

func expectAuthzCheck(attributes authorizer.AttributesRecord, decision authorizer.Decision) expectedAuthzCheck {
	return expectedAuthzCheck{
		attributes: attributes,
		decision:   decision,
	}
}

func expectAllow(attributes authorizer.AttributesRecord) expectedAuthzCheck {
	return expectAuthzCheck(attributes, authorizer.DecisionAllow)
}

func expectDeny(attributes authorizer.AttributesRecord) expectedAuthzCheck {
	return expectAuthzCheck(attributes, authorizer.DecisionNoOpinion)
}

func outerCacheKey(wantedUser, requestor *user.DefaultInfo, req *request.RequestInfo) impersonationCacheKey {
	attributes := withUser(requestInfoToAttributes(req), requestor)
	return impersonationCacheKey{
		wantedUser: wantedUser,
		attributes: &attributes, // use a *authorizer.AttributesRecord so that it can be used in a map key
	}
}

type expectedCache struct {
	modeIdx map[string]string            // username -> mode verb
	modes   map[string]expectedModeCache // mode verb -> cache
}

type expectedModeCache struct {
	outer map[impersonationCacheKey]*user.DefaultInfo
	inner map[innerKey]*user.DefaultInfo
}

type innerKey struct {
	wantedUser, requestor *user.DefaultInfo
}

type expectedAuthzCheck struct {
	attributes authorizer.AttributesRecord
	decision   authorizer.Decision
}

type testRequest struct {
	request          *request.RequestInfo
	requestor        *user.DefaultInfo
	impersonatedUser *user.DefaultInfo

	expectedImpersonatedUser *user.DefaultInfo
	expectedMessage          string

	expectedAttributesUser *user.DefaultInfo // nil means use requestor
	expectedAuthzChecks    []expectedAuthzCheck
	expectedCache          *expectedCache
	expectedCode           int
}

func associatedNodeTestCase() []testRequest {
	getSecretRequest := &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "get",
		APIVersion:        "v1",
		Resource:          "secrets",
		Name:              "foo",
		Namespace:         "bar",
	}
	getPodRequest := &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "get",
		APIVersion:        "v1",
		Resource:          "pods",
		Name:              "foo",
		Namespace:         "bar",
	}
	saDefaultOnNode1 := &user.DefaultInfo{
		Name:   "system:serviceaccount:default:default",
		Groups: []string{"associate-node-impersonater"},
		Extra: map[string][]string{
			serviceaccount.NodeNameKey: {"node1"},
		},
	}
	saDefaultOnNode2 := &user.DefaultInfo{
		Name:   "system:serviceaccount:default:default",
		Groups: []string{"associate-node-impersonater"},
		Extra: map[string][]string{
			serviceaccount.NodeNameKey: {"node2"},
		},
	}
	saDefaultOnAnyNode := &user.DefaultInfo{
		Name:   "system:serviceaccount:default:default",
		Groups: []string{"associate-node-impersonater"},
		Extra: map[string][]string{
			"authentication.kubernetes.io/associated-node-keys": {"authentication.kubernetes.io/node-name"},
		},
	}
	node1FullUserInfo := &user.DefaultInfo{
		Name:   "system:node:node1",
		Groups: []string{user.NodesGroup, "system:authenticated"},
	}
	node2FullUserInfo := &user.DefaultInfo{
		Name:   "system:node:node2",
		Groups: []string{user.NodesGroup, "system:authenticated"},
	}
	cacheWithOnlyNode1Data := &expectedCache{
		modeIdx: map[string]string{
			"system:serviceaccount:default:default": "impersonate:associated-node",
		},
		modes: map[string]expectedModeCache{
			"impersonate:associated-node": {
				outer: map[impersonationCacheKey]*user.DefaultInfo{
					outerCacheKey(&user.DefaultInfo{Name: "system:node:*"}, saDefaultOnAnyNode, getSecretRequest): node1FullUserInfo,
				},
				inner: map[innerKey]*user.DefaultInfo{
					{
						wantedUser: &user.DefaultInfo{Name: "system:node:*"},
						requestor:  saDefaultOnAnyNode,
					}: node1FullUserInfo,
				},
			},
		},
	}
	cacheWithMultipleRequests := &expectedCache{
		modeIdx: cacheWithOnlyNode1Data.modeIdx,
		modes: map[string]expectedModeCache{
			"impersonate:associated-node": {
				outer: map[impersonationCacheKey]*user.DefaultInfo{
					outerCacheKey(&user.DefaultInfo{Name: "system:node:*"}, saDefaultOnAnyNode, getSecretRequest): node1FullUserInfo,
					// even though this request was made on node2, since the inner cache matched it returned a value for node1
					outerCacheKey(&user.DefaultInfo{Name: "system:node:*"}, saDefaultOnAnyNode, getPodRequest): node1FullUserInfo,
				},
				inner: cacheWithOnlyNode1Data.modes["impersonate:associated-node"].inner,
			},
		},
	}
	return []testRequest{
		{
			request:                  getSecretRequest,
			requestor:                saDefaultOnNode1,
			impersonatedUser:         &user.DefaultInfo{Name: "system:node:node1"}, // node matches
			expectedImpersonatedUser: node1FullUserInfo,
			expectedAttributesUser:   saDefaultOnAnyNode,
			expectedAuthzChecks: []expectedAuthzCheck{
				expectAllow(withImpersonateOnAttributes(getSecretRequest, "associated-node")),
				expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "*"}, "associated-node")),
			},
			expectedCache: cacheWithOnlyNode1Data,
			expectedCode:  http.StatusOK,
		},
		{
			request:                  getSecretRequest,
			requestor:                saDefaultOnNode2,
			impersonatedUser:         &user.DefaultInfo{Name: "system:node:node2"}, // node matches
			expectedImpersonatedUser: node2FullUserInfo,
			expectedAttributesUser:   saDefaultOnAnyNode,
			expectedAuthzChecks:      nil, // no authz checks for the second request
			expectedCache:            cacheWithOnlyNode1Data,
			expectedCode:             http.StatusOK,
		},
		{
			request:                getSecretRequest,
			requestor:              saDefaultOnNode2,
			impersonatedUser:       &user.DefaultInfo{Name: "system:node:node1"}, // node does not match
			expectedMessage:        `nodes.authentication.k8s.io "node1" is forbidden: User "system:serviceaccount:default:default" cannot impersonate:arbitrary-node resource "nodes" in API group "authentication.k8s.io" at the cluster scope: deny by default`,
			expectedAttributesUser: nil,
			expectedAuthzChecks: []expectedAuthzCheck{
				expectAllow(withImpersonateOnAttributes(getSecretRequest, "arbitrary-node")),
				expectDeny(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "node1"}, "arbitrary-node")),
				expectDeny(withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:node:node1"})),
			},
			expectedCache: cacheWithOnlyNode1Data,
			expectedCode:  http.StatusForbidden,
		},
		{
			request:                  getPodRequest,
			requestor:                saDefaultOnNode2,
			impersonatedUser:         &user.DefaultInfo{Name: "system:node:node2"}, // node matches
			expectedImpersonatedUser: node2FullUserInfo,
			expectedAttributesUser:   saDefaultOnAnyNode,
			expectedAuthzChecks: []expectedAuthzCheck{
				expectAllow(withImpersonateOnAttributes(getPodRequest, "associated-node")), // one authz check because different request info
			},
			expectedCache: cacheWithMultipleRequests,
			expectedCode:  http.StatusOK,
		},
		{
			request:                  getPodRequest,
			requestor:                saDefaultOnNode1,
			impersonatedUser:         &user.DefaultInfo{Name: "system:node:node1"}, // node matches
			expectedImpersonatedUser: node1FullUserInfo,
			expectedAttributesUser:   nil,
			expectedAuthzChecks:      nil, // no authz checks for pod request via node1
			expectedCache:            cacheWithMultipleRequests,
			expectedCode:             http.StatusOK,
		},
	}
}

func TestConstrainedImpersonationFilter(t *testing.T) {
	getPodRequest := &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "get",
		APIVersion:        "v1",
		Resource:          "pods",
		Name:              "foo",
		Namespace:         "bar",
	}

	getAnotherPodRequest := &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "get",
		APIVersion:        "v1",
		Resource:          "pods",
		Name:              "foo1",
		Namespace:         "bar1",
	}

	createPodRequest := &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "create",
		APIVersion:        "v1",
		Resource:          "pods",
		Name:              "foo",
		Namespace:         "bar",
	}

	getDeploymentRequest := &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "get",
		APIVersion:        "v1",
		APIGroup:          "apps",
		Resource:          "deployments",
		Name:              "foo",
		Namespace:         "bar",
	}

	anyone := &user.DefaultInfo{Name: "anyone"}
	anyoneAuthenticated := &user.DefaultInfo{Name: "anyone", Groups: []string{"system:authenticated"}}
	saUser := &user.DefaultInfo{Name: "system:serviceaccount:default:default"}
	saUserAuthenticated := &user.DefaultInfo{
		Name:   "system:serviceaccount:default:default",
		Groups: []string{"system:serviceaccounts", "system:serviceaccounts:default", "system:authenticated"}}
	nodeUser := &user.DefaultInfo{Name: "system:node:node1"}
	nodeUserAuthenticated := &user.DefaultInfo{
		Name:   "system:node:node1",
		Groups: []string{user.NodesGroup, "system:authenticated"},
	}

	userImpersonator := &user.DefaultInfo{Name: "user-impersonater"}
	saImpersonator := &user.DefaultInfo{Name: "sa-impersonater"}
	nodeImpersonator := &user.DefaultInfo{Name: "node-impersonater"}

	testCases := []struct {
		name     string
		requests []testRequest
	}{
		{
			name: "impersonating-error",
			requests: []testRequest{
				{
					request:          getPodRequest,
					requestor:        &user.DefaultInfo{Name: "tester"},
					impersonatedUser: anyone,
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "user-info")),
						expectDeny(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "anyone"}, "user-info")),
						expectDeny(withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "anyone"})),
					},
					expectedCache:   nil,
					expectedCode:    http.StatusForbidden,
					expectedMessage: `users.authentication.k8s.io "anyone" is forbidden: User "tester" cannot impersonate:user-info resource "users" in API group "authentication.k8s.io" at the cluster scope: deny by default`,
				},
			},
		},
		{
			name: "impersonating-user-get-allowed-create-disallowed",
			requests: []testRequest{
				{
					request:                  getPodRequest,
					requestor:                userImpersonator,
					impersonatedUser:         anyone,
					expectedImpersonatedUser: anyoneAuthenticated,
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "anyone"}, "user-info")),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							userImpersonator.Name: "impersonate:user-info",
						},
						modes: map[string]expectedModeCache{
							"impersonate:user-info": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(anyone, userImpersonator, getPodRequest): anyoneAuthenticated,
								},
								inner: map[innerKey]*user.DefaultInfo{
									{wantedUser: anyone, requestor: userImpersonator}: anyoneAuthenticated,
								},
							},
						},
					},
					expectedCode: http.StatusOK,
				},
				{
					request:                  getAnotherPodRequest,
					requestor:                userImpersonator,
					impersonatedUser:         anyone,
					expectedImpersonatedUser: anyoneAuthenticated,
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getAnotherPodRequest, "user-info")),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							userImpersonator.Name: "impersonate:user-info",
						},
						modes: map[string]expectedModeCache{
							"impersonate:user-info": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(anyone, userImpersonator, getPodRequest):        anyoneAuthenticated,
									outerCacheKey(anyone, userImpersonator, getAnotherPodRequest): anyoneAuthenticated,
								},
								inner: map[innerKey]*user.DefaultInfo{
									{wantedUser: anyone, requestor: userImpersonator}: anyoneAuthenticated,
								},
							},
						},
					},
					expectedCode: http.StatusOK,
				},
				{
					request:          createPodRequest,
					requestor:        userImpersonator,
					impersonatedUser: anyone,
					expectedAuthzChecks: []expectedAuthzCheck{
						expectDeny(withImpersonateOnAttributes(createPodRequest, "user-info")),
						expectDeny(withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "anyone"})),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							userImpersonator.Name: "impersonate:user-info",
						},
						modes: map[string]expectedModeCache{
							"impersonate:user-info": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(anyone, userImpersonator, getPodRequest):        anyoneAuthenticated,
									outerCacheKey(anyone, userImpersonator, getAnotherPodRequest): anyoneAuthenticated,
								},
								inner: map[innerKey]*user.DefaultInfo{
									{wantedUser: anyone, requestor: userImpersonator}: anyoneAuthenticated,
								},
							},
						},
					},
					expectedCode:    http.StatusForbidden,
					expectedMessage: `pods "foo" is forbidden: User "user-impersonater" cannot impersonate-on:user-info:create resource "pods" in API group "" in the namespace "bar": deny by default`,
				},
			},
		},
		{
			name: "impersonating-sa-allowed",
			requests: []testRequest{
				{
					request:                  getPodRequest,
					requestor:                saImpersonator,
					impersonatedUser:         saUser,
					expectedImpersonatedUser: saUserAuthenticated,
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "serviceaccount")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "serviceaccounts", Namespace: "default", Name: "default"}, "serviceaccount")),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							saImpersonator.Name: "impersonate:serviceaccount",
						},
						modes: map[string]expectedModeCache{
							"impersonate:serviceaccount": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(saUser, saImpersonator, getPodRequest): saUserAuthenticated,
								},
								inner: map[innerKey]*user.DefaultInfo{
									{
										wantedUser: saUser,
										requestor:  saImpersonator,
									}: saUserAuthenticated,
								},
							},
						},
					},
					expectedCode: http.StatusOK,
				},
			},
		},
		{
			name: "impersonating-node-not-allowed",
			requests: []testRequest{
				{
					request:          getPodRequest,
					requestor:        saImpersonator,
					impersonatedUser: nodeUser,
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "arbitrary-node")),
						expectDeny(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "node1"}, "arbitrary-node")),
						expectDeny(withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:node:node1"})),
					},
					expectedCache:   nil,
					expectedCode:    http.StatusForbidden,
					expectedMessage: `nodes.authentication.k8s.io "node1" is forbidden: User "sa-impersonater" cannot impersonate:arbitrary-node resource "nodes" in API group "authentication.k8s.io" at the cluster scope: deny by default`,
				},
			},
		},
		{
			name: "impersonating-node-not-allowed-action",
			requests: []testRequest{
				{
					request:          createPodRequest,
					requestor:        nodeImpersonator,
					impersonatedUser: nodeUser,
					expectedAuthzChecks: []expectedAuthzCheck{
						expectDeny(withImpersonateOnAttributes(createPodRequest, "arbitrary-node")),
						expectDeny(withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:node:node1"})),
					},
					expectedCache:   nil,
					expectedCode:    http.StatusForbidden,
					expectedMessage: `pods "foo" is forbidden: User "node-impersonater" cannot impersonate-on:arbitrary-node:create resource "pods" in API group "" in the namespace "bar": deny by default`,
				},
			},
		},
		{
			name: "impersonating-node-allowed",
			requests: []testRequest{
				{
					request:                  getPodRequest,
					requestor:                nodeImpersonator,
					impersonatedUser:         nodeUser,
					expectedImpersonatedUser: nodeUserAuthenticated,
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "arbitrary-node")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "node1"}, "arbitrary-node")),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							nodeImpersonator.Name: "impersonate:arbitrary-node",
						},
						modes: map[string]expectedModeCache{
							"impersonate:arbitrary-node": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(nodeUser, nodeImpersonator, getPodRequest): nodeUserAuthenticated,
								},
								inner: map[innerKey]*user.DefaultInfo{
									{wantedUser: nodeUser, requestor: nodeImpersonator}: nodeUserAuthenticated,
								},
							},
						},
					},
					expectedCode: http.StatusOK,
				},
			},
		},
		{
			name: "disallowed-userextra-3",
			requests: []testRequest{
				{
					request: getPodRequest,
					requestor: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"group-impersonater"},
					},
					impersonatedUser: &user.DefaultInfo{
						Name:   "system:admin",
						Groups: []string{"extra-setter-scopes"},
						Extra:  map[string][]string{"pandas.io/scopes": {"scope-a", "scope-b"}},
					},
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "groups", Name: "extra-setter-scopes"}, "user-info")),
						expectDeny(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "scope-a"}, "user-info")),
						expectDeny(withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"})),
					},
					expectedCache:   nil,
					expectedCode:    http.StatusForbidden,
					expectedMessage: `userextras.authentication.k8s.io "scope-a" is forbidden: User "user-impersonater" cannot impersonate:user-info resource "userextras/pandas.io/scopes" in API group "authentication.k8s.io" at the cluster scope: deny by default`,
				},
			},
		},
		{
			name: "allowed-userextras",
			requests: []testRequest{
				{
					request: getPodRequest,
					requestor: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"extra-setter-scopes"},
					},
					impersonatedUser: &user.DefaultInfo{
						Name:  "system:admin",
						Extra: map[string][]string{"pandas.io/scopes": {"scope-a", "scope-b", "5", "4", "3", "2", "1"}},
					},
					expectedImpersonatedUser: &user.DefaultInfo{
						Name:   "system:admin",
						Groups: []string{"system:authenticated"},
						Extra:  map[string][]string{"pandas.io/scopes": {"scope-a", "scope-b", "5", "4", "3", "2", "1"}},
					},
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"}, "user-info")),
						expectDeny(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "*", Name: "*"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "scope-a"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "scope-b"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "5"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "4"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "3"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "2"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "1"}, "user-info")),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							userImpersonator.Name: "impersonate:user-info",
						},
						modes: map[string]expectedModeCache{
							"impersonate:user-info": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(
										&user.DefaultInfo{
											Name:  "system:admin",
											Extra: map[string][]string{"pandas.io/scopes": {"scope-a", "scope-b", "5", "4", "3", "2", "1"}},
										},
										&user.DefaultInfo{
											Name:   "user-impersonater",
											Groups: []string{"extra-setter-scopes"},
										},
										getPodRequest): {
										Name:   "system:admin",
										Groups: []string{"system:authenticated"},
										Extra:  map[string][]string{"pandas.io/scopes": {"scope-a", "scope-b", "5", "4", "3", "2", "1"}},
									},
								},
								inner: map[innerKey]*user.DefaultInfo{
									{
										wantedUser: &user.DefaultInfo{
											Name:  "system:admin",
											Extra: map[string][]string{"pandas.io/scopes": {"scope-a", "scope-b", "5", "4", "3", "2", "1"}},
										},
										requestor: &user.DefaultInfo{
											Name:   "user-impersonater",
											Groups: []string{"extra-setter-scopes"},
										},
									}: {
										Name:   "system:admin",
										Groups: []string{"system:authenticated"},
										Extra:  map[string][]string{"pandas.io/scopes": {"scope-a", "scope-b", "5", "4", "3", "2", "1"}},
									},
								},
							},
						},
					},
					expectedCode: http.StatusOK,
				},
			},
		},
		{
			name:     "allowed-associate-node",
			requests: associatedNodeTestCase(),
		},
		{
			name: "disallowed-associate-node-without-sa",
			requests: []testRequest{
				{
					request: getPodRequest,
					requestor: &user.DefaultInfo{
						Name:   "user-impersonater",
						Groups: []string{"associate-node-impersonater"},
						Extra: map[string][]string{
							serviceaccount.NodeNameKey: {"node1"},
						},
					},
					impersonatedUser: &user.DefaultInfo{Name: "system:node:node1"},
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "arbitrary-node")),
						expectDeny(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "node1"}, "arbitrary-node")),
						expectDeny(withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:node:node1"})),
					},
					expectedCache:   nil,
					expectedCode:    http.StatusForbidden,
					expectedMessage: `nodes.authentication.k8s.io "node1" is forbidden: User "user-impersonater" cannot impersonate:arbitrary-node resource "nodes" in API group "authentication.k8s.io" at the cluster scope: deny by default`,
				},
			},
		},
		{
			name: "allowed-legacy-impersonator",
			requests: []testRequest{
				{
					request:          getPodRequest,
					requestor:        &user.DefaultInfo{Name: "legacy-impersonater"},
					impersonatedUser: &user.DefaultInfo{Name: "system:admin"},
					expectedImpersonatedUser: &user.DefaultInfo{
						Name:   "system:admin",
						Groups: []string{"system:authenticated"},
					},
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "user-info")),
						expectDeny(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"}, "user-info")),
						expectAllow(withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"})),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							"legacy-impersonater": "impersonate",
						},
						modes: nil, // legacy impersonation does not cache
					},
					expectedCode: http.StatusOK,
				},
			},
		},
		{
			name: "system:masters-group-not-allowed",
			requests: []testRequest{
				{
					request:   getPodRequest,
					requestor: userImpersonator,
					impersonatedUser: &user.DefaultInfo{
						Name:   "admin-user",
						Groups: []string{user.SystemPrivilegedGroup},
					},
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "admin-user"}, "user-info")),
						expectDeny(withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "admin-user"})),
					},
					expectedCache:   nil,
					expectedCode:    http.StatusForbidden,
					expectedMessage: `groups.authentication.k8s.io "system:masters" is forbidden: User "user-impersonater" cannot impersonate:user-info resource "groups" in API group "authentication.k8s.io" at the cluster scope: impersonating the system:masters group is not allowed`,
				},
			},
		},
		{
			name: "many-groups-wildcard-allowed",
			requests: []testRequest{
				{
					request: getPodRequest,
					requestor: &user.DefaultInfo{
						Name: "many-groups-impersonater",
					},
					impersonatedUser: &user.DefaultInfo{
						Name:   "bob",
						Groups: []string{"group1", "group2", "group3", "group4"},
					},
					expectedImpersonatedUser: &user.DefaultInfo{
						Name:   "bob",
						Groups: []string{"group1", "group2", "group3", "group4", "system:authenticated"},
					},
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "bob"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "groups", Name: "*"}, "user-info")),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							"many-groups-impersonater": "impersonate:user-info",
						},
						modes: map[string]expectedModeCache{
							"impersonate:user-info": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(&user.DefaultInfo{
										Name:   "bob",
										Groups: []string{"group1", "group2", "group3", "group4"},
									}, &user.DefaultInfo{Name: "many-groups-impersonater"}, getPodRequest): {
										Name:   "bob",
										Groups: []string{"group1", "group2", "group3", "group4", "system:authenticated"},
									},
								},
								inner: map[innerKey]*user.DefaultInfo{
									{
										wantedUser: &user.DefaultInfo{
											Name:   "bob",
											Groups: []string{"group1", "group2", "group3", "group4"},
										},
										requestor: &user.DefaultInfo{Name: "many-groups-impersonater"},
									}: {
										Name:   "bob",
										Groups: []string{"group1", "group2", "group3", "group4", "system:authenticated"},
									},
								},
							},
						},
					},
					expectedCode: http.StatusOK,
				},
			},
		},
		{
			name: "many-extras-wildcard-allowed",
			requests: []testRequest{
				{
					request: getPodRequest,
					requestor: &user.DefaultInfo{
						Name: "many-extras-impersonater",
					},
					impersonatedUser: &user.DefaultInfo{
						Name: "alice",
						Extra: map[string][]string{
							"scopes.example.com/key1": {"val1"},
							"scopes.example.com/key2": {"val2"},
							"scopes.example.com/key3": {"val3"},
							"scopes.example.com/key4": {"val4"},
						},
					},
					expectedImpersonatedUser: &user.DefaultInfo{
						Name:   "alice",
						Groups: []string{"system:authenticated"},
						Extra: map[string][]string{
							"scopes.example.com/key1": {"val1"},
							"scopes.example.com/key2": {"val2"},
							"scopes.example.com/key3": {"val3"},
							"scopes.example.com/key4": {"val4"},
						},
					},
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getPodRequest, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "alice"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "*", Name: "*"}, "user-info")),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							"many-extras-impersonater": "impersonate:user-info",
						},
						modes: map[string]expectedModeCache{
							"impersonate:user-info": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(&user.DefaultInfo{
										Name: "alice",
										Extra: map[string][]string{
											"scopes.example.com/key1": {"val1"},
											"scopes.example.com/key2": {"val2"},
											"scopes.example.com/key3": {"val3"},
											"scopes.example.com/key4": {"val4"},
										},
									}, &user.DefaultInfo{Name: "many-extras-impersonater"}, getPodRequest): {
										Name:   "alice",
										Groups: []string{"system:authenticated"},
										Extra: map[string][]string{
											"scopes.example.com/key1": {"val1"},
											"scopes.example.com/key2": {"val2"},
											"scopes.example.com/key3": {"val3"},
											"scopes.example.com/key4": {"val4"},
										},
									},
								},
								inner: map[innerKey]*user.DefaultInfo{
									{
										wantedUser: &user.DefaultInfo{
											Name: "alice",
											Extra: map[string][]string{
												"scopes.example.com/key1": {"val1"},
												"scopes.example.com/key2": {"val2"},
												"scopes.example.com/key3": {"val3"},
												"scopes.example.com/key4": {"val4"},
											},
										},
										requestor: &user.DefaultInfo{Name: "many-extras-impersonater"},
									}: {
										Name:   "alice",
										Groups: []string{"system:authenticated"},
										Extra: map[string][]string{
											"scopes.example.com/key1": {"val1"},
											"scopes.example.com/key2": {"val2"},
											"scopes.example.com/key3": {"val3"},
											"scopes.example.com/key4": {"val4"},
										},
									},
								},
							},
						},
					},
					expectedCode: http.StatusOK,
				},
			},
		},
		{
			name: "mixed-groups-extras-wildcard-denied-individuals-allowed",
			requests: []testRequest{
				{
					request: getPodRequest,
					requestor: &user.DefaultInfo{
						Name: "mixed-impersonater",
					},
					impersonatedUser: &user.DefaultInfo{
						Name:   "frank",
						Groups: []string{"dev", "ops", "security", "audit"},
						Extra: map[string][]string{
							"tags.io/env": {"prod", "staging", "dev"},
						},
					},
					expectedImpersonatedUser: &user.DefaultInfo{
						Name:   "frank",
						Groups: []string{"dev", "ops", "security", "audit", "system:authenticated"},
						Extra: map[string][]string{
							"tags.io/env": {"prod", "staging", "dev"},
						},
					},
					expectedAuthzChecks: []expectedAuthzCheck{
						// impersonate-on check passes
						expectAllow(withImpersonateOnAttributes(getPodRequest, "user-info")),
						// user check passes
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "frank"}, "user-info")),
						// wildcard groups check FAILS (mixed-impersonater doesn't allow wildcard groups)
						expectDeny(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "groups", Name: "*"}, "user-info")),
						// individual group checks PASS - demonstrating fallback after wildcard failure
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "groups", Name: "dev"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "groups", Name: "ops"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "groups", Name: "security"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "groups", Name: "audit"}, "user-info")),
						// individual extra checks PASS (wildcard not attempted for extras with single subresource)
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "tags.io/env", Name: "prod"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "tags.io/env", Name: "staging"}, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "tags.io/env", Name: "dev"}, "user-info")),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							"mixed-impersonater": "impersonate:user-info",
						},
						modes: map[string]expectedModeCache{
							"impersonate:user-info": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(&user.DefaultInfo{
										Name:   "frank",
										Groups: []string{"dev", "ops", "security", "audit"},
										Extra: map[string][]string{
											"tags.io/env": {"prod", "staging", "dev"},
										},
									}, &user.DefaultInfo{Name: "mixed-impersonater"}, getPodRequest): {
										Name:   "frank",
										Groups: []string{"dev", "ops", "security", "audit", "system:authenticated"},
										Extra: map[string][]string{
											"tags.io/env": {"prod", "staging", "dev"},
										},
									},
								},
								inner: map[innerKey]*user.DefaultInfo{
									{
										wantedUser: &user.DefaultInfo{
											Name:   "frank",
											Groups: []string{"dev", "ops", "security", "audit"},
											Extra: map[string][]string{
												"tags.io/env": {"prod", "staging", "dev"},
											},
										},
										requestor: &user.DefaultInfo{Name: "mixed-impersonater"},
									}: {
										Name:   "frank",
										Groups: []string{"dev", "ops", "security", "audit", "system:authenticated"},
										Extra: map[string][]string{
											"tags.io/env": {"prod", "staging", "dev"},
										},
									},
								},
							},
						},
					},
					expectedCode: http.StatusOK,
				},
			},
		},
		{
			name: "continuous-same-allowed-user-requests",
			requests: []testRequest{
				{
					request:          getDeploymentRequest,
					requestor:        userImpersonator,
					impersonatedUser: &user.DefaultInfo{Name: "bob"},
					expectedImpersonatedUser: &user.DefaultInfo{
						Name:   "bob",
						Groups: []string{"system:authenticated"},
					},
					expectedAuthzChecks: []expectedAuthzCheck{
						expectAllow(withImpersonateOnAttributes(getDeploymentRequest, "user-info")),
						expectAllow(withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "bob"}, "user-info")),
					},
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							userImpersonator.Name: "impersonate:user-info",
						},
						modes: map[string]expectedModeCache{
							"impersonate:user-info": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(&user.DefaultInfo{Name: "bob"}, userImpersonator, getDeploymentRequest): {
										Name:   "bob",
										Groups: []string{"system:authenticated"},
									},
								},
								inner: map[innerKey]*user.DefaultInfo{
									{wantedUser: &user.DefaultInfo{Name: "bob"}, requestor: userImpersonator}: {
										Name:   "bob",
										Groups: []string{"system:authenticated"},
									},
								},
							},
						},
					},
					expectedCode: http.StatusOK,
				},
				{
					request:          getDeploymentRequest,
					requestor:        userImpersonator,
					impersonatedUser: &user.DefaultInfo{Name: "bob"},
					expectedImpersonatedUser: &user.DefaultInfo{
						Name:   "bob",
						Groups: []string{"system:authenticated"},
					},
					expectedAuthzChecks: nil, // no authz checks for cached request
					expectedCode:        http.StatusOK,
					expectedCache: &expectedCache{
						modeIdx: map[string]string{
							userImpersonator.Name: "impersonate:user-info",
						},
						modes: map[string]expectedModeCache{
							"impersonate:user-info": {
								outer: map[impersonationCacheKey]*user.DefaultInfo{
									outerCacheKey(&user.DefaultInfo{Name: "bob"}, userImpersonator, getDeploymentRequest): {
										Name:   "bob",
										Groups: []string{"system:authenticated"},
									},
								},
								inner: map[innerKey]*user.DefaultInfo{
									{wantedUser: &user.DefaultInfo{Name: "bob"}, requestor: userImpersonator}: {
										Name:   "bob",
										Groups: []string{"system:authenticated"},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	var mux http.ServeMux
	tests := make([]*constrainedImpersonationTest, len(testCases))
	handlers := make([]http.Handler, len(testCases))
	for i := range len(testCases) {
		mux.Handle("/"+strconv.Itoa(i), http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			handlers[i].ServeHTTP(w, req)
		}))
	}

	server := httptest.NewServer(&mux)
	t.Cleanup(server.Close)

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			test := &constrainedImpersonationTest{t: t}
			tests[i] = test
			handlers[i] = test.handler()

			for _, r := range tc.requests {
				client := &http.Client{
					Transport: &testRoundTripper{
						user:        r.requestor,
						requestInfo: r.request,
						delegate: transport.NewImpersonatingRoundTripper(
							transport.ImpersonationConfig{
								UserName: r.impersonatedUser.Name,
								UID:      r.impersonatedUser.UID,
								Groups:   r.impersonatedUser.Groups,
								Extra:    r.impersonatedUser.Extra,
							},
							http.DefaultTransport,
						),
					},
				}

				req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL+"/"+strconv.Itoa(i), nil)
				require.NoError(t, err)

				resp, err := client.Do(req)
				require.NoError(t, err)
				require.Equal(t, r.expectedCode, resp.StatusCode)

				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err)
				_ = resp.Body.Close()

				if r.expectedCode == http.StatusOK {
					var actualUser user.DefaultInfo
					if err := json.Unmarshal(body, &actualUser); err != nil {
						t.Errorf("unexpected error: %v, body=\n%s", err, string(body))
					}

					require.Equal(t, r.expectedImpersonatedUser, &actualUser)
					test.assertEchoCalled(true)
					require.NotNil(t, r.expectedImpersonatedUser) // sanity check test data
					require.Empty(t, r.expectedMessage)           // sanity check test data
				} else {
					var status metav1.Status
					if err := json.Unmarshal(body, &status); err != nil {
						t.Errorf("unexpected error: %v, body=\n%s", err, string(body))
					}
					require.Equal(t, r.expectedMessage, status.Message)
					test.assertEchoCalled(false)
					require.NotEmpty(t, r.expectedMessage)     // sanity check test data
					require.Nil(t, r.expectedImpersonatedUser) // sanity check test data
				}

				test.assertAttributes(r)
				test.assertCache(r)
			}
		})
	}
}
