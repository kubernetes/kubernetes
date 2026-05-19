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
	checkedAttrs                    []authorizer.Attributes
	echoCalled                      bool
}

func (c *constrainedImpersonationTest) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	c.checkedAttrs = append(c.checkedAttrs, a)

	u := a.GetUser()

	if u.GetName() == "sa-impersonater" && a.GetVerb() == "impersonate:serviceaccount" && a.GetResource() == "serviceaccounts" {
		return authorizer.DecisionAllow, "", nil
	}

	if u.GetName() == "system:serviceaccount:default:node" && a.GetVerb() == "impersonate:arbitrary-node" && a.GetResource() == "nodes" {
		return authorizer.DecisionAllow, "", nil
	}

	if u.GetName() == "node-impersonater" && a.GetVerb() == "impersonate:arbitrary-node" && a.GetResource() == "nodes" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(u.GetGroups()) > 0 && u.GetGroups()[0] == "associate-node-impersonater" && a.GetVerb() == "impersonate:associated-node" && a.GetResource() == "nodes" {
		return authorizer.DecisionAllow, "", nil
	}

	if u.GetName() == "user-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "users" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(u.GetGroups()) > 0 && u.GetGroups()[0] == "group-impersonater" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "groups" {
		return authorizer.DecisionAllow, "", nil
	}

	if len(u.GetGroups()) > 0 && u.GetGroups()[0] == "extra-setter-scopes" && a.GetVerb() == "impersonate:user-info" && a.GetResource() == "userextras" && a.GetSubresource() == "pandas.io/scopes" {
		return authorizer.DecisionAllow, "", nil
	}

	if u.GetName() == "legacy-impersonater" && a.GetVerb() == "impersonate" {
		return authorizer.DecisionAllow, "", nil
	}

	if u.GetName() != "legacy-impersonator" &&
		strings.HasPrefix(a.GetVerb(), "impersonate-on:") &&
		(strings.HasSuffix(a.GetVerb(), "list") || strings.HasSuffix(a.GetVerb(), "get")) {
		return authorizer.DecisionAllow, "", nil
	}

	return authorizer.DecisionNoOpinion, "deny by default", nil
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
	checkedAttrs := c.checkedAttrs
	c.checkedAttrs = nil

	require.Equal(c.t, len(r.expectedAttributes), len(checkedAttrs))

	// normally all authorization checks are done against the requestor
	// but in some cases such as associated-node, we check against a slightly different user
	expectedAttributesUser := r.expectedAttributesUser
	if expectedAttributesUser == nil {
		expectedAttributesUser = r.requestor
	}

	for i := range len(r.expectedAttributes) {
		expectedAttributes := withUser(r.expectedAttributes[i], expectedAttributesUser)
		require.Equal(c.t, expectedAttributes, comparableAttributes(checkedAttrs[i]))
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

type testRequest struct {
	request          *request.RequestInfo
	requestor        *user.DefaultInfo
	impersonatedUser *user.DefaultInfo

	expectedImpersonatedUser *user.DefaultInfo
	expectedMessage          string

	expectedAttributesUser *user.DefaultInfo // nil means use requestor
	expectedAttributes     []authorizer.AttributesRecord
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
			expectedAttributes: []authorizer.AttributesRecord{
				withImpersonateOnAttributes(getSecretRequest, "associated-node"),
				withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "*"}, "associated-node"),
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
			expectedAttributes:       nil, // no authz checks for the second request
			expectedCache:            cacheWithOnlyNode1Data,
			expectedCode:             http.StatusOK,
		},
		{
			request:                getSecretRequest,
			requestor:              saDefaultOnNode2,
			impersonatedUser:       &user.DefaultInfo{Name: "system:node:node1"}, // node does not match
			expectedMessage:        `nodes.authentication.k8s.io "node1" is forbidden: User "system:serviceaccount:default:default" cannot impersonate:arbitrary-node resource "nodes" in API group "authentication.k8s.io" at the cluster scope: deny by default`,
			expectedAttributesUser: nil,
			expectedAttributes: []authorizer.AttributesRecord{
				withImpersonateOnAttributes(getSecretRequest, "arbitrary-node"),
				withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "node1"}, "arbitrary-node"),
				withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:node:node1"}),
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
			expectedAttributes: []authorizer.AttributesRecord{
				withImpersonateOnAttributes(getPodRequest, "associated-node"), // one authz check because different request info
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
			expectedAttributes:       nil, // no authz checks for pod request via node1
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getPodRequest, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "anyone"}, "user-info"),
						withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "anyone"}),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getPodRequest, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "anyone"}, "user-info"),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getAnotherPodRequest, "user-info"),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(createPodRequest, "user-info"),
						withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "anyone"}),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getPodRequest, "serviceaccount"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "serviceaccounts", Namespace: "default", Name: "default"}, "serviceaccount"),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getPodRequest, "arbitrary-node"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "node1"}, "arbitrary-node"),
						withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:node:node1"}),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(createPodRequest, "arbitrary-node"),
						withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:node:node1"}),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getPodRequest, "arbitrary-node"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "node1"}, "arbitrary-node"),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getPodRequest, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "groups", Name: "extra-setter-scopes"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "scope-a"}, "user-info"),
						withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"}),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getPodRequest, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "*", Name: "*"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "scope-a"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "scope-b"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "5"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "4"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "3"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "2"}, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "userextras", Subresource: "pandas.io/scopes", Name: "1"}, "user-info"),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getPodRequest, "arbitrary-node"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "nodes", Name: "node1"}, "arbitrary-node"),
						withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:node:node1"}),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getPodRequest, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"}, "user-info"),
						withLegacyImpersonateAttributes(authorizer.AttributesRecord{Resource: "users", Name: "system:admin"}),
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
					expectedAttributes: []authorizer.AttributesRecord{
						withImpersonateOnAttributes(getDeploymentRequest, "user-info"),
						withConstrainedImpersonationAttributes(authorizer.AttributesRecord{Resource: "users", Name: "bob"}, "user-info"),
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
					expectedCode: http.StatusOK,
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
