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
	"fmt"
	"slices"
	"strings"

	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

type impersonatedUserInfo struct {
	user       user.Info
	constraint string // the verb used in impersonationModeState.check that allowed this user to be impersonated
}

// impersonationMode is a type that represents a specific impersonation mode
// it checks if a requester is allowed to make an API request (the attributes) while impersonating a user (the wantedUser)
// a mode may return a cached result if it supports caching (using the input cache key if appropriate)
// a nil impersonatedUserInfo is returned if the mode does not support impersonating the wantedUser
type impersonationMode interface {
	check(ctx context.Context, key *impersonationCacheKey, wantedUser *user.DefaultInfo, attributes authorizer.Attributes) (*impersonatedUserInfo, error)

	// all methods below are only used in unit tests
	verbForTests() string
	cachesForTests() (outer, inner *impersonationCache)
}

// constrainedImpersonationModeFilter is a function that defines if a specific constrained impersonation mode
// supports the requestor impersonating the wantedUser.  It serves as a sudo authorization check for the mode.
type constrainedImpersonationModeFilter func(wantedUser *user.DefaultInfo, requestor user.Info) bool

func allImpersonationModes(a authorizer.Authorizer) []impersonationMode {
	return []impersonationMode{
		associatedNodeImpersonationMode(a),
		arbitraryNodeImpersonationMode(a),
		serviceAccountImpersonationMode(a),
		userInfoImpersonationMode(a),
		legacyImpersonationMode(a),
	}
}

// associatedNodeImpersonationMode allows a requestor service account to impersonate the node that it is
// associated with.  this is by far the most complex impersonation mode because it caches successful
// impersonation attempts in a way that results in a high cache hit ratio even when the same service account
// is used across different pods running on different nodes (i.e. a node agent running as a daemonset).
// only the username can be specified by the requester.  all other fields in user.Info are controlled by the API server.
func associatedNodeImpersonationMode(a authorizer.Authorizer) impersonationMode {
	// we wrap the authorizer so that we can override the requestor service account's extra values
	// and the node name used in the authorization check.  this makes our authorization checks match
	// the exact semantics of our cache key which prevents unexpected privilege escalation on a cache
	// hit.  see the comment below for the cache key details.
	wrappedAuthorizer := authorizer.AuthorizerFunc(func(ctx context.Context, attributes authorizer.Attributes) (authorizer.Decision, string, error) {
		// we use checkAuthorization instead of directly calling the authorizer so we can
		// make the error message line up with the actual attributes authorized against
		if err := checkAuthorization(ctx, a, &associatedNodeImpersonationAttributes{Attributes: attributes}); err != nil {
			return authorizer.DecisionDeny, "", err
		}
		return authorizer.DecisionAllow, "", nil
	})
	mode := newConstrainedImpersonationMode(wrappedAuthorizer, "associated-node",
		func(wantedUser *user.DefaultInfo, requestor user.Info) bool {
			wantedNodeName := wantedUser.Name
			return onlyUsernameSet(wantedUser) && requesterAssociatedWithRequestedNodeUsername(requestor, wantedNodeName)
		},
	)
	return &associatedNodeImpersonationCheck{mode: mode}
}

type associatedNodeImpersonationCheck struct {
	mode impersonationMode
}

func (a *associatedNodeImpersonationCheck) check(ctx context.Context, _ *impersonationCacheKey, wantedUser *user.DefaultInfo, attributes authorizer.Attributes) (*impersonatedUserInfo, error) {
	wantedNodeName := wantedUser.Name
	// ignore the input cache key because the cache semantics for associated-node require custom logic.
	// we know that by the time this key is used, the filter has already verified that the requestor is
	// a service account with a node ref that matches the node it is trying to impersonate.
	// the actual node being impersonated is not relevant to the cache key, so we just use a static wantedUser.
	// we wrap the attributes so that we can drop the requestor service account's extra values.
	// this results in the cache key being the same for the same service account across all nodes.
	// this is only safe because of the aforementioned filter running before the cache lookup happens.
	key := &impersonationCacheKey{wantedUser: &user.DefaultInfo{Name: "system:node:*"}, attributes: &associatedNodeImpersonationAttributes{Attributes: attributes}}
	impersonatedNodeWithMaybeIncorrectUsername, err := a.mode.check(ctx, key, wantedUser, attributes)
	if err != nil || impersonatedNodeWithMaybeIncorrectUsername == nil {
		return nil, err
	}
	// at this point, we know that we have a successful associated-node impersonation.
	// the value could have come from the cache and thus could be for any node, so we wrap the result
	// here so that the username matches the node associated with the requestor service account.
	return &impersonatedUserInfo{
		user: &associatedNodeImpersonationWantedUserInfo{
			Info: impersonatedNodeWithMaybeIncorrectUsername.user,
			name: wantedNodeName,
		},
		constraint: impersonatedNodeWithMaybeIncorrectUsername.constraint,
	}, nil
}

func (a *associatedNodeImpersonationCheck) verbForTests() string {
	return a.mode.verbForTests()
}

func (a *associatedNodeImpersonationCheck) cachesForTests() (*impersonationCache, *impersonationCache) {
	return a.mode.cachesForTests()
}

type associatedNodeImpersonationAttributes struct {
	authorizer.Attributes
}

func (a *associatedNodeImpersonationAttributes) GetUser() user.Info {
	return &associatedNodeImpersonationRequestorUserInfo{Info: a.Attributes.GetUser()}
}

func (a *associatedNodeImpersonationAttributes) GetName() string {
	if a.GetVerb() == "impersonate:associated-node" {
		return "*" // our cache key ignores the node name, so our authorization check needs to be valid for all node names
	}
	return a.Attributes.GetName()
}

type associatedNodeImpersonationRequestorUserInfo struct {
	user.Info
}

func (a *associatedNodeImpersonationRequestorUserInfo) GetExtra() map[string][]string {
	return map[string][]string{
		// we know the requestor is a service account with a node ref that matches the node it is trying to impersonate
		// basically all the extra values would cause cache misses (for example, the node name itself)
		// so we drop all the extra values but keep the associated key names
		// the authorizer can trust that we have performed the node association check correctly
		// the audit log will still contain the full requestor extra fields
		// different bound object ref types can result in different extra keys, and thus a different cache key
		"authentication.kubernetes.io/associated-node-keys": sets.StringKeySet(a.Info.GetExtra()).List(),
	}
}

type associatedNodeImpersonationWantedUserInfo struct {
	user.Info
	name string
}

func (a *associatedNodeImpersonationWantedUserInfo) GetName() string {
	return a.name
}

// arbitraryNodeImpersonationMode implements constrained impersonation for nodes.
// Only the username can be specified by the requester.  All other fields in user.Info are controlled by the API server.
func arbitraryNodeImpersonationMode(a authorizer.Authorizer) impersonationMode {
	return newConstrainedImpersonationMode(a, "arbitrary-node",
		func(wantedUser *user.DefaultInfo, _ user.Info) bool {
			if !onlyUsernameSet(wantedUser) {
				return false
			}
			_, ok := isNodeUsername(wantedUser.Name)
			return ok
		},
	)
}

// serviceAccountImpersonationMode implements constrained impersonation for service accounts.
// Only the username can be specified by the requester.  All other fields in user.Info are controlled by the API server.
func serviceAccountImpersonationMode(a authorizer.Authorizer) impersonationMode {
	return newConstrainedImpersonationMode(a, "serviceaccount",
		func(wantedUser *user.DefaultInfo, _ user.Info) bool {
			if !onlyUsernameSet(wantedUser) {
				return false
			}
			_, _, ok := isServiceAccountUsername(wantedUser.Name)
			return ok
		},
	)
}

// userInfoImpersonationMode implements constrained impersonation for non-node and non-service account users.
// Unlike the other constrained impersonation modes, it supports impersonating all fields of user.Info.
func userInfoImpersonationMode(a authorizer.Authorizer) impersonationMode {
	return newConstrainedImpersonationMode(a, "user-info",
		func(wantedUser *user.DefaultInfo, _ user.Info) bool {
			// nodes and service accounts cannot be impersonated in this mode.
			// the user-info bucket is reserved for the "other" users, that is,
			// users that do not have an explicit schema defined by Kube.
			if _, ok := isNodeUsername(wantedUser.Name); ok {
				return false
			}
			if _, _, ok := isServiceAccountUsername(wantedUser.Name); ok {
				return false
			}
			return true
		},
	)
}

// legacyImpersonationMode is a complete reimplementation of the original impersonation mode that has
// existed in kube since v1.3.  The behavior is expected to be identical to the original implementation.
func legacyImpersonationMode(a authorizer.Authorizer) impersonationMode {
	return &legacyImpersonationCheck{m: newImpersonationModeState(a, "impersonate", false)}
}

type legacyImpersonationCheck struct {
	m *impersonationModeState
}

func (l *legacyImpersonationCheck) check(ctx context.Context, key *impersonationCacheKey, wantedUser *user.DefaultInfo, attributes authorizer.Attributes) (*impersonatedUserInfo, error) {
	requestor := attributes.GetUser()
	return l.m.check(ctx, key, wantedUser, requestor)
}

func (l *legacyImpersonationCheck) verbForTests() string {
	return l.m.verb
}

func (l *legacyImpersonationCheck) cachesForTests() (*impersonationCache, *impersonationCache) {
	// legacy impersonation has no outer layer so just return an empty cache
	// though an inner cache is present, it is unused
	return newImpersonationCache(false), l.m.cache
}

func newConstrainedImpersonationMode(a authorizer.Authorizer, mode string, filter constrainedImpersonationModeFilter) impersonationMode {
	return &constrainedImpersonationModeState{
		state:      newImpersonationModeState(a, "impersonate:"+mode, true),
		cache:      newImpersonationCache(false),
		authorizer: a,
		mode:       mode,
		filter:     filter,
	}
}

// constrainedImpersonationModeState implements the secondary authorization check via impersonate-on:<mode>:<verb> to
// determine if the requestor is authorized to perform the specific verb when impersonating the wantedUser via mode.
// if this check succeeds, the primary authorization checks are run, see impersonationModeState for details.
// if the mode's filter does not match the inputs, the impersonation automatically fails and returns a nil impersonatedUserInfo.
type constrainedImpersonationModeState struct {
	state *impersonationModeState
	// this outer cache covers the overall impersonation for this mode, i.e. a cache hit here short-circuits all checks
	// skipAttributes is false, i.e. this cache depends on the request being made, not just the user being impersonated by the requestor
	// it is expected to have a low hit ratio because the requestor is unlikely to make the same request multiple times in a short period
	cache      *impersonationCache
	authorizer authorizer.Authorizer
	mode       string
	filter     constrainedImpersonationModeFilter
}

func (c *constrainedImpersonationModeState) check(ctx context.Context, key *impersonationCacheKey, wantedUser *user.DefaultInfo, attributes authorizer.Attributes) (*impersonatedUserInfo, error) {
	requestor := attributes.GetUser()
	// we must call the filter before doing anything because this serves as a sudo authorization check to say "does this mode even apply?"
	// also the cache key is not always a direct match with wantedUser+attributes, so again, we must call the filter first
	if !c.filter(wantedUser, requestor) {
		return nil, nil
	}

	if impersonatedUser := c.cache.get(key); impersonatedUser != nil {
		return impersonatedUser, nil
	}

	if err := checkAuthorization(ctx, c.authorizer, &impersonateOnAttributes{mode: c.mode, Attributes: attributes}); err != nil {
		return nil, err
	}

	impersonatedUser, err := c.state.check(ctx, key, wantedUser, requestor)
	if err != nil || impersonatedUser == nil {
		return nil, err
	}
	c.cache.set(key, impersonatedUser)
	return impersonatedUser, nil
}

func (c *constrainedImpersonationModeState) verbForTests() string {
	return c.state.verb
}

func (c *constrainedImpersonationModeState) cachesForTests() (*impersonationCache, *impersonationCache) {
	return c.cache, c.state.cache
}

// impersonationModeState implements the primary authorization checks via the impersonate:<mode> verb for constrained
// impersonation and the impersonate verb for legacy impersonation.  each field that is set in the wantedUser
// results in one or more authorization checks to determine if the requestor has access to impersonate that value.
type impersonationModeState struct {
	authorizer                 authorizer.Authorizer
	verb                       string
	isConstrainedImpersonation bool

	usernameAndGroupGV schema.GroupVersion
	constraint         string

	// this inner cache covers the checks related to the specific fields set in wantedUser
	// skipAttributes is true, i.e. this cache only depends on the user being impersonated by the requestor
	// it is expected to have a high hit ratio because the requestor may impersonate the same user for many different requests
	cache *impersonationCache
}

func newImpersonationModeState(a authorizer.Authorizer, verb string, isConstrainedImpersonation bool) *impersonationModeState {
	usernameAndGroupGV := authenticationv1.SchemeGroupVersion
	constraint := verb
	if !isConstrainedImpersonation {
		usernameAndGroupGV = corev1.SchemeGroupVersion
		constraint = ""
	}
	return &impersonationModeState{
		authorizer:                 a,
		verb:                       verb,
		isConstrainedImpersonation: isConstrainedImpersonation,

		usernameAndGroupGV: usernameAndGroupGV,
		constraint:         constraint,
		cache:              newImpersonationCache(true),
	}
}

func (m *impersonationModeState) check(ctx context.Context, key *impersonationCacheKey, wantedUser *user.DefaultInfo, requestor user.Info) (*impersonatedUserInfo, error) {
	// we only use caching in constrained impersonation mode to avoid any behavioral changes with legacy impersonation
	if m.isConstrainedImpersonation {
		if impersonatedUser := m.cache.get(key); impersonatedUser != nil {
			return impersonatedUser, nil
		}
	}

	actualUser := *wantedUser

	if err := m.authorizeUsername(ctx, requestor, wantedUser.Name, wantedUser.Groups, &actualUser); err != nil {
		return nil, err
	}

	if err := m.authorizeUID(ctx, requestor, wantedUser.UID); err != nil {
		return nil, err
	}

	if err := m.authorizeGroups(ctx, requestor, wantedUser.Groups); err != nil {
		return nil, err
	}

	if err := m.authorizeExtra(ctx, requestor, wantedUser.Extra); err != nil {
		return nil, err
	}

	if actualUser.Name == user.Anonymous {
		ensureGroup(&actualUser, user.AllUnauthenticated)
	} else if !slices.Contains(actualUser.Groups, user.AllUnauthenticated) {
		ensureGroup(&actualUser, user.AllAuthenticated)
	}

	impersonatedUser := &impersonatedUserInfo{user: &actualUser, constraint: m.constraint}
	if m.isConstrainedImpersonation {
		m.cache.set(key, impersonatedUser)
	}
	return impersonatedUser, nil
}

func (m *impersonationModeState) authorizeUsername(ctx context.Context, requestor user.Info, username string, wantedUserGroups []string, actualUser *user.DefaultInfo) error {
	usernameAttributes := impersonationAttributes(requestor, m.usernameAndGroupGV, m.verb, "users", username)

	if m.isConstrainedImpersonation {
		if name, ok := isNodeUsername(username); ok {
			usernameAttributes.Resource = "nodes"
			usernameAttributes.Name = name

			// this should be impossible to reach but check just in case
			if len(wantedUserGroups) != 0 {
				return responsewriters.ForbiddenStatusError(usernameAttributes, fmt.Sprintf("when impersonating a node, cannot impersonate groups %q", wantedUserGroups))
			}

			actualUser.Groups = []string{user.NodesGroup} // all nodes have a fixed group list in constrained impersonation
		}
	}

	if namespace, name, ok := isServiceAccountUsername(username); ok {
		usernameAttributes.Resource = "serviceaccounts"
		usernameAttributes.Namespace = namespace
		usernameAttributes.Name = name

		// this should be impossible to reach but check just in case
		if m.isConstrainedImpersonation && len(wantedUserGroups) != 0 {
			return responsewriters.ForbiddenStatusError(usernameAttributes, fmt.Sprintf("when impersonating a service account, cannot impersonate groups %q", wantedUserGroups))
		}

		if len(wantedUserGroups) == 0 {
			// if groups are not specified for a service account, we know the groups because it is a fixed mapping.  Add them
			actualUser.Groups = serviceaccount.MakeGroupNames(namespace)
		}
	}

	return checkAuthorization(ctx, m.authorizer, usernameAttributes)
}

func (m *impersonationModeState) authorizeUID(ctx context.Context, requestor user.Info, uid string) error {
	if len(uid) == 0 {
		return nil
	}
	uidAttributes := impersonationAttributes(requestor, authenticationv1.SchemeGroupVersion, m.verb, "uids", uid)
	return checkAuthorization(ctx, m.authorizer, uidAttributes)
}

// manyAuthorizationChecksInLoop is an arbitrary value used in constrained impersonation modes to decide if they
// should try to perform a single wildcard authorization check before making many individual checks in a loop.
const manyAuthorizationChecksInLoop = 4

func (m *impersonationModeState) authorizeGroups(ctx context.Context, requestor user.Info, groups []string) error {
	if len(groups) == 0 {
		return nil
	}

	groupAttributes := impersonationAttributes(requestor, m.usernameAndGroupGV, m.verb, "groups", "")

	// perform extra sanity checks that would be backwards incompatible with legacy impersonation
	if m.isConstrainedImpersonation {
		if slices.Contains(groups, "") {
			return responsewriters.ForbiddenStatusError(groupAttributes, "impersonating the empty string group is not allowed")
		}
		if slices.Contains(groups, user.SystemPrivilegedGroup) {
			groupAttributes.Name = user.SystemPrivilegedGroup
			return responsewriters.ForbiddenStatusError(groupAttributes, "impersonating the system:masters group is not allowed")
		}
	}

	// if the requestor is trying to impersonate many groups at once, see if they are authorized to impersonate all groups
	// we only do this in constrained impersonation mode to avoid any behavioral changes with legacy impersonation
	if m.isConstrainedImpersonation && len(groups) >= manyAuthorizationChecksInLoop {
		groupAttributes.Name = "*"
		if err := checkAuthorization(ctx, m.authorizer, groupAttributes); err == nil {
			return nil
		}
	}

	for _, group := range groups {
		groupAttributes.Name = group
		if err := checkAuthorization(ctx, m.authorizer, groupAttributes); err != nil {
			return err
		}
	}

	return nil
}

func (m *impersonationModeState) authorizeExtra(ctx context.Context, requestor user.Info, extra map[string][]string) error {
	if len(extra) == 0 {
		return nil
	}

	extraAttributes := impersonationAttributes(requestor, authenticationv1.SchemeGroupVersion, m.verb, "userextras", "")

	// perform extra sanity checks that would be backwards incompatible with legacy impersonation
	if m.isConstrainedImpersonation {
		if err := validateExtra(extra); err != nil {
			return responsewriters.ForbiddenStatusError(extraAttributes, err.Error())
		}
	}

	// if the requestor is trying to impersonate many extras at once, see if they are authorized to impersonate all extras
	// we only do this in constrained impersonation mode to avoid any behavioral changes with legacy impersonation
	if m.isConstrainedImpersonation && isLargeExtra(extra) {
		extraAttributes.Subresource = "*"
		extraAttributes.Name = "*"
		if err := checkAuthorization(ctx, m.authorizer, extraAttributes); err == nil {
			return nil
		}
	}

	for key, values := range extra {
		extraAttributes.Subresource = key
		for _, value := range values {
			extraAttributes.Name = value
			if err := checkAuthorization(ctx, m.authorizer, extraAttributes); err != nil {
				return err
			}
		}
	}

	return nil
}

func validateExtra(extra map[string][]string) error {
	fp := field.NewPath("extra", "key")
	for key, values := range extra {
		if len(key) == 0 {
			return fmt.Errorf("impersonating the empty string key in extra is not allowed")
		}
		if err := utilvalidation.IsDomainPrefixedPath(fp, key).ToAggregate(); err != nil {
			return fmt.Errorf("impersonating an invalid key in extra is not allowed: %w", err)
		}
		if key != strings.ToLower(key) {
			return fmt.Errorf("impersonating a non-lowercase key in extra is not allowed: %q", key)
		}
		if len(values) == 0 {
			return fmt.Errorf("impersonating empty values in extra is not allowed")
		}
		if slices.Contains(values, "") {
			return fmt.Errorf("impersonating the empty string value in extra is not allowed")
		}
	}
	return nil
}

func isLargeExtra(extra map[string][]string) bool {
	if len(extra) >= manyAuthorizationChecksInLoop {
		return true
	}
	var count int
	for _, values := range extra {
		count += len(values)
		if count >= manyAuthorizationChecksInLoop {
			return true
		}
	}
	return false
}

func impersonationAttributes(requestor user.Info, gv schema.GroupVersion, verb, resource, name string) authorizer.AttributesRecord {
	return authorizer.AttributesRecord{
		User:            requestor,
		Verb:            verb,
		APIGroup:        gv.Group,
		APIVersion:      gv.Version,
		Resource:        resource,
		Name:            name,
		ResourceRequest: true,
	}
}

// impersonateOnAttributes is a simple wrapper that updates the verb of the attributes to impersonate-on:<mode>:<verb>
// This allows the expression of "a subject can perform this verb while using this impersonation mode"
type impersonateOnAttributes struct {
	mode string
	authorizer.Attributes
}

func (i *impersonateOnAttributes) GetVerb() string {
	return "impersonate-on:" + i.mode + ":" + i.Attributes.GetVerb()
}

func checkAuthorization(ctx context.Context, a authorizer.Authorizer, attributes authorizer.Attributes) error {
	authorized, reason, err := a.Authorize(ctx, attributes)

	// an authorizer like RBAC could encounter evaluation errors and still allow the request, so authorizer decision is checked before error here.
	if authorized == authorizer.DecisionAllow {
		return nil
	}

	// if the authorizer gave us a forbidden error, do not wrap it again
	if errors.IsForbidden(err) {
		return err
	}

	msg := reason
	switch {
	case err != nil && len(reason) > 0:
		msg = fmt.Sprintf("%v: %s", err, reason)
	case err != nil:
		msg = err.Error()
	}

	return responsewriters.ForbiddenStatusError(attributes, msg)
}

func ensureGroup(u *user.DefaultInfo, group string) {
	if slices.Contains(u.Groups, group) {
		return
	}

	// do not mutate a slice that we did not create
	groups := make([]string, 0, len(u.Groups)+1)
	groups = append(groups, u.Groups...)
	groups = append(groups, group)
	u.Groups = groups
}

func isServiceAccountUsername(username string) (namespace, name string, ok bool) {
	namespace, name, err := serviceaccount.SplitUsername(username)
	return namespace, name, err == nil
}

// matches the real ValidateNodeName from k8s.io/kubernetes/pkg/apis/core/validation
// which we are not allowed to import here
var validateNodeName = validation.NameIsDNSSubdomain

func isNodeUsername(username string) (string, bool) {
	const nodeUsernamePrefix = "system:node:"
	if !strings.HasPrefix(username, nodeUsernamePrefix) {
		return "", false
	}
	name := strings.TrimPrefix(username, nodeUsernamePrefix)
	if len(validateNodeName(name, false)) != 0 {
		return "", false
	}
	return name, true
}

func requesterAssociatedWithRequestedNodeUsername(requestor user.Info, username string) bool {
	nodeName, ok := isNodeUsername(username)
	if !ok {
		return false
	}
	if _, _, ok := isServiceAccountUsername(requestor.GetName()); !ok {
		return false
	}
	return getExtraValue(requestor, serviceaccount.NodeNameKey) == nodeName
}

func getExtraValue(u user.Info, key string) string {
	values := u.GetExtra()[key]
	if len(values) != 1 {
		return ""
	}
	return values[0]
}

func onlyUsernameSet(u user.Info) bool {
	return len(u.GetUID()) == 0 && len(u.GetGroups()) == 0 && len(u.GetExtra()) == 0
}
