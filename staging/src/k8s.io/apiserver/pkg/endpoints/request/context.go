/*
Copyright 2014 The Kubernetes Authors.

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

package request

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// The key type is unexported to prevent collisions
type key int

const (
	// namespaceKey is the context key for the request namespace.
	namespaceKey key = iota

	// userKey is the context key for the request user.
	userKey

	// used for propagating a conditional authorization decision between authorization and admission
	conditionallyAuthorizedDecisionKey
)

// NewContext instantiates a base context object for request flows.
func NewContext() context.Context {
	return context.TODO()
}

// NewDefaultContext instantiates a base context object for request flows in the default namespace
func NewDefaultContext() context.Context {
	return WithNamespace(NewContext(), metav1.NamespaceDefault)
}

// WithValue returns a copy of parent in which the value associated with key is val.
func WithValue(parent context.Context, key interface{}, val interface{}) context.Context {
	return context.WithValue(parent, key, val)
}

// WithNamespace returns a copy of parent in which the namespace value is set
func WithNamespace(parent context.Context, namespace string) context.Context {
	return WithValue(parent, namespaceKey, namespace)
}

// NamespaceFrom returns the value of the namespace key on the ctx
func NamespaceFrom(ctx context.Context) (string, bool) {
	namespace, ok := ctx.Value(namespaceKey).(string)
	return namespace, ok
}

// NamespaceValue returns the value of the namespace key on the ctx, or the empty string if none
func NamespaceValue(ctx context.Context) string {
	namespace, _ := NamespaceFrom(ctx)
	return namespace
}

// WithUser returns a copy of parent in which the user value is set
func WithUser(parent context.Context, user user.Info) context.Context {
	return WithValue(parent, userKey, user)
}

// UserFrom returns the value of the user key on the ctx
func UserFrom(ctx context.Context) (user.Info, bool) {
	user, ok := ctx.Value(userKey).(user.Info)
	return user, ok
}

type authorizerDecisionTuple struct {
	authorizer authorizer.Authorizer
	decision   authorizer.Decision
}

// WithConditionallyAllowedDecision returns a copy of parent in which the conditionally allowed authorization decision is set
func WithConditionallyAuthorizedDecision(parent context.Context, authorizer authorizer.Authorizer, decision authorizer.Decision) context.Context {
	return WithValue(parent, conditionallyAuthorizedDecisionKey, authorizerDecisionTuple{
		authorizer: authorizer,
		decision:   decision,
	})
}

// ConditionalAuthorizationContextFrom returns the conditional authorization enforcer associated with the ctx
func ConditionallyAuthorizedDecisionFrom(ctx context.Context) (authorizer.Authorizer, authorizer.Decision, bool) {
	tuple, ok := ctx.Value(conditionallyAuthorizedDecisionKey).(authorizerDecisionTuple)
	return tuple.authorizer, tuple.decision, ok
}
