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
	stderrs "errors"
	"time"

	"golang.org/x/net/context"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/authentication/user"
)

// Context carries values across API boundaries.
// This context matches the context.Context interface
// (https://blog.golang.org/context), for the purposes
// of passing the api.Context through to the storage tier.
// TODO: Determine the extent that this abstraction+interface
// is used by the api, and whether we can remove.
type Context interface {
	// Value returns the value associated with key or nil if none.
	Value(key interface{}) interface{}

	// Deadline returns the time when this Context will be canceled, if any.
	Deadline() (deadline time.Time, ok bool)

	// Done returns a channel that is closed when this Context is canceled
	// or times out.
	Done() <-chan struct{}

	// Err indicates why this context was canceled, after the Done channel
	// is closed.
	Err() error
}

// The key type is unexported to prevent collisions
type key int

const (
	// namespaceKey is the context key for the request namespace.
	namespaceKey key = iota

	// userKey is the context key for the request user.
	userKey

	// uidKey is the context key for the uid to assign to an object on create.
	uidKey

	// userAgentKey is the context key for the request user agent.
	userAgentKey

	namespaceDefault = "default" // TODO(sttts): solve import cycle when using metav1.NamespaceDefault
)

// NewContext instantiates a base context object for request flows.
func NewContext() Context {
	return context.TODO()
}

// NewDefaultContext instantiates a base context object for request flows in the default namespace
func NewDefaultContext() Context {
	return WithNamespace(NewContext(), namespaceDefault)
}

// WithValue returns a copy of parent in which the value associated with key is val.
func WithValue(parent Context, key interface{}, val interface{}) Context {
	internalCtx, ok := parent.(context.Context)
	if !ok {
		panic(stderrs.New("Invalid context type"))
	}
	return context.WithValue(internalCtx, key, val)
}

// WithNamespace returns a copy of parent in which the namespace value is set
func WithNamespace(parent Context, namespace string) Context {
	return WithValue(parent, namespaceKey, namespace)
}

// NamespaceFrom returns the value of the namespace key on the ctx
func NamespaceFrom(ctx Context) (string, bool) {
	namespace, ok := ctx.Value(namespaceKey).(string)
	return namespace, ok
}

// NamespaceValue returns the value of the namespace key on the ctx, or the empty string if none
func NamespaceValue(ctx Context) string {
	namespace, _ := NamespaceFrom(ctx)
	return namespace
}

// WithNamespaceDefaultIfNone returns a context whose namespace is the default if and only if the parent context has no namespace value
func WithNamespaceDefaultIfNone(parent Context) Context {
	namespace, ok := NamespaceFrom(parent)
	if !ok || len(namespace) == 0 {
		return WithNamespace(parent, namespaceDefault)
	}
	return parent
}

// WithUser returns a copy of parent in which the user value is set
func WithUser(parent Context, user user.Info) Context {
	return WithValue(parent, userKey, user)
}

// UserFrom returns the value of the user key on the ctx
func UserFrom(ctx Context) (user.Info, bool) {
	user, ok := ctx.Value(userKey).(user.Info)
	return user, ok
}

// WithUID returns a copy of parent in which the uid value is set
func WithUID(parent Context, uid types.UID) Context {
	return WithValue(parent, uidKey, uid)
}

// UIDFrom returns the value of the uid key on the ctx
func UIDFrom(ctx Context) (types.UID, bool) {
	uid, ok := ctx.Value(uidKey).(types.UID)
	return uid, ok
}

// WithUserAgent returns a copy of parent in which the user value is set
func WithUserAgent(parent Context, userAgent string) Context {
	return WithValue(parent, userAgentKey, userAgent)
}

// UserAgentFrom returns the value of the userAgent key on the ctx
func UserAgentFrom(ctx Context) (string, bool) {
	userAgent, ok := ctx.Value(userAgentKey).(string)
	return userAgent, ok
}
