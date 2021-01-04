/*
Copyright 2021 The Kubernetes Authors.

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

package injection

import (
	"context"

	"k8s.io/client-go/rest"
)

// nsKey is the key that namespaces are associated with on
// contexts returned by WithNamespaceScope.
type nsKey struct{}

// WithNamespaceScope associates a namespace scoping with the
// provided context, which will scope the informers produced
// by the downstream informer factories.
func WithNamespaceScope(ctx context.Context, namespace string) context.Context {
	return context.WithValue(ctx, nsKey{}, namespace)
}

// HasNamespaceScope determines whether the provided context has
// been scoped to a particular namespace.
func HasNamespaceScope(ctx context.Context) bool {
	return GetNamespaceScope(ctx) != ""
}

// GetNamespaceScope accesses the namespace associated with the
// provided context.  This should be called when the injection
// logic is setting up shared informer factories.
func GetNamespaceScope(ctx context.Context) string {
	value := ctx.Value(nsKey{})
	if value == nil {
		return ""
	}
	return value.(string)
}

// cfgKey is the key that the config is associated with.
type cfgKey struct{}

// WithConfig associates a given config with the context.
func WithConfig(ctx context.Context, cfg *rest.Config) context.Context {
	return context.WithValue(ctx, cfgKey{}, cfg)
}

// GetConfig gets the current config from the context.
func GetConfig(ctx context.Context) *rest.Config {
	value := ctx.Value(cfgKey{})
	if value == nil {
		return nil
	}
	return value.(*rest.Config)
}
