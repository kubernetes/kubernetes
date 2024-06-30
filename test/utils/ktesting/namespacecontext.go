/*
Copyright 2024 The Kubernetes Authors.

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

package ktesting

// WithNamespace add a default namespace name for operations using the context.
func WithNamespace(tCtx TContext, namespace string) TContext {
	return namespaceContext{TContext: tCtx, namespace: namespace}
}

type namespaceContext struct {
	TContext

	namespace string
}

func (nCtx namespaceContext) Namespace() string {
	return nCtx.namespace
}
