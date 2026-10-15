/*
Copyright The Kubernetes Authors.

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

// Package coverage lets an API server require that admission hooks which
// intercept some request shapes of a resource also intercept another shape.
//
// When a server adds a new way to modify an object (a new subresource or
// operation), webhooks and admission policies written before it existed don't
// intercept it, so they can be bypassed silently. A Requirement names the new
// shape (the target) and the existing shapes (the sources). A hook that would
// have intercepted the request had it been one of the sources, but doesn't
// intercept the actual target request, is in violation. Webhooks and
// MutatingAdmissionPolicies reject such requests; ValidatingAdmissionPolicy
// bindings apply their validationActions.
//
// The relation is directional: hooks on the target never affect requests for
// the sources. It is version-agnostic: a source is evaluated as the same
// request with a different operation and subresource.
//
// The embedding server injects its requirements via the admission initializer.
package coverage
