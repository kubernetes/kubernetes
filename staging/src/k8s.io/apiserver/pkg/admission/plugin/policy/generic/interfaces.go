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

package generic

import (
	"context"

	"k8s.io/apiserver/pkg/admission"
)

// Hook represents a dynamic admission hook. The hook may be a webhook or a
// policy. For webhook, the Hook may describe how to contact the endpoint, expected
// cert, etc. For policies, the hook may describe a compiled policy-binding pair.
type Hook interface {
	// All hooks are expected to contain zero or more match conditions, object
	// selectors, namespace selectors to help the dispatcher decide when to apply
	// the hook.
	//
	// Methods of matching logic is applied are specific to the hook and left up
	// to the implementation.
}

// Source can list dynamic admission plugins.
type Source[H Hook] interface {
	// Hooks returns the list of currently known admission hooks.
	Hooks() []H

	// Run the source. This method should be called only once at startup.
	Run(ctx context.Context) error

	// HasSynced returns true if the source has completed its initial sync.
	HasSynced() bool
}

// Dispatcher dispatches evaluates an admission request against the currently
// active hooks returned by the source.
type Dispatcher[H Hook] interface {
	// Run the dispatcher. This method should be called only once at startup.
	Run(ctx context.Context) error

	// Dispatch a request to the policies. Dispatcher may choose not to
	// call a hook, either because the rules of the hook does not match, or
	// the namespaceSelector or the objectSelector of the hook does not
	// match. A non-nil error means the request is rejected.
	Dispatch(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces, hooks []H) error
}

// An evaluator represents a compiled CEL expression that can be evaluated a
// given a set of inputs used by the generic PolicyHook for Mutating and
// ValidatingAdmissionPolicy.
// Mutating and Validating may have different forms of evaluators
type Evaluator interface {
}
