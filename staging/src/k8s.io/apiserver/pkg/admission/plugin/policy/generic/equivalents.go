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

package generic

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/equivalents"
)

// CheckAdmissionEquivalents returns a Forbidden error listing every (policy, binding) pair in hooks
// that applies to an admission equivalent of the request but not to the request itself, or nil.
// Policy dispatchers must call it before evaluating any policy. The generic policy dispatcher
// (used by MutatingAdmissionPolicy) calls it; the ValidatingAdmissionPolicy dispatcher calls it
// directly.
//
// Targets are not filtered by the operations the plugin handles. MutatingAdmissionPolicy never
// runs on DELETE, so a policy that applies to a DELETE target but not to the request is reported
// even though it could not have run on that target either; the fix (adding the request's
// resource to the policy's rules) is harmless.
func CheckAdmissionEquivalents[P, B runtime.Object, E Evaluator](
	ctx context.Context,
	pluginName string,
	a admission.Attributes,
	o admission.ObjectInterfaces,
	matcher PolicyMatcher,
	hooks []PolicyHook[P, B, E],
	newPolicyAccessor func(P) PolicyAccessor,
	newBindingAccessor func(B) BindingAccessor,
) error {
	targets := equivalents.Targets(a, o)
	if targets == nil {
		return nil
	}
	var violations []equivalents.Violation
	for _, hook := range hooks {
		policy := newPolicyAccessor(hook.Policy)
		for _, b := range hook.Bindings {
			binding := newBindingAccessor(b)
			h := equivalents.Hook{Plugin: pluginName, Name: policy.GetName(), Binding: binding.GetName()}
			v, ok := equivalents.Check(a, targets, h, func(attr admission.Attributes) (bool, error) {
				// The pair applies only if both the definition and the binding do, so a clean
				// non-match from either is a clean non-match, even if the other errors. Check
				// relies on this to tell rules gaps apart from evaluation errors.
				defMatches, _, _, defErr := matcher.DefinitionMatches(attr, o, policy)
				if !defMatches && defErr == nil {
					return false, nil
				}
				bindMatches, bindErr := matcher.BindingMatches(attr, o, binding)
				if !bindMatches && bindErr == nil {
					return false, nil
				}
				if defErr != nil {
					return false, defErr
				}
				return bindMatches, bindErr
			})
			if ok {
				violations = append(violations, v)
			}
		}
	}
	if len(violations) > 0 {
		return equivalents.Reject(ctx, a, violations)
	}
	return nil
}
