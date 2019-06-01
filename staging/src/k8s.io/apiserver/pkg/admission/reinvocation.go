/*
Copyright 2019 The Kubernetes Authors.

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

package admission

// newReinvocationHandler creates a handler that wraps the provided admission chain and reinvokes it
// if needed according to re-invocation policy of the webhooks.
func newReinvocationHandler(admissionChain Interface) Interface {
	return &reinvoker{admissionChain}
}

type reinvoker struct {
	admissionChain Interface
}

// Admit performs an admission control check using the wrapped admission chain, reinvoking the
// admission chain if needed according to the reinvocation policy.  Plugins are expected to check
// the admission attributes' reinvocation context against their reinvocation policy to decide if
// they should re-run, and to update the reinvocation context if they perform any mutations.
func (r *reinvoker) Admit(a Attributes, o ObjectInterfaces) error {
	if mutator, ok := r.admissionChain.(MutationInterface); ok {
		err := mutator.Admit(a, o)
		if err != nil {
			return err
		}
		s := a.GetReinvocationContext()
		if s.ShouldReinvoke() {
			s.SetIsReinvoke()
			// Calling admit a second time will reinvoke all in-tree plugins
			// as well as any webhook plugins that need to be reinvoked based on the
			// reinvocation policy.
			return mutator.Admit(a, o)
		}
	}
	return nil
}

// Validate performs an admission control check using the wrapped admission chain, and returns immediately on first error.
func (r *reinvoker) Validate(a Attributes, o ObjectInterfaces) error {
	if validator, ok := r.admissionChain.(ValidationInterface); ok {
		return validator.Validate(a, o)
	}
	return nil
}

// Handles will return true if any of the admission chain handlers handle the given operation.
func (r *reinvoker) Handles(operation Operation) bool {
	return r.admissionChain.Handles(operation)
}
