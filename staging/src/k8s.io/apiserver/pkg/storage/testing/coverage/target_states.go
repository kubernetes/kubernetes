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

package coverage

// TargetStates returns the catalog of API states (~80, per #141652 step 1)
// that coverage.Wrap-based test suites are expected to exercise.
//
// This is a curated list, not a full cartesian product of every dimension:
// crossing all of List's dimensions alone would yield well over a thousand
// combinations, most of them redundant for coverage purposes (e.g. selector
// shape rarely interacts with pagination shape in ways worth distinguishing).
// Instead, each verb gets a baseline grid over its most important dimensions
// (typically ResourceVersion x ResourceVersionMatch, held at Outcome=Success),
// plus a handful of rows that vary one additional dimension - pagination,
// selectors, preconditions, terminal error outcome - at a fixed baseline
// shape. Add or remove rows here as review surfaces states worth tracking
// separately, or states that turn out to be redundant. Each verb's rows live
// in their own target_<verb>_states.go file.
func TargetStates() []State {
	var states []State
	states = append(states, getTargetStates()...)
	states = append(states, listTargetStates()...)
	states = append(states, watchTargetStates()...)
	states = append(states, deleteTargetStates()...)
	return states
}
