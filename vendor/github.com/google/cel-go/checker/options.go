// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package checker

type options struct {
	crossTypeNumericComparisons  bool
	homogeneousAggregateLiterals bool
	validatedDeclarations        *Scopes
}

// Option is a functional option for configuring the type-checker
type Option func(*options) error

// CrossTypeNumericComparisons toggles type-checker support for numeric comparisons across type
// See https://github.com/google/cel-spec/wiki/proposal-210 for more details.
func CrossTypeNumericComparisons(enabled bool) Option {
	return func(opts *options) error {
		opts.crossTypeNumericComparisons = enabled
		return nil
	}
}

// ValidatedDeclarations provides a references to validated declarations which will be copied
// into new checker instances.
func ValidatedDeclarations(env *Env) Option {
	return func(opts *options) error {
		opts.validatedDeclarations = env.validatedDeclarations()
		return nil
	}
}
