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

import (
	"cel.dev/cel-go/common/env"
)

// CatalogSupplier returns a Catalog of symbols.
type CatalogSupplier func() *env.Catalog

type options struct {
	crossTypeNumericComparisons  bool
	homogeneousAggregateLiterals bool
	validatedDeclarations        *Scopes
	jsonFieldNames               bool
	catalogSupplier              CatalogSupplier
}

// Option is a functional option for configuring the type-checker
type Option func(*options) error

// Catalog configures the checker with a symbol catalog supplier used for error suggestions and refinement.
func Catalog(cat CatalogSupplier) Option {
	return func(opts *options) error {
		opts.catalogSupplier = cat
		return nil
	}
}

// CrossTypeNumericComparisons toggles type-checker support for numeric comparisons across type
// See https://github.com/google/cel-spec/wiki/proposal-210 for more details.
func CrossTypeNumericComparisons(enabled bool) Option {
	return func(opts *options) error {
		opts.crossTypeNumericComparisons = enabled
		return nil
	}
}

// ValidatedDeclarations provides a reference to validated declarations which will be inherited
// as a parent scope without copying.
func ValidatedDeclarations(env *Env) Option {
	return func(opts *options) error {
		opts.validatedDeclarations = env.validatedDeclarations()
		return nil
	}
}

// JSONFieldNames enables the use of json names instead of the standard protobuf snake_case field names
func JSONFieldNames(enabled bool) Option {
	return func(opts *options) error {
		opts.jsonFieldNames = enabled
		return nil
	}
}
