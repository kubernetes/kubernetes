// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package validate

import (
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
)

// SchemaValidatorOptions defines optional rules for schema validation
type SchemaValidatorOptions struct {
	validationRulesEnabled bool
	NewValidatorForIndex   func(index int, schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, opts ...Option) ValueValidator
	NewValidatorForField   func(field string, schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, opts ...Option) ValueValidator
}

// Option sets optional rules for schema validation
type Option func(*SchemaValidatorOptions)

// Options returns current options
func (svo SchemaValidatorOptions) Options() []Option {
	return []Option{}
}
