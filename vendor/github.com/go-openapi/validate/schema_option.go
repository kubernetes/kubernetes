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

// SchemaValidatorOptions defines optional rules for schema validation
type SchemaValidatorOptions struct {
	EnableObjectArrayTypeCheck    bool
	EnableArrayMustHaveItemsCheck bool
}

// Option sets optional rules for schema validation
type Option func(*SchemaValidatorOptions)

// EnableObjectArrayTypeCheck activates the swagger rule: an items must be in type: array
func EnableObjectArrayTypeCheck(enable bool) Option {
	return func(svo *SchemaValidatorOptions) {
		svo.EnableObjectArrayTypeCheck = enable
	}
}

// EnableArrayMustHaveItemsCheck activates the swagger rule: an array must have items defined
func EnableArrayMustHaveItemsCheck(enable bool) Option {
	return func(svo *SchemaValidatorOptions) {
		svo.EnableArrayMustHaveItemsCheck = enable
	}
}

// SwaggerSchema activates swagger schema validation rules
func SwaggerSchema(enable bool) Option {
	return func(svo *SchemaValidatorOptions) {
		svo.EnableObjectArrayTypeCheck = enable
		svo.EnableArrayMustHaveItemsCheck = enable
	}
}

// Options returns current options
func (svo SchemaValidatorOptions) Options() []Option {
	return []Option{
		EnableObjectArrayTypeCheck(svo.EnableObjectArrayTypeCheck),
		EnableArrayMustHaveItemsCheck(svo.EnableArrayMustHaveItemsCheck),
	}
}
