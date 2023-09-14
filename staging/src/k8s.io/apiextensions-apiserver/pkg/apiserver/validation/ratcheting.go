/*
Copyright 2023 The Kubernetes Authors.

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

package validation

import (
	"reflect"

	"k8s.io/apiserver/pkg/cel/common"
	celopenapi "k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
	"k8s.io/kube-openapi/pkg/validation/validate"
)

// schemaArgs are the arguments to constructor for OpenAPI schema validator,
// NewSchemaValidator
type schemaArgs struct {
	schema       *spec.Schema
	root         interface{}
	path         string
	knownFormats strfmt.Registry
	options      []validate.Option
}

// RatchetingSchemaValidator wraps kube-openapis SchemaValidator to provide a
// ValidateUpdate function which allows ratcheting
type RatchetingSchemaValidator struct {
	schemaArgs
}

func NewRatchetingSchemaValidator(schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, options ...validate.Option) *RatchetingSchemaValidator {
	return &RatchetingSchemaValidator{
		schemaArgs: schemaArgs{
			schema:       schema,
			root:         rootSchema,
			path:         root,
			knownFormats: formats,
			options:      options,
		},
	}
}

func (r *RatchetingSchemaValidator) Validate(new interface{}) *validate.Result {
	sv := validate.NewSchemaValidator(r.schema, r.root, r.path, r.knownFormats, r.options...)
	return sv.Validate(new)
}

func (r *RatchetingSchemaValidator) ValidateUpdate(new, old interface{}) *validate.Result {
	return newRatchetingValueValidator(new, old, r.schemaArgs).Validate()
}

// ratchetingValueValidator represents an invocation of SchemaValidator.ValidateUpdate
// for specific arguments for `old` and `new`
//
// It follows the openapi SchemaValidator down its traversal of the new value
// by injecting validate.Option into each recursive invocation.
//
// A ratchetingValueValidator will be constructed and added to the tree for
// each explored sub-index and sub-property during validation.
//
// It's main job is to keep the old/new values correlated as the traversal
// continues, and postprocess errors according to our ratcheting policy.
//
// ratchetingValueValidator is not thread safe.
type ratchetingValueValidator struct {
	// schemaArgs provides the arguments to use in the temporary SchemaValidator
	// that is created during a call to Validate.
	schemaArgs

	// Currently correlated old value during traversal of the schema/object
	oldValue interface{}

	// Value being validated
	value interface{}

	// Scratch space below, may change during validation

	// Cached comparison result of DeepEqual of `value` and `thunk.oldValue`
	comparisonResult *bool

	// Cached map representation of a map-type list, or nil if not map-type list
	mapList common.MapList

	// Children spawned by a call to `Validate` on this object
	// key is either a string or an index, depending upon whether `value` is
	// a map or a list, respectively.
	//
	// The list of children may be incomplete depending upon if the internal
	// logic of kube-openapi's SchemaValidator short-circuited before
	// reaching all of the children.
	//
	// It should be expected to have an entry for either all of the children, or
	// none of them.
	children map[interface{}]*ratchetingValueValidator
}

func newRatchetingValueValidator(newValue, oldValue interface{}, args schemaArgs) *ratchetingValueValidator {
	return &ratchetingValueValidator{
		oldValue:   oldValue,
		value:      newValue,
		schemaArgs: args,
		children:   map[interface{}]*ratchetingValueValidator{},
	}
}

// getValidateOption provides a kube-openapi validate.Option for SchemaValidator
// that injects a ratchetingValueValidator to be used for all subkeys and subindices
func (r *ratchetingValueValidator) getValidateOption() validate.Option {
	return func(svo *validate.SchemaValidatorOptions) {
		svo.NewValidatorForField = func(field string, schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, opts ...validate.Option) validate.ValueValidator {
			return r.SubPropertyValidator(field, schema, rootSchema, root, formats, opts...)
		}
		svo.NewValidatorForIndex = func(index int, schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, opts ...validate.Option) validate.ValueValidator {
			return r.SubIndexValidator(index, schema, rootSchema, root, formats, opts...)
		}
	}
}

// Validate validates the update from r.oldValue to r.value
//
// During evaluation, a temporary tree of ratchetingValueValidator is built for all
// traversed field paths. It is necessary to build the tree to take advantage of
// DeepEqual checks performed by lower levels of the object during validation without
// greatly modifying `kube-openapi`'s implementation.
//
// The tree, and all cache storage/scratch space for the validation of a single
// call to `Validate` is thrown away at the end of the top-level call
// to `Validate`.
//
// `Validate` will create a node in the tree to for each of the explored children.
// The node's main purpose is to store a lazily computed DeepEqual check between
// the oldValue and the currently passed value. If the check is performed, it
// will be stored in the node to be re-used by a parent node during a DeepEqual
// comparison, if necessary.
//
// This call has a side-effect of populating it's `children` variable with
// the explored nodes of the object tree.
func (r *ratchetingValueValidator) Validate() *validate.Result {
	opts := append([]validate.Option{
		r.getValidateOption(),
	}, r.options...)

	s := validate.NewSchemaValidator(r.schema, r.root, r.path, r.knownFormats, opts...)

	res := s.Validate(r.value)

	if res.IsValid() {
		return res
	}

	// Current ratcheting rule is to ratchet errors if DeepEqual(old, new) is true.
	if r.CachedDeepEqual() {
		newRes := &validate.Result{}
		newRes.MergeAsWarnings(res)
		return newRes
	}

	return res
}

// SubPropertyValidator overrides the standard validator constructor for sub-properties by
// returning our special ratcheting variant.
//
// If we can correlate an old value, we return a ratcheting validator to
// use for the child.
//
// If the old value cannot be correlated, then default validation is used.
func (r *ratchetingValueValidator) SubPropertyValidator(field string, schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, options ...validate.Option) validate.ValueValidator {
	// Find correlated old value
	asMap, ok := r.oldValue.(map[string]interface{})
	if !ok {
		return validate.NewSchemaValidator(schema, rootSchema, root, formats, options...)
	}

	oldValueForField, ok := asMap[field]
	if !ok {
		return validate.NewSchemaValidator(schema, rootSchema, root, formats, options...)
	}

	return inlineValidator(func(new interface{}) *validate.Result {
		childNode := newRatchetingValueValidator(new, oldValueForField, schemaArgs{
			schema:       schema,
			root:         rootSchema,
			path:         root,
			knownFormats: formats,
			options:      options,
		})

		r.children[field] = childNode
		return childNode.Validate()
	})

}

// SubIndexValidator overrides the standard validator constructor for sub-indicies by
// returning our special ratcheting variant.
//
// If we can correlate an old value, we return a ratcheting validator to
// use for the child.
//
// If the old value cannot be correlated, then default validation is used.
func (r *ratchetingValueValidator) SubIndexValidator(index int, schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, options ...validate.Option) validate.ValueValidator {
	oldValueForIndex := r.correlateOldValueForChildAtNewIndex(index)
	if oldValueForIndex == nil {
		// If correlation fails, default to non-ratcheting logic
		return validate.NewSchemaValidator(schema, rootSchema, root, formats, options...)
	}

	return inlineValidator(func(new interface{}) *validate.Result {
		childNode := newRatchetingValueValidator(new, oldValueForIndex, schemaArgs{
			schema:       schema,
			root:         rootSchema,
			path:         root,
			knownFormats: formats,
			options:      options,
		})

		r.children[index] = childNode
		return childNode.Validate()
	})
}

// If oldValue is not a list, returns nil
// If oldValue is a list takes mapType into account and attempts to find the
// old value with the same index or key, depending upon the mapType.
//
// If listType is map, creates a map representation of the list using the designated
// map-keys and caches it for future calls.
func (r *ratchetingValueValidator) correlateOldValueForChildAtNewIndex(index int) any {
	oldAsList, ok := r.oldValue.([]interface{})
	if !ok {
		return nil
	}

	asList, ok := r.value.([]interface{})
	if !ok {
		return nil
	} else if len(asList) <= index {
		// Cannot correlate out of bounds index
		return nil
	}

	listType, _ := r.schema.Extensions.GetString("x-kubernetes-list-type")
	switch listType {
	case "map":
		// Look up keys for this index in current object
		currentElement := asList[index]

		oldList := r.mapList
		if oldList == nil {
			oldList = celopenapi.MakeMapList(r.schema, oldAsList)
			r.mapList = oldList
		}
		return oldList.Get(currentElement)

	case "set":
		// Are sets correlatable? Only if the old value equals the current value.
		// We might be able to support this, but do not currently see a lot
		// of value
		// (would allow you to add/remove items from sets with ratcheting but not change them)
		return nil
	case "atomic":
		// Atomic lists are not correlatable by item
		// Ratcheting is not available on a per-index basis
		return nil
	default:
		// Correlate by-index by default.
		//
		// Cannot correlate an out-of-bounds index
		if len(oldAsList) <= index {
			return nil
		}

		return oldAsList[index]
	}
}

// CachedDeepEqual is equivalent to reflect.DeepEqual, but caches the
// results in the tree of ratchetInvocationScratch objects on the way:
//
// For objects and arrays, this function will make a best effort to make
// use of past DeepEqual checks performed by this Node's children, if available.
//
// If a lazy computation could not be found for all children possibly due
// to validation logic short circuiting and skipping the children, then
// this function simply defers to reflect.DeepEqual.
func (r *ratchetingValueValidator) CachedDeepEqual() (res bool) {
	if r.comparisonResult != nil {
		return *r.comparisonResult
	}

	defer func() {
		r.comparisonResult = &res
	}()

	if r.value == nil && r.oldValue == nil {
		return true
	} else if r.value == nil || r.oldValue == nil {
		return false
	}

	oldAsArray, oldIsArray := r.oldValue.([]interface{})
	newAsArray, newIsArray := r.value.([]interface{})

	if oldIsArray != newIsArray {
		return false
	} else if oldIsArray {
		if len(oldAsArray) != len(newAsArray) {
			return false
		} else if len(r.children) != len(oldAsArray) {
			// kube-openapi validator is written to always visit all
			// children of a slice, so this case is only possible if
			// one of the children could not be correlated. In that case,
			// we know the objects are not equal.
			//
			return false
		}

		// Correctly considers map-type lists due to fact that index here
		// is only used for numbering. The correlation is stored in the
		// childInvocation itself
		//
		// NOTE: This does not consider sets, since we don't correlate them.
		for i := range newAsArray {
			// Query for child
			child, ok := r.children[i]
			if !ok {
				// This should not happen
				return false
			} else if !child.CachedDeepEqual() {
				// If one child is not equal the entire object is not equal
				return false
			}
		}

		return true
	}

	oldAsMap, oldIsMap := r.oldValue.(map[string]interface{})
	newAsMap, newIsMap := r.value.(map[string]interface{})

	if oldIsMap != newIsMap {
		return false
	} else if oldIsMap {
		if len(oldAsMap) != len(newAsMap) {
			return false
		} else if len(oldAsMap) == 0 && len(newAsMap) == 0 {
			// Both empty
			return true
		} else if len(r.children) != len(oldAsMap) {
			// If we are missing a key it is because the old value could not
			// be correlated to the new, so the objects are not equal.
			//
			return false
		}

		for k := range oldAsMap {
			// Check to see if this child was explored during validation
			child, ok := r.children[k]
			if !ok {
				// Child from old missing in new due to key change
				// Objects are not equal.
				return false
			} else if !child.CachedDeepEqual() {
				// If one child is not equal the entire object is not equal
				return false
			}
		}

		return true
	}

	return reflect.DeepEqual(r.oldValue, r.value)
}

// A validator which just calls a validate function, and advertises that it
// validates anything
//
// In the future kube-openapi's ValueValidator interface can be simplified
// to be closer to `currentValidator.Options.NewValidator(value, ...).Validate()`
// so that a tree of "validation nodes" can be more formally encoded in the API.
// In that case this class would not be necessary.
type inlineValidator func(new interface{}) *validate.Result

var _ validate.ValueValidator = inlineValidator(nil)

func (f inlineValidator) Validate(new interface{}) *validate.Result {
	return f(new)
}

func (f inlineValidator) SetPath(path string) {
	// Do nothing
	// Unused by kube-openapi
}

func (f inlineValidator) Applies(source interface{}, valueKind reflect.Kind) bool {
	return true
}
