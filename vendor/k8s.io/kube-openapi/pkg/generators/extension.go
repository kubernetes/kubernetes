/*
Copyright 2018 The Kubernetes Authors.

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

package generators

import (
	"fmt"
	"sort"
	"strings"

	"k8s.io/gengo/examples/set-gen/sets"
	"k8s.io/gengo/types"
)

const extensionPrefix = "x-kubernetes-"

// extensionAttributes encapsulates common traits for particular extensions.
type extensionAttributes struct {
	xName         string
	kind          types.Kind
	allowedValues sets.String
	enforceArray  bool
}

// Extension tag to openapi extension attributes
var tagToExtension = map[string]extensionAttributes{
	"patchMergeKey": {
		xName: "x-kubernetes-patch-merge-key",
		kind:  types.Slice,
	},
	"patchStrategy": {
		xName:         "x-kubernetes-patch-strategy",
		kind:          types.Slice,
		allowedValues: sets.NewString("merge", "retainKeys"),
	},
	"listMapKey": {
		xName:        "x-kubernetes-list-map-keys",
		kind:         types.Slice,
		enforceArray: true,
	},
	"listType": {
		xName:         "x-kubernetes-list-type",
		kind:          types.Slice,
		allowedValues: sets.NewString("atomic", "set", "map"),
	},
	"mapType": {
		xName:         "x-kubernetes-map-type",
		kind:          types.Map,
		allowedValues: sets.NewString("atomic", "granular"),
	},
	"structType": {
		xName:         "x-kubernetes-map-type",
		kind:          types.Struct,
		allowedValues: sets.NewString("atomic", "granular"),
	},
	"validations": {
		xName: "x-kubernetes-validations",
		kind:  types.Slice,
	},
}

// Extension encapsulates information necessary to generate an OpenAPI extension.
type extension struct {
	idlTag string   // Example: listType
	xName  string   // Example: x-kubernetes-list-type
	values []string // Example: [atomic]
}

func (e extension) hasAllowedValues() bool {
	return tagToExtension[e.idlTag].allowedValues.Len() > 0
}

func (e extension) allowedValues() sets.String {
	return tagToExtension[e.idlTag].allowedValues
}

func (e extension) hasKind() bool {
	return len(tagToExtension[e.idlTag].kind) > 0
}

func (e extension) kind() types.Kind {
	return tagToExtension[e.idlTag].kind
}

func (e extension) validateAllowedValues() error {
	// allowedValues not set means no restrictions on values.
	if !e.hasAllowedValues() {
		return nil
	}
	// Check for missing value.
	if len(e.values) == 0 {
		return fmt.Errorf("%s needs a value, none given.", e.idlTag)
	}
	// For each extension value, validate that it is allowed.
	allowedValues := e.allowedValues()
	if !allowedValues.HasAll(e.values...) {
		return fmt.Errorf("%v not allowed for %s. Allowed values: %v",
			e.values, e.idlTag, allowedValues.List())
	}
	return nil
}

func (e extension) validateType(kind types.Kind) error {
	// If this extension class has no kind, then don't validate the type.
	if !e.hasKind() {
		return nil
	}
	if kind != e.kind() {
		return fmt.Errorf("tag %s on type %v; only allowed on type %v",
			e.idlTag, kind, e.kind())
	}
	return nil
}

func (e extension) hasMultipleValues() bool {
	return len(e.values) > 1
}

func (e extension) isAlwaysArrayFormat() bool {
	return tagToExtension[e.idlTag].enforceArray
}

// Returns sorted list of map keys. Needed for deterministic testing.
func sortedMapKeys(m map[string][]string) []string {
	keys := make([]string, len(m))
	i := 0
	for k := range m {
		keys[i] = k
		i++
	}
	sort.Strings(keys)
	return keys
}

// Parses comments to return openapi extensions. Returns a list of
// extensions which parsed correctly, as well as a list of the
// parse errors. Validating extensions is performed separately.
// NOTE: Non-empty errors does not mean extensions is empty.
func parseExtensions(comments []string) ([]extension, []error) {
	extensions := []extension{}
	errors := []error{}
	// First, generate extensions from "+k8s:openapi-gen=x-kubernetes-*" annotations.
	values := getOpenAPITagValue(comments)
	for _, val := range values {
		// Example: x-kubernetes-member-tag:member_test
		if strings.HasPrefix(val, extensionPrefix) {
			parts := strings.SplitN(val, ":", 2)
			if len(parts) != 2 {
				errors = append(errors, fmt.Errorf("invalid extension value: %v", val))
				continue
			}
			e := extension{
				idlTag: tagName,            // Example: k8s:openapi-gen
				xName:  parts[0],           // Example: x-kubernetes-member-tag
				values: []string{parts[1]}, // Example: member_test
			}
			extensions = append(extensions, e)
		}
	}
	// Next, generate extensions from "idlTags" (e.g. +listType)
	tagValues := types.ExtractCommentTags("+", comments)
	for _, idlTag := range sortedMapKeys(tagValues) {
		xAttrs, exists := tagToExtension[idlTag]
		if !exists {
			continue
		}
		values := tagValues[idlTag]
		e := extension{
			idlTag: idlTag,       // listType
			xName:  xAttrs.xName, // x-kubernetes-list-type
			values: values,       // [atomic]
		}
		extensions = append(extensions, e)
	}
	return extensions, errors
}

func validateMemberExtensions(extensions []extension, m *types.Member) []error {
	errors := []error{}
	for _, e := range extensions {
		if err := e.validateAllowedValues(); err != nil {
			errors = append(errors, err)
		}
		if err := e.validateType(m.Type.Kind); err != nil {
			errors = append(errors, err)
		}
	}
	return errors
}
