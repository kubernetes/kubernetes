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

	"github.com/golang/glog"
	"k8s.io/gengo/examples/set-gen/sets"
	"k8s.io/gengo/types"
)

const extensionTag = "x-kubernetes-"

// Extension tag to openapi extension
var tagToExtension = map[string]string{
	"patchMergeKey": "x-kubernetes-patch-merge-key",
	"patchStrategy": "x-kubernetes-patch-strategy",
	"listType":      "x-kubernetes-list-type",
	"listMapKey":    "x-kubernetes-list-map-keys",
}

// Enum values per extension
var allowedExtensionValues = map[string]sets.String{
	"x-kubernetes-patch-strategy": sets.NewString("merge", "retainKeys"),
	"x-kubernetes-list-type":      sets.NewString("atomic", "set", "map"),
}

// Extension encapsulates information necessary to generate an OpenAPI extension.
// TODO(seans3): Create interface for an extension validation function, and create
// validation functions for different extensions.
type extension struct {
	name   string   // Example: x-kubernetes-list-type
	values []string // Example: atomic
}

func (e extension) validateAllowedValues() error {
	// allowedValues not set means no restrictions on values.
	allowedValues, exists := allowedExtensionValues[e.name]
	if !exists {
		return nil
	}
	// For each extension value, validate that it is allowed.
	if !allowedValues.HasAll(e.values...) {
		return fmt.Errorf("%s: value(s) %v not allowed. Allowed values: %v\n",
			e.name, e.values, allowedValues.List())
	}
	return nil
}

func (e extension) hasMultipleValues() bool {
	return len(e.values) > 1
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

// Parses comments to return openapi extensions
func parseExtensions(comments []string) []extension {
	extensions := []extension{}
	// First, generate extensions from "+k8s:openapi-gen=x-kubernetes-*" annotations.
	values := getOpenAPITagValue(comments)
	for _, val := range values {
		// Example: x-kubernetes-member-tag:member_test
		if strings.HasPrefix(val, extensionTag) {
			parts := strings.SplitN(val, ":", 2)
			if len(parts) != 2 {
				// TODO(seans3): Spit out source file/line number with error.
				glog.V(2).Info(fmt.Sprintf("Invalid extension value: %v", val))
				continue
			}
			e := extension{
				name:   parts[0],           // Example: x-kubernetes-member-tag
				values: []string{parts[1]}, // Example: member_test
			}
			extensions = append(extensions, e)
		}
	}
	// Next, generate extensions from "tags" (e.g. +listType)
	tagValues := types.ExtractCommentTags("+", comments)
	for _, tag := range sortedMapKeys(tagValues) {
		extensionName, exists := tagToExtension[tag]
		if !exists {
			continue
		}
		extensionValues := tagValues[tag]
		e := extension{
			name:   extensionName,
			values: extensionValues,
		}
		// TODO(seans3): Validate the number of allowed values
		// TODO(seans3): Validate the field type for this extension.
		// Example: +listType is only allowed on an array.
		if err := e.validateAllowedValues(); err != nil {
			// For now, only log the extension validation errors.
			glog.V(2).Info(err)
		}
		extensions = append(extensions, e)
	}
	return extensions
}
