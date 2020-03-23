/*
Copyright 2017 The Kubernetes Authors.

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

package parse

import (
	"fmt"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kubectl/pkg/apply"
)

// nilSafeLookup returns the value from the map if the map is non-nil
func nilSafeLookup(key string, from map[string]interface{}) (interface{}, bool) {
	if from != nil {
		value, found := from[key]
		return value, found
	}
	// Not present
	return nil, false
}

// boundsSafeLookup returns the value from the slice if the slice is non-nil and
// the index is in bounds.
func boundsSafeLookup(index int, from []interface{}) (interface{}, bool) {
	if from != nil && len(from) > index {
		return from[index], true
	}
	return nil, false
}

// keysUnion returns a slice containing the union of the keys present in the arguments
func keysUnion(maps ...map[string]interface{}) []string {
	keys := map[string]interface{}{}
	for _, m := range maps {
		for k := range m {
			keys[k] = nil
		}
	}
	result := []string{}
	for key := range keys {
		result = append(result, key)
	}
	return result
}

// max returns the argument with the highest value
func max(values ...int) int {
	v := 0
	for _, i := range values {
		if i > v {
			v = i
		}
	}
	return v
}

// getType returns the type of the arguments.  If the arguments don't have matching
// types, getType returns an error.  Nil types matching everything.
func getType(args ...interface{}) (reflect.Type, error) {
	var last interface{}
	for _, next := range args {
		// Skip nil values
		if next == nil {
			continue
		}

		// Set the first non-nil value we find and continue
		if last == nil {
			last = next
			continue
		}

		// Verify the types of the values match
		if reflect.TypeOf(last).Kind() != reflect.TypeOf(next).Kind() {
			return nil, fmt.Errorf("missmatching non-nil types for the same field: %T %T", last, next)
		}
	}

	return reflect.TypeOf(last), nil
}

// getFieldMeta parses the metadata about the field from the openapi spec
func getFieldMeta(s proto.Schema, name string) (apply.FieldMetaImpl, error) {
	m := apply.FieldMetaImpl{}
	if s != nil {
		ext := s.GetExtensions()
		if e, found := ext["x-kubernetes-patch-strategy"]; found {
			strategy, ok := e.(string)
			if !ok {
				return apply.FieldMetaImpl{}, fmt.Errorf("Expected string for x-kubernetes-patch-strategy by got %T", e)
			}

			// Take the first strategy if there are substrategies.
			// Sub strategies are copied to sub types in openapi.go
			strategies := strings.Split(strategy, ",")
			if len(strategies) > 2 {
				return apply.FieldMetaImpl{}, fmt.Errorf("Expected between 0 and 2 elements for x-kubernetes-patch-merge-strategy by got %v", strategies)
			}
			// For lists, choose the strategy for this type, not the subtype
			m.MergeType = strategies[0]
		}
		if k, found := ext["x-kubernetes-patch-merge-key"]; found {
			key, ok := k.(string)
			if !ok {
				return apply.FieldMetaImpl{}, fmt.Errorf("Expected string for x-kubernetes-patch-merge-key by got %T", k)
			}
			m.MergeKeys = apply.MergeKeys(strings.Split(key, ","))
		}
	}
	m.Name = name
	return m, nil
}

// getCommonGroupVersionKind verifies that the recorded, local and remote all share
// the same GroupVersionKind and returns the value
func getCommonGroupVersionKind(recorded, local, remote map[string]interface{}) (schema.GroupVersionKind, error) {
	recordedGVK, err := getGroupVersionKind(recorded)
	if err != nil {
		return schema.GroupVersionKind{}, err
	}
	localGVK, err := getGroupVersionKind(local)
	if err != nil {
		return schema.GroupVersionKind{}, err
	}
	remoteGVK, err := getGroupVersionKind(remote)
	if err != nil {
		return schema.GroupVersionKind{}, err
	}

	if !reflect.DeepEqual(recordedGVK, localGVK) || !reflect.DeepEqual(localGVK, remoteGVK) {
		return schema.GroupVersionKind{},
			fmt.Errorf("group version kinds do not match (recorded: %v local: %v remote: %v)",
				recordedGVK, localGVK, remoteGVK)
	}
	return recordedGVK, nil
}

// getGroupVersionKind returns the GroupVersionKind of the object
func getGroupVersionKind(config map[string]interface{}) (schema.GroupVersionKind, error) {
	gvk := schema.GroupVersionKind{}
	if gv, found := config["apiVersion"]; found {
		casted, ok := gv.(string)
		if !ok {
			return gvk, fmt.Errorf("Expected string for apiVersion, found %T", gv)
		}
		s := strings.Split(casted, "/")
		if len(s) != 1 {
			gvk.Group = s[0]
		}
		gvk.Version = s[len(s)-1]
	} else {
		return gvk, fmt.Errorf("Missing apiVersion in Kind %v", config)
	}
	if k, found := config["kind"]; found {
		casted, ok := k.(string)
		if !ok {
			return gvk, fmt.Errorf("Expected string for kind, found %T", k)
		}
		gvk.Kind = casted
	} else {
		return gvk, fmt.Errorf("Missing kind in Kind %v", config)
	}
	return gvk, nil
}
