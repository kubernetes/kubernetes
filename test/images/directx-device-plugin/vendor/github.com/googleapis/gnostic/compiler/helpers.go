// Copyright 2017 Google Inc. All Rights Reserved.
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

package compiler

import (
	"fmt"
	"gopkg.in/yaml.v2"
	"regexp"
	"sort"
	"strconv"
)

// compiler helper functions, usually called from generated code

// UnpackMap gets a yaml.MapSlice if possible.
func UnpackMap(in interface{}) (yaml.MapSlice, bool) {
	m, ok := in.(yaml.MapSlice)
	if ok {
		return m, true
	}
	// do we have an empty array?
	a, ok := in.([]interface{})
	if ok && len(a) == 0 {
		// if so, return an empty map
		return yaml.MapSlice{}, true
	}
	return nil, false
}

// SortedKeysForMap returns the sorted keys of a yaml.MapSlice.
func SortedKeysForMap(m yaml.MapSlice) []string {
	keys := make([]string, 0)
	for _, item := range m {
		keys = append(keys, item.Key.(string))
	}
	sort.Strings(keys)
	return keys
}

// MapHasKey returns true if a yaml.MapSlice contains a specified key.
func MapHasKey(m yaml.MapSlice, key string) bool {
	for _, item := range m {
		itemKey, ok := item.Key.(string)
		if ok && key == itemKey {
			return true
		}
	}
	return false
}

// MapValueForKey gets the value of a map value for a specified key.
func MapValueForKey(m yaml.MapSlice, key string) interface{} {
	for _, item := range m {
		itemKey, ok := item.Key.(string)
		if ok && key == itemKey {
			return item.Value
		}
	}
	return nil
}

// ConvertInterfaceArrayToStringArray converts an array of interfaces to an array of strings, if possible.
func ConvertInterfaceArrayToStringArray(interfaceArray []interface{}) []string {
	stringArray := make([]string, 0)
	for _, item := range interfaceArray {
		v, ok := item.(string)
		if ok {
			stringArray = append(stringArray, v)
		}
	}
	return stringArray
}

// MissingKeysInMap identifies which keys from a list of required keys are not in a map.
func MissingKeysInMap(m yaml.MapSlice, requiredKeys []string) []string {
	missingKeys := make([]string, 0)
	for _, k := range requiredKeys {
		if !MapHasKey(m, k) {
			missingKeys = append(missingKeys, k)
		}
	}
	return missingKeys
}

// InvalidKeysInMap returns keys in a map that don't match a list of allowed keys and patterns.
func InvalidKeysInMap(m yaml.MapSlice, allowedKeys []string, allowedPatterns []*regexp.Regexp) []string {
	invalidKeys := make([]string, 0)
	for _, item := range m {
		itemKey, ok := item.Key.(string)
		if ok {
			key := itemKey
			found := false
			// does the key match an allowed key?
			for _, allowedKey := range allowedKeys {
				if key == allowedKey {
					found = true
					break
				}
			}
			if !found {
				// does the key match an allowed pattern?
				for _, allowedPattern := range allowedPatterns {
					if allowedPattern.MatchString(key) {
						found = true
						break
					}
				}
				if !found {
					invalidKeys = append(invalidKeys, key)
				}
			}
		}
	}
	return invalidKeys
}

// DescribeMap describes a map (for debugging purposes).
func DescribeMap(in interface{}, indent string) string {
	description := ""
	m, ok := in.(map[string]interface{})
	if ok {
		keys := make([]string, 0)
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			v := m[k]
			description += fmt.Sprintf("%s%s:\n", indent, k)
			description += DescribeMap(v, indent+"  ")
		}
		return description
	}
	a, ok := in.([]interface{})
	if ok {
		for i, v := range a {
			description += fmt.Sprintf("%s%d:\n", indent, i)
			description += DescribeMap(v, indent+"  ")
		}
		return description
	}
	description += fmt.Sprintf("%s%+v\n", indent, in)
	return description
}

// PluralProperties returns the string "properties" pluralized.
func PluralProperties(count int) string {
	if count == 1 {
		return "property"
	}
	return "properties"
}

// StringArrayContainsValue returns true if a string array contains a specified value.
func StringArrayContainsValue(array []string, value string) bool {
	for _, item := range array {
		if item == value {
			return true
		}
	}
	return false
}

// StringArrayContainsValues returns true if a string array contains all of a list of specified values.
func StringArrayContainsValues(array []string, values []string) bool {
	for _, value := range values {
		if !StringArrayContainsValue(array, value) {
			return false
		}
	}
	return true
}

// StringValue returns the string value of an item.
func StringValue(item interface{}) (value string, ok bool) {
	value, ok = item.(string)
	if ok {
		return value, ok
	}
	intValue, ok := item.(int)
	if ok {
		return strconv.Itoa(intValue), true
	}
	return "", false
}
