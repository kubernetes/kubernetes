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
	"strings"
)

// compiler helper functions, usually called from generated code

func UnpackMap(in interface{}) (yaml.MapSlice, bool) {
	m, ok := in.(yaml.MapSlice)
	if ok {
		return m, ok
	} else {
		// do we have an empty array?
		a, ok := in.([]interface{})
		if ok && len(a) == 0 {
			// if so, return an empty map
			return yaml.MapSlice{}, ok
		} else {
			return nil, ok
		}
	}
}

func SortedKeysForMap(m yaml.MapSlice) []string {
	keys := make([]string, 0)
	for _, item := range m {
		keys = append(keys, item.Key.(string))
	}
	sort.Strings(keys)
	return keys
}

func MapHasKey(m yaml.MapSlice, key string) bool {
	for _, item := range m {
		itemKey, ok := item.Key.(string)
		if ok && key == itemKey {
			return true
		}
	}
	return false
}

func MapValueForKey(m yaml.MapSlice, key string) interface{} {
	for _, item := range m {
		itemKey, ok := item.Key.(string)
		if ok && key == itemKey {
			return item.Value
		}
	}
	return nil
}

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

func PatternMatches(pattern string, value string) bool {
	// if pattern contains a subpattern like "{path}", replace it with ".*"
	if pattern[0] != '^' {
		subpatternPattern := regexp.MustCompile("^.*(\\{.*\\}).*$")
		if matches := subpatternPattern.FindSubmatch([]byte(pattern)); matches != nil {
			match := string(matches[1])
			pattern = strings.Replace(pattern, match, ".*", -1)
		}
	}
	matched, err := regexp.Match(pattern, []byte(value))
	if err != nil {
		panic(err)
	}
	return matched
}

func MissingKeysInMap(m yaml.MapSlice, requiredKeys []string) []string {
	missingKeys := make([]string, 0)
	for _, k := range requiredKeys {
		if !MapHasKey(m, k) {
			missingKeys = append(missingKeys, k)
		}
	}
	return missingKeys
}

func InvalidKeysInMap(m yaml.MapSlice, allowedKeys []string, allowedPatterns []string) []string {
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
					if PatternMatches(allowedPattern, key) {
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

// describe a map (for debugging purposes)
func DescribeMap(in interface{}, indent string) string {
	description := ""
	m, ok := in.(map[string]interface{})
	if ok {
		keys := make([]string, 0)
		for k, _ := range m {
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

func PluralProperties(count int) string {
	if count == 1 {
		return "property"
	} else {
		return "properties"
	}
}

func StringArrayContainsValue(array []string, value string) bool {
	for _, item := range array {
		if item == value {
			return true
		}
	}
	return false
}

func StringArrayContainsValues(array []string, values []string) bool {
	for _, value := range values {
		if !StringArrayContainsValue(array, value) {
			return false
		}
	}
	return true
}
