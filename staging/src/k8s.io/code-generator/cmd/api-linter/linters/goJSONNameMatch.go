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

// Go field names must be CamelCase. JSON field names must be camelCase. Other than capitalization of the initial letter, the two should almost always match. No underscores nor dashes in either.
// This validator verifies the convention "Other than capitalization of the initial letter, the two should almost always match."
// Examples (also in unit test):
//     Go name | JSON name | match
//     PodSpec   podSpec     true
//     PodSpec   spec        false
//     Spec      podSpec     false
//     JSONSpec  jsonSpec    true
//     JSONSpec  jsonspec    true

package linters

import (
	"fmt"
	"reflect"
	"strings"

	"k8s.io/gengo/types"
)

type goJSONNameMatchAPIConvention struct{}

func (c goJSONNameMatchAPIConvention) Name() string {
	return "go_json_name_match"
}

func (c goJSONNameMatchAPIConvention) Validate(t *types.Type) ([]string, error) {
	violationIDs := make([]string, 0)

	// Only validate struct type and ignore the rest
	switch t.Kind {
	case types.Struct:
		for _, m := range t.Members {
			goName := m.Name
			jsonTag := reflect.StructTag(m.Tags).Get("json")
			jsonName := strings.Split(jsonTag, ",")[0]
			// Skip empty json name "" and omitted json name "-"
			// Object and list meta are special cases. Skip when json name is "metadata"
			if jsonName == "" || jsonName == "-" || jsonName == "metadata" {
				continue
			}

			if !goJSONNameMatch(goName, jsonName) {
				violationIDs = append(violationIDs, fmt.Sprintf("%v:%v", goName, jsonName))
			}
		}
	}
	return violationIDs, nil
}

func goJSONNameMatch(goName, jsonName string) bool {
	if len(goName) != len(jsonName) {
		return false
	}

	uppercaseLength := 0
	for ; uppercaseLength < len(goName); uppercaseLength++ {
		if goName[uppercaseLength] < 'A' || goName[uppercaseLength] > 'Z' {
			break
		}
	}
	for j := 0; j < uppercaseLength-1; j++ {
		if goName[j]+32 != jsonName[j] {
			return false
		}
	}

	return uppercaseLength >= len(goName) || strings.Compare(goName[uppercaseLength:], jsonName[uppercaseLength:]) == 0
}
