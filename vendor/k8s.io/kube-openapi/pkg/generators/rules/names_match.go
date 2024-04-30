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

package rules

import (
	"reflect"
	"strings"

	"k8s.io/kube-openapi/pkg/util/sets"

	"k8s.io/gengo/v2/types"
)

var (
	// Blacklist of JSON tags that should skip match evaluation
	jsonTagBlacklist = sets.NewString(
		// Omitted field is ignored by the package
		"-",
	)

	// Blacklist of JSON names that should skip match evaluation
	jsonNameBlacklist = sets.NewString(
		// Empty name is used for inline struct field (e.g. metav1.TypeMeta)
		"",
		// Special case for object and list meta
		"metadata",
	)

	// List of substrings that aren't allowed in Go name and JSON name
	disallowedNameSubstrings = sets.NewString(
		// Underscore is not allowed in either name
		"_",
		// Dash is not allowed in either name. Note that since dash is a valid JSON tag, this should be checked
		// after JSON tag blacklist check.
		"-",
	)
)

/*
NamesMatch implements APIRule interface.
Go field names must be CamelCase. JSON field names must be camelCase. Other than capitalization of the
initial letter, the two should almost always match. No underscores nor dashes in either.
This rule verifies the convention "Other than capitalization of the initial letter, the two should almost always match."
Examples (also in unit test):

	Go name      | JSON name    | match
	               podSpec        false
	PodSpec        podSpec        true
	PodSpec        PodSpec        false
	podSpec        podSpec        false
	PodSpec        spec           false
	Spec           podSpec        false
	JSONSpec       jsonSpec       true
	JSONSpec       jsonspec       false
	HTTPJSONSpec   httpJSONSpec   true

NOTE: this validator cannot tell two sequential all-capital words from one word, therefore the case below
is also considered matched.

	HTTPJSONSpec   httpjsonSpec   true

NOTE: JSON names in jsonNameBlacklist should skip evaluation

	                              true
	podSpec                       true
	podSpec        -              true
	podSpec        metadata       true
*/
type NamesMatch struct{}

// Name returns the name of APIRule
func (n *NamesMatch) Name() string {
	return "names_match"
}

// Validate evaluates API rule on type t and returns a list of field names in
// the type that violate the rule. Empty field name [""] implies the entire
// type violates the rule.
func (n *NamesMatch) Validate(t *types.Type) ([]string, error) {
	fields := make([]string, 0)

	// Only validate struct type and ignore the rest
	switch t.Kind {
	case types.Struct:
		for _, m := range t.Members {
			goName := m.Name
			jsonTag, ok := reflect.StructTag(m.Tags).Lookup("json")
			// Distinguish empty JSON tag and missing JSON tag. Empty JSON tag / name is
			// allowed (in JSON name blacklist) but missing JSON tag is invalid.
			if !ok {
				fields = append(fields, goName)
				continue
			}
			if jsonTagBlacklist.Has(jsonTag) {
				continue
			}
			jsonName := strings.Split(jsonTag, ",")[0]
			if !namesMatch(goName, jsonName) {
				fields = append(fields, goName)
			}
		}
	}
	return fields, nil
}

// namesMatch evaluates if goName and jsonName match the API rule
// TODO: Use an off-the-shelf CamelCase solution instead of implementing this logic. The following existing
//
//	      packages have been tried out:
//			github.com/markbates/inflect
//			github.com/segmentio/go-camelcase
//			github.com/iancoleman/strcase
//			github.com/fatih/camelcase
//		 Please see https://github.com/kubernetes/kube-openapi/pull/83#issuecomment-400842314 for more details
//		 about why they don't satisfy our need. What we need can be a function that detects an acronym at the
//		 beginning of a string.
func namesMatch(goName, jsonName string) bool {
	if jsonNameBlacklist.Has(jsonName) {
		return true
	}
	if !isAllowedName(goName) || !isAllowedName(jsonName) {
		return false
	}
	if !strings.EqualFold(goName, jsonName) {
		return false
	}
	// Go field names must be CamelCase. JSON field names must be camelCase.
	if !isCapital(goName[0]) || isCapital(jsonName[0]) {
		return false
	}
	for i := 0; i < len(goName); i++ {
		if goName[i] == jsonName[i] {
			// goName[0:i-1] is uppercase and jsonName[0:i-1] is lowercase, goName[i:]
			// and jsonName[i:] should match;
			// goName[i] should be lowercase if i is equal to 1, e.g.:
			//	goName   | jsonName
			//	PodSpec     podSpec
			// or uppercase if i is greater than 1, e.g.:
			//      goname   | jsonName
			//      JSONSpec   jsonSpec
			// This is to rule out cases like:
			//      goname   | jsonName
			//      JSONSpec   jsonspec
			return goName[i:] == jsonName[i:] && (i == 1 || isCapital(goName[i]))
		}
	}
	return true
}

// isCapital returns true if one character is capital
func isCapital(b byte) bool {
	return b >= 'A' && b <= 'Z'
}

// isAllowedName checks the list of disallowedNameSubstrings and returns true if name doesn't contain
// any disallowed substring.
func isAllowedName(name string) bool {
	for _, substr := range disallowedNameSubstrings.UnsortedList() {
		if strings.Contains(name, substr) {
			return false
		}
	}
	return true
}
