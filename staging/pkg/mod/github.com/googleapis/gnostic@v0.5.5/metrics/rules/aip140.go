// Copyright 2020 Google LLC. All Rights Reserved.
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

package rules

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/stoewer/go-strcase"
)

// checkSnakeCase ensures that the field is in lower snake case.
// If not, the function returns false and the suggested reformat
// of the field.
func checkSnakeCase(field string) (bool, string) {
	snake := strcase.SnakeCase(field)
	snake = strings.ToLower(snake)

	return snake == field, snake
}

// checkAbbreviation checks if the field name is a common abbreviation.
// If true, the functions returns true and the suggested abbreviation.
func checkAbbreviation(field string) (bool, string) {
	var expectedAbbreviations = map[string]string{
		"configuration": "config",
		"identifier":    "id",
		"information":   "info",
		"specification": "spec",
		"statistics":    "stats",
	}

	if suggestion, exists := expectedAbbreviations[field]; exists {
		return true, suggestion
	}
	return false, field
}

// checkNumbers ensures that no word within the field name begins with a number.
// If it starts with a number, the function returns true. False if not.
func checkNumbers(field string) bool {
	var numberStart = regexp.MustCompile("^[0-9]")
	for _, segment := range strings.Split(field, "_") {
		if numberStart.MatchString(segment) {
			return true
		}
	}
	return false
}

// checkReservedWords ensures that no word within the field is a reserved word.
// If it is a reserved word, the function returns true. False if not.
func checkReservedWords(field string) bool {
	reservedWordsSet := []string{"abstract", "and", "arguments", "as", "assert", "async", "await", "boolean", "break", "byte",
		"case", "catch", "char", "class", "const", "continue", "debugger", "def", "default", "del", "delete", "do", "double", "elif",
		"else", "enum", "eval", "except", "export", "extends", "false", "final", "finally", "float", "for", "from", "function", "global",
		"goto", "if", "implements", "import", "in", "instanceof", "int", "interface", "is", "lambda", "let", "long", "native", "new", "nonlocal",
		"not", "null", "or", "package", "pass", "private", "protected", "public", "raise", "return", "short", "static", "strictfp",
		"super", "switch", "synchronized", "this", "throw", "throws", "transient", "true", "try", "typeof", "var", "void", "volatile",
		"while", "with", "yield"}

	for _, segment := range strings.Split(field, "_") {
		result := sort.SearchStrings(reservedWordsSet, segment)
		if result < len(reservedWordsSet) && reservedWordsSet[result] == segment {
			return true
		}
	}
	return false
}

// checkPrepositions ensures that no word within the field name is a preposition.
// If it is a preposition, the function returns true. False if not.
func checkPrepositions(field string) bool {
	preps := []string{"after", "at", "before", "between", "but", "by", "except",
		"for", "from", "in", "including", "into", "of", "over", "since", "to",
		"toward", "under", "upon", "with", "within", "without"}
	for _, segment := range strings.Split(field, "_") {
		result := sort.SearchStrings(preps, segment)
		if result < len(preps) && preps[result] == segment {
			return true
		}
	}
	return false
}

// AIP140Driver calls all functions for AIP rule 140
func AIP140Driver(f Field) []MessageType {
	messages := make([]MessageType, 0)
	val, sugg := checkSnakeCase(f.Name)
	if !val {
		m := []string{"Error", "Parameter names must follow case convention: lower_snake_case\n",
			fmt.Sprintf("Rename field %s to %s\n", f.Name, sugg)}
		temp := MessageType{Message: m, Path: f.Path}
		messages = append(messages, temp)
	}
	val, sugg = checkAbbreviation(f.Name)
	if val {
		m := []string{"Error", "Parameters should use common abbreviations if applicable\n",
			fmt.Sprintf("Rename field %s to %s\n", f.Name, sugg)}
		temp := MessageType{Message: m, Path: f.Path}
		messages = append(messages, temp)

	}
	val = checkNumbers(f.Name)
	if val {
		m := []string{"Error", fmt.Sprintf("Parameters must not begin with a number: %s\n", f.Name),
			""}
		temp := MessageType{Message: m, Path: f.Path}
		messages = append(messages, temp)

	}
	val = checkReservedWords(f.Name)
	if val {
		m := []string{"Error", fmt.Sprintf("Parameter names must not be reserved words: %s\n", f.Name),
			""}
		temp := MessageType{Message: m, Path: f.Path}
		messages = append(messages, temp)

	}
	val = checkPrepositions(f.Name)
	if val {
		m := []string{"Error", fmt.Sprintf("Parameter must not include prepositions in their names: %s\n", f.Name),
			""}
		temp := MessageType{Message: m, Path: f.Path}
		messages = append(messages, temp)

	}
	return messages

}
