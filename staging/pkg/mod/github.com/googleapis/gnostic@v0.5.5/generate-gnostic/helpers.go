// Copyright 2017 Google LLC. All Rights Reserved.
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

package main

import (
	"strings"
	"unicode"
)

// Returns a "snake case" form of a camel-cased string.
func camelCaseToSnakeCase(input string) string {
	out := ""
	for index, runeValue := range input {
		//fmt.Printf("%#U starts at byte position %d\n", runeValue, index)
		if runeValue >= 'A' && runeValue <= 'Z' {
			if index > 0 {
				out += "_"
			}
			out += string(runeValue - 'A' + 'a')
		} else {
			out += string(runeValue)
		}
	}
	return out
}

func snakeCaseToCamelCase(input string) string {
	out := ""

	words := strings.Split(input, "_")

	for i, word := range words {
		if (i > 0) && len(word) > 0 {
			w := []rune(word)
			w[0] = unicode.ToUpper(w[0])
			out += string(w)
		} else {
			out += word
		}
	}

	return out
}
