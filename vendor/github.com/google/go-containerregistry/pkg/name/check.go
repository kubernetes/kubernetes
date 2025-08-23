// Copyright 2018 Google LLC All Rights Reserved.
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

package name

import (
	"strings"
	"unicode/utf8"
)

// stripRunesFn returns a function which returns -1 (i.e. a value which
// signals deletion in strings.Map) for runes in 'runes', and the rune otherwise.
func stripRunesFn(runes string) func(rune) rune {
	return func(r rune) rune {
		if strings.ContainsRune(runes, r) {
			return -1
		}
		return r
	}
}

// checkElement checks a given named element matches character and length restrictions.
// Returns true if the given element adheres to the given restrictions, false otherwise.
func checkElement(name, element, allowedRunes string, minRunes, maxRunes int) error {
	numRunes := utf8.RuneCountInString(element)
	if (numRunes < minRunes) || (maxRunes < numRunes) {
		return newErrBadName("%s must be between %d and %d characters in length: %s", name, minRunes, maxRunes, element)
	} else if len(strings.Map(stripRunesFn(allowedRunes), element)) != 0 {
		return newErrBadName("%s can only contain the characters `%s`: %s", name, allowedRunes, element)
	}
	return nil
}
