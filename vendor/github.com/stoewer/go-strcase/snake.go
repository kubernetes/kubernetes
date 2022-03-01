// Copyright (c) 2017, A. Stoewer <adrian.stoewer@rz.ifi.lmu.de>
// All rights reserved.

package strcase

import (
	"strings"
)

// SnakeCase converts a string into snake case.
func SnakeCase(s string) string {
	return delimiterCase(s, '_', false)
}

// UpperSnakeCase converts a string into snake case with capital letters.
func UpperSnakeCase(s string) string {
	return delimiterCase(s, '_', true)
}

// delimiterCase converts a string into snake_case or kebab-case depending on the delimiter passed
// as second argument. When upperCase is true the result will be UPPER_SNAKE_CASE or UPPER-KEBAB-CASE.
func delimiterCase(s string, delimiter rune, upperCase bool) string {
	s = strings.TrimSpace(s)
	buffer := make([]rune, 0, len(s)+3)

	adjustCase := toLower
	if upperCase {
		adjustCase = toUpper
	}

	var prev rune
	var curr rune
	for _, next := range s {
		if isDelimiter(curr) {
			if !isDelimiter(prev) {
				buffer = append(buffer, delimiter)
			}
		} else if isUpper(curr) {
			if isLower(prev) || (isUpper(prev) && isLower(next)) {
				buffer = append(buffer, delimiter)
			}
			buffer = append(buffer, adjustCase(curr))
		} else if curr != 0 {
			buffer = append(buffer, adjustCase(curr))
		}
		prev = curr
		curr = next
	}

	if len(s) > 0 {
		if isUpper(curr) && isLower(prev) && prev != 0 {
			buffer = append(buffer, delimiter)
		}
		buffer = append(buffer, adjustCase(curr))
	}

	return string(buffer)
}
