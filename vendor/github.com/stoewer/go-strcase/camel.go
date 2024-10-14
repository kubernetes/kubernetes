// Copyright (c) 2017, A. Stoewer <adrian.stoewer@rz.ifi.lmu.de>
// All rights reserved.

package strcase

import (
	"strings"
)

// UpperCamelCase converts a string into camel case starting with a upper case letter.
func UpperCamelCase(s string) string {
	return camelCase(s, true)
}

// LowerCamelCase converts a string into camel case starting with a lower case letter.
func LowerCamelCase(s string) string {
	return camelCase(s, false)
}

func camelCase(s string, upper bool) string {
	s = strings.TrimSpace(s)
	buffer := make([]rune, 0, len(s))

	stringIter(s, func(prev, curr, next rune) {
		if !isDelimiter(curr) {
			if isDelimiter(prev) || (upper && prev == 0) {
				buffer = append(buffer, toUpper(curr))
			} else if isLower(prev) {
				buffer = append(buffer, curr)
			} else if isUpper(prev) && isUpper(curr) && isLower(next) {
				// Assume a case like "R" for "XRequestId"
				buffer = append(buffer, curr)
			} else {
				buffer = append(buffer, toLower(curr))
			}
		}
	})

	return string(buffer)
}
