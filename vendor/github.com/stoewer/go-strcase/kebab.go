// Copyright (c) 2017, A. Stoewer <adrian.stoewer@rz.ifi.lmu.de>
// All rights reserved.

package strcase

// KebabCase converts a string into kebab case.
func KebabCase(s string) string {
	return delimiterCase(s, '-', false)
}

// UpperKebabCase converts a string into kebab case with capital letters.
func UpperKebabCase(s string) string {
	return delimiterCase(s, '-', true)
}
