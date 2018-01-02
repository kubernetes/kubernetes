package testutil

import (
	"strings"
	"unicode"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ErrorContains checks that the error is not nil, and contains the expected
// substring.
func ErrorContains(t require.TestingT, err error, expectedError string) {
	require.Error(t, err)
	assert.Contains(t, err.Error(), expectedError)
}

// EqualNormalizedString compare the actual value to the expected value after applying the specified
// transform function. It fails the test if these two transformed string are not equal.
// For example `EqualNormalizedString(t, RemoveSpace, "foo\n", "foo")` wouldn't fail the test as
// spaces (and thus '\n') are removed before comparing the string.
func EqualNormalizedString(t require.TestingT, transformFun func(rune) rune, actual, expected string) {
	require.Equal(t, strings.Map(transformFun, expected), strings.Map(transformFun, actual))
}

// RemoveSpace returns -1 if the specified runes is considered as a space (unicode)
// and the rune itself otherwise.
func RemoveSpace(r rune) rune {
	if unicode.IsSpace(r) {
		return -1
	}
	return r
}
