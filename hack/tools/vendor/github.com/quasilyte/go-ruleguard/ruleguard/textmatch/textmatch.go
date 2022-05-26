package textmatch

import "regexp"

// Pattern is a compiled regular expression.
type Pattern interface {
	MatchString(s string) bool
	Match(b []byte) bool
}

// Compile parses a regular expression and returns a compiled
// pattern that can match inputs descriped by the regexp.
//
// Semantically it's close to the regexp.Compile, but
// it does recognize some common patterns and creates
// a more optimized matcher for them.
func Compile(re string) (Pattern, error) {
	return compile(re)
}

// IsRegexp reports whether p is implemented using regexp.
// False means that the underlying matcher is something optimized.
func IsRegexp(p Pattern) bool {
	_, ok := p.(*regexp.Regexp)
	return ok
}
