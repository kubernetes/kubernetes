package content

import (
	"regexp"
)

const cIdentifierFmt string = "[A-Za-z_][A-Za-z0-9_]*"
const identifierErrMsg string = "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_'"

var cIdentifierRegexp = regexp.MustCompile("^" + cIdentifierFmt + "$")

// IsCIdentifier tests for a string that conforms the definition of an identifier
// in C. This checks the format, but not the length.
func IsCIdentifier(value string) []string {
	if !cIdentifierRegexp.MatchString(value) {
		return []string{RegexError(identifierErrMsg, cIdentifierFmt, "my_name", "MY_NAME", "MyName")}
	}
	return nil
}
