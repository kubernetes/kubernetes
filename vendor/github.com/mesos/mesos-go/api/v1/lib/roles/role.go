package roles

import (
	"fmt"
	"strings"
	"unicode"
)

// Role is a deprecated type.
type Role string

const defaultRole = Role("*")

func (r Role) IsDefault() bool {
	return r == defaultRole
}

func (r Role) Assign() func(interface{}) {
	return func(v interface{}) {
		type roler interface {
			WithRole(string)
		}
		if ri, ok := v.(roler); ok {
			ri.WithRole(string(r))
		}
	}
}

func (r Role) Proto() *string {
	s := string(r)
	return &s
}

// IsStrictSubroleOf returns true if left is a strict subrole of right.
func IsStrictSubroleOf(left, right string) bool {
	return len(left) > len(right) && left[len(right)] == '/' && strings.HasPrefix(left, right)
}

var illegalComponents = map[string]struct{}{
	".":  struct{}{},
	"..": struct{}{},
	"*":  struct{}{},
}

func Parse(s string) (string, error) {
	if s == string(defaultRole) {
		return s, nil
	}
	if strings.HasPrefix(s, "/") {
		return "", fmt.Errorf("role %q cannot start with a slash", s)
	}
	if strings.HasSuffix(s, "/") {
		return "", fmt.Errorf("role %q cannot end with a slash", s)
	}

	// validate each component in the role path
	for _, part := range strings.Split(s, "/") {
		if part == "" {
			return "", fmt.Errorf("role %q cannot contain two adjacent slashes", s)
		}
		if bad, found := illegalComponents[part]; found {
			return "", fmt.Errorf("role %q cannot contain %q as a component", s, bad)
		}
		if strings.HasPrefix(part, "-") {
			return "", fmt.Errorf("role component %q is invalid because it begins with a dash", part)
		}
		if strings.IndexFunc(part, func(r rune) bool { return unicode.IsSpace(r) || unicode.IsControl(r) }) > -1 {
			return "", fmt.Errorf("role component %q is invalid because it contains backspace or whitespace", part)
		}
	}
	return s, nil
}

func Validate(roles ...string) error {
	for i := range roles {
		_, err := Parse(roles[i])
		if err != nil {
			return err
		}
	}
	return nil
}
