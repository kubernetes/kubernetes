// Package identifiers provides common validation for identifiers and keys
// across containerd.
//
// Identifiers in containerd must be a alphanumeric, allowing limited
// underscores, dashes and dots.
//
// While the character set may be expanded in the future, identifiers
// are guaranteed to be safely used as filesystem path components.
package identifiers

import (
	"regexp"

	"github.com/containerd/containerd/errdefs"
	"github.com/pkg/errors"
)

const (
	maxLength  = 76
	alphanum   = `[A-Za-z0-9]+`
	separators = `[._-]`
)

var (
	// identifierRe defines the pattern for valid identifiers.
	identifierRe = regexp.MustCompile(reAnchor(alphanum + reGroup(separators+reGroup(alphanum)) + "*"))
)

// Validate return nil if the string s is a valid identifier.
//
// identifiers must be valid domain names according to RFC 1035, section 2.3.1.  To
// enforce case insensitvity, all characters must be lower case.
//
// In general, identifiers that pass this validation, should be safe for use as
// a domain names or filesystem path component.
func Validate(s string) error {
	if len(s) == 0 {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "identifier must not be empty")
	}

	if len(s) > maxLength {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "identifier %q greater than maximum length (%d characters)", s, maxLength)
	}

	if !identifierRe.MatchString(s) {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "identifier %q must match %v", s, identifierRe)
	}
	return nil
}

func reGroup(s string) string {
	return `(?:` + s + `)`
}

func reAnchor(s string) string {
	return `^` + s + `$`
}
