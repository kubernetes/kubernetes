//go:build go1.20

package errs

import "errors"

// Is checks if any of the underlying errors matches target
func Is(err, target error) bool { return errors.Is(err, target) }
