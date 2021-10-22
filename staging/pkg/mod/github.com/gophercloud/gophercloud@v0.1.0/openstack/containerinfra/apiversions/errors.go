package apiversions

import (
	"fmt"
)

// ErrVersionNotFound is the error when the requested API version
// could not be found.
type ErrVersionNotFound struct{}

func (e ErrVersionNotFound) Error() string {
	return fmt.Sprintf("Unable to find requested API version")
}

// ErrMultipleVersionsFound is the error when a request for an API
// version returns multiple results.
type ErrMultipleVersionsFound struct {
	Count int
}

func (e ErrMultipleVersionsFound) Error() string {
	return fmt.Sprintf("Found %d API versions", e.Count)
}
