package errors

import "fmt"

// ErrNotFound error type for objects not found
type ErrNotFound struct {
	// ID unique object identifier.
	ID   string
	// Type of the object which wasn't found
	Type string
}

func (e *ErrNotFound) Error() string {
	return fmt.Sprintf("%v with ID: %v not found", e.Type, e.ID)
}