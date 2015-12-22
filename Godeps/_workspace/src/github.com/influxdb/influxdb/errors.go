package influxdb

import (
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrFieldsRequired is returned when a point does not any fields.
	ErrFieldsRequired = errors.New("fields required")

	// ErrFieldTypeConflict is returned when a new field already exists with a different type.
	ErrFieldTypeConflict = errors.New("field type conflict")
)

// ErrDatabaseNotFound indicates that a database operation failed on the
// specified database because the specified database does not exist.
func ErrDatabaseNotFound(name string) error { return fmt.Errorf("database not found: %s", name) }

// ErrRetentionPolicyNotFound indicates that the named retention policy could
// not be found in the database.
func ErrRetentionPolicyNotFound(name string) error {
	return fmt.Errorf("retention policy not found: %s", name)
}

// IsClientError indicates whether an error is a known client error.
func IsClientError(err error) bool {
	if err == nil {
		return false
	}

	if err == ErrFieldsRequired {
		return true
	}
	if err == ErrFieldTypeConflict {
		return true
	}

	if strings.Contains(err.Error(), ErrFieldTypeConflict.Error()) {
		return true
	}

	return false
}
