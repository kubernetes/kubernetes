package store

import (
	"strings"

	"github.com/pkg/errors"
)

var (
	// errVolumeInUse is a typed error returned when trying to remove a volume that is currently in use by a container
	errVolumeInUse = errors.New("volume is in use")
	// errNoSuchVolume is a typed error returned if the requested volume doesn't exist in the volume store
	errNoSuchVolume = errors.New("no such volume")
	// errInvalidName is a typed error returned when creating a volume with a name that is not valid on the platform
	errInvalidName = errors.New("volume name is not valid on this platform")
	// errNameConflict is a typed error returned on create when a volume exists with the given name, but for a different driver
	errNameConflict = errors.New("volume name must be unique")
)

// OpErr is the error type returned by functions in the store package. It describes
// the operation, volume name, and error.
type OpErr struct {
	// Err is the error that occurred during the operation.
	Err error
	// Op is the operation which caused the error, such as "create", or "list".
	Op string
	// Name is the name of the resource being requested for this op, typically the volume name or the driver name.
	Name string
	// Refs is the list of references associated with the resource.
	Refs []string
}

// Error satisfies the built-in error interface type.
func (e *OpErr) Error() string {
	if e == nil {
		return "<nil>"
	}
	s := e.Op
	if e.Name != "" {
		s = s + " " + e.Name
	}

	s = s + ": " + e.Err.Error()
	if len(e.Refs) > 0 {
		s = s + " - " + "[" + strings.Join(e.Refs, ", ") + "]"
	}
	return s
}

// IsInUse returns a boolean indicating whether the error indicates that a
// volume is in use
func IsInUse(err error) bool {
	return isErr(err, errVolumeInUse)
}

// IsNotExist returns a boolean indicating whether the error indicates that the volume does not exist
func IsNotExist(err error) bool {
	return isErr(err, errNoSuchVolume)
}

// IsNameConflict returns a boolean indicating whether the error indicates that a
// volume name is already taken
func IsNameConflict(err error) bool {
	return isErr(err, errNameConflict)
}

func isErr(err error, expected error) bool {
	err = errors.Cause(err)
	switch pe := err.(type) {
	case nil:
		return false
	case *OpErr:
		err = errors.Cause(pe.Err)
	}
	return err == expected
}
