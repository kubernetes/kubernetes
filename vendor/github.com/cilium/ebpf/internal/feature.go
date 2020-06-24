package internal

import (
	"fmt"
	"sync"

	"golang.org/x/xerrors"
)

// ErrNotSupported indicates that a feature is not supported by the current kernel.
var ErrNotSupported = xerrors.New("not supported")

// UnsupportedFeatureError is returned by FeatureTest() functions.
type UnsupportedFeatureError struct {
	// The minimum Linux mainline version required for this feature.
	// Used for the error string, and for sanity checking during testing.
	MinimumVersion Version

	// The name of the feature that isn't supported.
	Name string
}

func (ufe *UnsupportedFeatureError) Error() string {
	return fmt.Sprintf("%s not supported (requires >= %s)", ufe.Name, ufe.MinimumVersion)
}

// Is indicates that UnsupportedFeatureError is ErrNotSupported.
func (ufe *UnsupportedFeatureError) Is(target error) bool {
	return target == ErrNotSupported
}

// FeatureTest wraps a function so that it is run at most once.
//
// name should identify the tested feature, while version must be in the
// form Major.Minor[.Patch].
//
// Returns a descriptive UnsupportedFeatureError if the feature is not available.
func FeatureTest(name, version string, fn func() bool) func() error {
	v, err := NewVersion(version)
	if err != nil {
		return func() error { return err }
	}

	var (
		once   sync.Once
		result error
	)

	return func() error {
		once.Do(func() {
			if !fn() {
				result = &UnsupportedFeatureError{
					MinimumVersion: v,
					Name:           name,
				}
			}
		})
		return result
	}
}

// A Version in the form Major.Minor.Patch.
type Version [3]uint16

// NewVersion creates a version from a string like "Major.Minor.Patch".
//
// Patch is optional.
func NewVersion(ver string) (Version, error) {
	var major, minor, patch uint16
	n, _ := fmt.Sscanf(ver, "%d.%d.%d", &major, &minor, &patch)
	if n < 2 {
		return Version{}, xerrors.Errorf("invalid version: %s", ver)
	}
	return Version{major, minor, patch}, nil
}

func (v Version) String() string {
	if v[2] == 0 {
		return fmt.Sprintf("v%d.%d", v[0], v[1])
	}
	return fmt.Sprintf("v%d.%d.%d", v[0], v[1], v[2])
}

// Less returns true if the version is less than another version.
func (v Version) Less(other Version) bool {
	for i, a := range v {
		if a == other[i] {
			continue
		}
		return a < other[i]
	}
	return false
}
