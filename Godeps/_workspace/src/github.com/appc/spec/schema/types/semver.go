package types

import (
	"encoding/json"

	"github.com/coreos/go-semver/semver"
)

var (
	ErrNoZeroSemVer = ACVersionError("SemVer cannot be zero")
	ErrBadSemVer    = ACVersionError("SemVer is bad")
)

// SemVer implements the Unmarshaler interface to define a field that must be
// a semantic version string
// TODO(jonboulle): extend upstream instead of wrapping?
type SemVer semver.Version

// NewSemVer generates a new SemVer from a string. If the given string does
// not represent a valid SemVer, nil and an error are returned.
func NewSemVer(s string) (*SemVer, error) {
	nsv, err := semver.NewVersion(s)
	if err != nil {
		return nil, ErrBadSemVer
	}
	v := SemVer(*nsv)
	if v.Empty() {
		return nil, ErrNoZeroSemVer
	}
	return &v, nil
}

func (sv SemVer) String() string {
	s := semver.Version(sv)
	return s.String()
}

func (sv SemVer) Empty() bool {
	return semver.Version(sv) == semver.Version{}
}

// UnmarshalJSON implements the json.Unmarshaler interface
func (sv *SemVer) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	v, err := NewSemVer(s)
	if err != nil {
		return err
	}
	*sv = *v
	return nil
}

// MarshalJSON implements the json.Marshaler interface
func (sv SemVer) MarshalJSON() ([]byte, error) {
	if sv.Empty() {
		return nil, ErrNoZeroSemVer
	}
	return json.Marshal(sv.String())
}
