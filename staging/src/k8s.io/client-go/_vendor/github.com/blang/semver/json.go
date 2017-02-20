package semver

import (
	"encoding/json"
)

// MarshalJSON implements the encoding/json.Marshaler interface.
func (v Version) MarshalJSON() ([]byte, error) {
	return json.Marshal(v.String())
}

// UnmarshalJSON implements the encoding/json.Unmarshaler interface.
func (v *Version) UnmarshalJSON(data []byte) (err error) {
	var versionString string

	if err = json.Unmarshal(data, &versionString); err != nil {
		return
	}

	*v, err = Parse(versionString)

	return
}
