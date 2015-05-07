package types

import (
	"encoding/json"
	"fmt"
)

var (
	ErrNoACKind = ACKindError("ACKind must be set")
)

// ACKind wraps a string to define a field which must be set with one of
// several ACKind values. If it is unset, or has an invalid value, the field
// will refuse to marshal/unmarshal.
type ACKind string

func (a ACKind) String() string {
	return string(a)
}

func (a ACKind) assertValid() error {
	s := a.String()
	switch s {
	case "ImageManifest", "PodManifest":
		return nil
	case "":
		return ErrNoACKind
	default:
		msg := fmt.Sprintf("bad ACKind: %s", s)
		return ACKindError(msg)
	}
}

func (a ACKind) MarshalJSON() ([]byte, error) {
	if err := a.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(a.String())
}

func (a *ACKind) UnmarshalJSON(data []byte) error {
	var s string
	err := json.Unmarshal(data, &s)
	if err != nil {
		return err
	}
	na := ACKind(s)
	if err := na.assertValid(); err != nil {
		return err
	}
	*a = na
	return nil
}
