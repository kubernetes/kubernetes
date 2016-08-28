package types

import (
	"bytes"
	b64 "encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"

	"github.com/akutz/goof"
)

// InstanceIDMap is a map of InstanceID objects.
type InstanceIDMap map[string]*InstanceID

// InstanceID identifies a host to a remote storage platform.
type InstanceID struct {

	// ID is the simple part of the InstanceID.
	ID string `json:"id" yaml:"id"`

	// Driver is the name of the StorageExecutor that created the InstanceID
	// as well as the name of the StorageDriver for which the InstanceID is
	// valid.
	Driver string `json:"driver"`

	metadata json.RawMessage
}

var (
	// ErrIIDMetadataNil is returned by *InstanceID.UnmarshalMetadata when
	// the InstanceID's metadata is empty or nil.
	ErrIIDMetadataNil = goof.New("cannot unmarshal nil metadata")

	// ErrIIDMetadataNilData is returned by *InstanceID.MarshalMetadata when
	// the provided object to marshal is nil.
	ErrIIDMetadataNilData = goof.New("cannot marshal nil metadata")

	// ErrIIDMetadataNilDest is returned by *InstanceID.UnmarshalMetadata when
	// the provided destination into which the metadata should be unmarshaled
	// is nil.
	ErrIIDMetadataNilDest = goof.New("cannot unmarshal into nil receiver")
)

// String returns the string representation of an InstanceID object.
func (i *InstanceID) String() string {
	buf, err := i.MarshalText()
	if err != nil {
		panic(err)
	}
	return string(buf)
}

// DeleteMetadata deletes the metadata from the InstanceID.
func (i *InstanceID) DeleteMetadata() {
	i.metadata = json.RawMessage{}
}

// MarshalMetadata encodes the provided object to JSON and assigns the result
// to the InstanceID's metadata field.
func (i *InstanceID) MarshalMetadata(data interface{}) error {

	if fmt.Sprintf("%v", data) == "<nil>" {
		return ErrIIDMetadataNilData
	}

	var err error
	if i.metadata, err = json.Marshal(data); err != nil {
		return err
	}
	return nil
}

// UnmarshalMetadata decodes the InstanceID's metadata into the provided object.
func (i *InstanceID) UnmarshalMetadata(dest interface{}) error {

	if fmt.Sprintf("%v", dest) == "<nil>" {
		return ErrIIDMetadataNilDest
	}

	if len(i.metadata) == 0 {
		return ErrIIDMetadataNil
	}
	if err := json.Unmarshal(i.metadata, dest); err != nil {
		return err
	}
	return nil
}

// MarshalText marshals InstanceID to a text string that adheres to the format
// `DRIVER=ID[,METADATA]`. If metadata is present it is encoded as a base64
// string.
func (i *InstanceID) MarshalText() ([]byte, error) {

	t := &bytes.Buffer{}
	fmt.Fprintf(t, "%s=%s", i.Driver, i.ID)

	if len(i.metadata) == 0 {
		return t.Bytes(), nil
	}

	t.WriteByte(',')

	enc := b64.NewEncoder(b64.StdEncoding, t)
	if _, err := enc.Write(i.metadata); err != nil {
		return nil, err
	}
	if err := enc.Close(); err != nil {
		return nil, err
	}
	return t.Bytes(), nil
}

var iidRX = regexp.MustCompile(
	`(?i)^([^=]+)=([^,]*)(?:,(.+))?$`)

// UnmarshalText unmarshals the data into a an InstanceID provided the data
// adheres to the format described in the MarshalText function.
func (i *InstanceID) UnmarshalText(value []byte) error {

	m := iidRX.FindSubmatch(value)
	lm := len(m)

	if lm < 3 {
		return goof.WithField("value", string(value), "invalid InstanceID")
	}

	i.Driver = string(m[1])
	i.ID = string(m[2])

	if lm > 3 && len(m[3]) > 0 {
		enc := m[3]
		dec := b64.NewDecoder(b64.StdEncoding, bytes.NewReader(enc))
		i.metadata = make([]byte, b64.StdEncoding.DecodedLen(len(enc)))
		n, err := dec.Read(i.metadata)
		if err != nil {
			return err
		}
		i.metadata = i.metadata[:n]
	}

	return nil
}

// MarshalJSON marshals the InstanceID to JSON.
func (i *InstanceID) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		ID       string          `json:"id"`
		Driver   string          `json:"driver"`
		Metadata json.RawMessage `json:"metadata,omitempty"`
	}{i.ID, i.Driver, i.metadata})
}

// UnmarshalJSON marshals the InstanceID to JSON.
func (i *InstanceID) UnmarshalJSON(data []byte) error {

	iid := &struct {
		ID       string          `json:"id"`
		Driver   string          `json:"driver"`
		Metadata json.RawMessage `json:"metadata,omitempty"`
	}{}

	if err := json.Unmarshal(data, iid); err != nil {
		return err
	}

	i.ID = iid.ID
	i.Driver = iid.Driver
	i.metadata = iid.Metadata

	return nil
}

// MarshalYAML returns the object to marshal to the YAML representation of the
// InstanceID.
func (i *InstanceID) MarshalYAML() (interface{}, error) {

	var metadata map[string]interface{}
	if len(i.metadata) > 0 {
		if err := json.Unmarshal(i.metadata, &metadata); err != nil {
			return nil, err
		}
	}

	return &struct {
		ID       string                 `yaml:"id"`
		Driver   string                 `yaml:"driver"`
		Metadata map[string]interface{} `yaml:"metadata,omitempty"`
	}{i.ID, i.Driver, metadata}, nil
}
