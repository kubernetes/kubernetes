package tsdb

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

const (
	fieldFloat   = 1
	fieldInteger = 2
	fieldBoolean = 3
	fieldString  = 4
)

var (
	// ErrFieldNotFound is returned when a field cannot be found.
	ErrFieldNotFound = errors.New("field not found")

	// ErrFieldUnmappedID is returned when the system is presented, during decode, with a field ID
	// there is no mapping for.
	ErrFieldUnmappedID = errors.New("field ID not mapped")
)

// FieldCodec provides encoding and decoding functionality for the fields of a given
// Measurement.
type FieldCodec struct {
	fieldsByID   map[uint8]*Field
	fieldsByName map[string]*Field
}

// NewFieldCodec returns a FieldCodec for the given Measurement. Must be called with
// a RLock that protects the Measurement.
func NewFieldCodec(fields map[string]*Field) *FieldCodec {
	fieldsByID := make(map[uint8]*Field, len(fields))
	fieldsByName := make(map[string]*Field, len(fields))
	for _, f := range fields {
		fieldsByID[f.ID] = f
		fieldsByName[f.Name] = f
	}
	return &FieldCodec{fieldsByID: fieldsByID, fieldsByName: fieldsByName}
}

// FieldIDByName returns the ID for the given field.
func (f *FieldCodec) FieldIDByName(s string) (uint8, error) {
	fi := f.fieldsByName[s]
	if fi == nil {
		return 0, ErrFieldNotFound
	}
	return fi.ID, nil
}

// DecodeByID scans a byte slice for a field with the given ID, converts it to its
// expected type, and return that value.
func (f *FieldCodec) DecodeByID(targetID uint8, b []byte) (interface{}, error) {
	var value interface{}
	for {
		if len(b) == 0 {
			// No more bytes.
			return nil, ErrFieldNotFound
		}

		field := f.fieldsByID[b[0]]
		if field == nil {
			// This can happen, though is very unlikely. If this node receives encoded data, to be written
			// to disk, and is queried for that data before its metastore is updated, there will be no field
			// mapping for the data during decode. All this can happen because data is encoded by the node
			// that first received the write request, not the node that actually writes the data to disk.
			// So if this happens, the read must be aborted.
			return nil, ErrFieldUnmappedID
		}

		switch field.Type {
		case fieldFloat:
			if field.ID == targetID {
				value = math.Float64frombits(binary.BigEndian.Uint64(b[1:9]))
			}
			b = b[9:]
		case fieldInteger:
			if field.ID == targetID {
				value = int64(binary.BigEndian.Uint64(b[1:9]))
			}
			b = b[9:]
		case fieldBoolean:
			if field.ID == targetID {
				value = b[1] == 1
			}
			b = b[2:]
		case fieldString:
			length := binary.BigEndian.Uint16(b[1:3])
			if field.ID == targetID {
				value = string(b[3 : 3+length])
			}
			b = b[3+length:]
		default:
			panic(fmt.Sprintf("unsupported value type during decode by id: %T", field.Type))
		}

		if value != nil {
			return value, nil
		}
	}
}

// DecodeByName scans a byte slice for a field with the given name, converts it to its
// expected type, and return that value.
func (f *FieldCodec) DecodeByName(name string, b []byte) (interface{}, error) {
	fi := f.FieldByName(name)
	if fi == nil {
		return 0, ErrFieldNotFound
	}
	return f.DecodeByID(fi.ID, b)
}

// FieldByName returns the field by its name. It will return a nil if not found
func (f *FieldCodec) FieldByName(name string) *Field {
	return f.fieldsByName[name]
}
