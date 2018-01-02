package tsdb

import (
	"encoding/binary"
	"strings"

	"github.com/influxdata/influxdb/cmd/influx_tsm/tsdb/internal"
	"github.com/influxdata/influxdb/influxql"

	"github.com/gogo/protobuf/proto"
)

// Field represents an encoded field.
type Field struct {
	ID   uint8             `json:"id,omitempty"`
	Name string            `json:"name,omitempty"`
	Type influxql.DataType `json:"type,omitempty"`
}

// MeasurementFields is a mapping from measurements to its fields.
type MeasurementFields struct {
	Fields map[string]*Field `json:"fields"`
	Codec  *FieldCodec
}

// UnmarshalBinary decodes the object from a binary format.
func (m *MeasurementFields) UnmarshalBinary(buf []byte) error {
	var pb internal.MeasurementFields
	if err := proto.Unmarshal(buf, &pb); err != nil {
		return err
	}
	m.Fields = make(map[string]*Field)
	for _, f := range pb.Fields {
		m.Fields[f.GetName()] = &Field{ID: uint8(f.GetID()), Name: f.GetName(), Type: influxql.DataType(f.GetType())}
	}
	return nil
}

// Series represents a series in the shard.
type Series struct {
	Key  string
	Tags map[string]string
}

// MeasurementFromSeriesKey returns the Measurement name for a given series.
func MeasurementFromSeriesKey(key string) string {
	return strings.SplitN(key, ",", 2)[0]
}

// DecodeKeyValue decodes the key and value from bytes.
func DecodeKeyValue(field string, dec *FieldCodec, k, v []byte) (int64, interface{}) {
	// Convert key to a timestamp.
	key := int64(binary.BigEndian.Uint64(k[0:8]))

	decValue, err := dec.DecodeByName(field, v)
	if err != nil {
		return key, nil
	}
	return key, decValue
}
