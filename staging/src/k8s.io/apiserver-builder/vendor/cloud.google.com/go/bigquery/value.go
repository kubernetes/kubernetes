// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bigquery

import (
	"errors"
	"fmt"
	"strconv"
	"time"

	bq "google.golang.org/api/bigquery/v2"
)

// Value stores the contents of a single cell from a BigQuery result.
type Value interface{}

// ValueLoader stores a slice of Values representing a result row from a Read operation.
// See Iterator.Get for more information.
type ValueLoader interface {
	Load(v []Value) error
}

// ValueList converts a []Value to implement ValueLoader.
type ValueList []Value

// Load stores a sequence of values in a ValueList.
func (vs *ValueList) Load(v []Value) error {
	*vs = append(*vs, v...)
	return nil
}

// A ValueSaver returns a row of data to be inserted into a table.
type ValueSaver interface {
	// Save returns a row to be inserted into a BigQuery table, represented
	// as a map from field name to Value.
	// If insertID is non-empty, BigQuery will use it to de-duplicate
	// insertions of this row on a best-effort basis.
	Save() (row map[string]Value, insertID string, err error)
}

// ValuesSaver implements ValueSaver for a slice of Values.
type ValuesSaver struct {
	Schema Schema

	// If non-empty, BigQuery will use InsertID to de-duplicate insertions
	// of this row on a best-effort basis.
	InsertID string

	Row []Value
}

// Save implements ValueSaver
func (vls *ValuesSaver) Save() (map[string]Value, string, error) {
	m, err := valuesToMap(vls.Row, vls.Schema)
	return m, vls.InsertID, err
}

func valuesToMap(vs []Value, schema Schema) (map[string]Value, error) {
	if len(vs) != len(schema) {
		return nil, errors.New("Schema does not match length of row to be inserted")
	}

	m := make(map[string]Value)
	for i, fieldSchema := range schema {
		if fieldSchema.Type == RecordFieldType {
			nested, ok := vs[i].([]Value)
			if !ok {
				return nil, errors.New("Nested record is not a []Value")
			}
			value, err := valuesToMap(nested, fieldSchema.Schema)
			if err != nil {
				return nil, err
			}
			m[fieldSchema.Name] = value
		} else {
			m[fieldSchema.Name] = vs[i]
		}
	}
	return m, nil
}

// convertRows converts a series of TableRows into a series of Value slices.
// schema is used to interpret the data from rows; its length must match the
// length of each row.
func convertRows(rows []*bq.TableRow, schema Schema) ([][]Value, error) {
	var rs [][]Value
	for _, r := range rows {
		row, err := convertRow(r, schema)
		if err != nil {
			return nil, err
		}
		rs = append(rs, row)
	}
	return rs, nil
}

func convertRow(r *bq.TableRow, schema Schema) ([]Value, error) {
	if len(schema) != len(r.F) {
		return nil, errors.New("schema length does not match row length")
	}
	var values []Value
	for i, cell := range r.F {
		fs := schema[i]
		v, err := convertValue(cell.V, fs.Type, fs.Schema)
		if err != nil {
			return nil, err
		}
		values = append(values, v)
	}
	return values, nil
}

func convertValue(val interface{}, typ FieldType, schema Schema) (Value, error) {
	switch val := val.(type) {
	case nil:
		return nil, nil
	case []interface{}:
		return convertRepeatedRecord(val, typ, schema)
	case map[string]interface{}:
		return convertNestedRecord(val, schema)
	case string:
		return convertBasicType(val, typ)
	default:
		return nil, fmt.Errorf("got value %v; expected a value of type %s", val, typ)
	}
}

func convertRepeatedRecord(vals []interface{}, typ FieldType, schema Schema) (Value, error) {
	var values []Value
	for _, cell := range vals {
		// each cell contains a single entry, keyed by "v"
		val := cell.(map[string]interface{})["v"]
		v, err := convertValue(val, typ, schema)
		if err != nil {
			return nil, err
		}
		values = append(values, v)
	}
	return values, nil
}

func convertNestedRecord(val map[string]interface{}, schema Schema) (Value, error) {
	// convertNestedRecord is similar to convertRow, as a record has the same structure as a row.

	// Nested records are wrapped in a map with a single key, "f".
	record := val["f"].([]interface{})
	if len(record) != len(schema) {
		return nil, errors.New("schema length does not match record length")
	}

	var values []Value
	for i, cell := range record {
		// each cell contains a single entry, keyed by "v"
		val := cell.(map[string]interface{})["v"]

		fs := schema[i]
		v, err := convertValue(val, fs.Type, fs.Schema)
		if err != nil {
			return nil, err
		}
		values = append(values, v)
	}
	return values, nil
}

// convertBasicType returns val as an interface with a concrete type specified by typ.
func convertBasicType(val string, typ FieldType) (Value, error) {
	switch typ {
	case StringFieldType:
		return val, nil
	case IntegerFieldType:
		return strconv.Atoi(val)
	case FloatFieldType:
		return strconv.ParseFloat(val, 64)
	case BooleanFieldType:
		return strconv.ParseBool(val)
	case TimestampFieldType:
		f, err := strconv.ParseFloat(val, 64)
		return Value(time.Unix(0, int64(f*1e9))), err
	default:
		return nil, errors.New("unrecognized type")
	}
}
