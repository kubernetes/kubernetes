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
	"fmt"
	"reflect"
	"testing"
	"time"

	bq "google.golang.org/api/bigquery/v2"
)

func TestConvertBasicValues(t *testing.T) {
	schema := []*FieldSchema{
		{Type: StringFieldType},
		{Type: IntegerFieldType},
		{Type: FloatFieldType},
		{Type: BooleanFieldType},
	}
	row := &bq.TableRow{
		F: []*bq.TableCell{
			{V: "a"},
			{V: "1"},
			{V: "1.2"},
			{V: "true"},
		},
	}
	got, err := convertRow(row, schema)
	if err != nil {
		t.Fatalf("error converting: %v", err)
	}
	want := []Value{"a", 1, 1.2, true}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("converting basic values: got:\n%v\nwant:\n%v", got, want)
	}
}

func TestConvertTime(t *testing.T) {
	schema := []*FieldSchema{
		{Type: TimestampFieldType},
	}
	thyme := time.Date(1970, 1, 1, 10, 0, 0, 10, time.UTC)
	row := &bq.TableRow{
		F: []*bq.TableCell{
			{V: fmt.Sprintf("%.10f", float64(thyme.UnixNano())/1e9)},
		},
	}
	got, err := convertRow(row, schema)
	if err != nil {
		t.Fatalf("error converting: %v", err)
	}
	if !got[0].(time.Time).Equal(thyme) {
		t.Errorf("converting basic values: got:\n%v\nwant:\n%v", got, thyme)
	}
}

func TestConvertNullValues(t *testing.T) {
	schema := []*FieldSchema{
		{Type: StringFieldType},
	}
	row := &bq.TableRow{
		F: []*bq.TableCell{
			{V: nil},
		},
	}
	got, err := convertRow(row, schema)
	if err != nil {
		t.Fatalf("error converting: %v", err)
	}
	want := []Value{nil}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("converting null values: got:\n%v\nwant:\n%v", got, want)
	}
}

func TestBasicRepetition(t *testing.T) {
	schema := []*FieldSchema{
		{Type: IntegerFieldType, Repeated: true},
	}
	row := &bq.TableRow{
		F: []*bq.TableCell{
			{
				V: []interface{}{
					map[string]interface{}{
						"v": "1",
					},
					map[string]interface{}{
						"v": "2",
					},
					map[string]interface{}{
						"v": "3",
					},
				},
			},
		},
	}
	got, err := convertRow(row, schema)
	if err != nil {
		t.Fatalf("error converting: %v", err)
	}
	want := []Value{[]Value{1, 2, 3}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("converting basic repeated values: got:\n%v\nwant:\n%v", got, want)
	}
}

func TestNestedRecordContainingRepetition(t *testing.T) {
	schema := []*FieldSchema{
		{
			Type: RecordFieldType,
			Schema: Schema{
				{Type: IntegerFieldType, Repeated: true},
			},
		},
	}
	row := &bq.TableRow{
		F: []*bq.TableCell{
			{
				V: map[string]interface{}{
					"f": []interface{}{
						map[string]interface{}{
							"v": []interface{}{
								map[string]interface{}{"v": "1"},
								map[string]interface{}{"v": "2"},
								map[string]interface{}{"v": "3"},
							},
						},
					},
				},
			},
		},
	}

	got, err := convertRow(row, schema)
	if err != nil {
		t.Fatalf("error converting: %v", err)
	}
	want := []Value{[]Value{[]Value{1, 2, 3}}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("converting basic repeated values: got:\n%v\nwant:\n%v", got, want)
	}
}

func TestRepeatedRecordContainingRepetition(t *testing.T) {
	schema := []*FieldSchema{
		{
			Type:     RecordFieldType,
			Repeated: true,
			Schema: Schema{
				{Type: IntegerFieldType, Repeated: true},
			},
		},
	}
	row := &bq.TableRow{F: []*bq.TableCell{
		{
			V: []interface{}{ // repeated records.
				map[string]interface{}{ // first record.
					"v": map[string]interface{}{ // pointless single-key-map wrapper.
						"f": []interface{}{ // list of record fields.
							map[string]interface{}{ // only record (repeated ints)
								"v": []interface{}{ // pointless wrapper.
									map[string]interface{}{
										"v": "1",
									},
									map[string]interface{}{
										"v": "2",
									},
									map[string]interface{}{
										"v": "3",
									},
								},
							},
						},
					},
				},
				map[string]interface{}{ // second record.
					"v": map[string]interface{}{
						"f": []interface{}{
							map[string]interface{}{
								"v": []interface{}{
									map[string]interface{}{
										"v": "4",
									},
									map[string]interface{}{
										"v": "5",
									},
									map[string]interface{}{
										"v": "6",
									},
								},
							},
						},
					},
				},
			},
		},
	}}

	got, err := convertRow(row, schema)
	if err != nil {
		t.Fatalf("error converting: %v", err)
	}
	want := []Value{ // the row is a list of length 1, containing an entry for the repeated record.
		[]Value{ // the repeated record is a list of length 2, containing an entry for each repetition.
			[]Value{ // the record is a list of length 1, containing an entry for the repeated integer field.
				[]Value{1, 2, 3}, // the repeated integer field is a list of length 3.
			},
			[]Value{ // second record
				[]Value{4, 5, 6},
			},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("converting repeated records with repeated values: got:\n%v\nwant:\n%v", got, want)
	}
}

func TestRepeatedRecordContainingRecord(t *testing.T) {
	schema := []*FieldSchema{
		{
			Type:     RecordFieldType,
			Repeated: true,
			Schema: Schema{
				{
					Type: StringFieldType,
				},
				{
					Type: RecordFieldType,
					Schema: Schema{
						{Type: IntegerFieldType},
						{Type: StringFieldType},
					},
				},
			},
		},
	}
	row := &bq.TableRow{F: []*bq.TableCell{
		{
			V: []interface{}{ // repeated records.
				map[string]interface{}{ // first record.
					"v": map[string]interface{}{ // pointless single-key-map wrapper.
						"f": []interface{}{ // list of record fields.
							map[string]interface{}{ // first record field (name)
								"v": "first repeated record",
							},
							map[string]interface{}{ // second record field (nested record).
								"v": map[string]interface{}{ // pointless single-key-map wrapper.
									"f": []interface{}{ // nested record fields
										map[string]interface{}{
											"v": "1",
										},
										map[string]interface{}{
											"v": "two",
										},
									},
								},
							},
						},
					},
				},
				map[string]interface{}{ // second record.
					"v": map[string]interface{}{
						"f": []interface{}{
							map[string]interface{}{
								"v": "second repeated record",
							},
							map[string]interface{}{
								"v": map[string]interface{}{
									"f": []interface{}{
										map[string]interface{}{
											"v": "3",
										},
										map[string]interface{}{
											"v": "four",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}}

	got, err := convertRow(row, schema)
	if err != nil {
		t.Fatalf("error converting: %v", err)
	}
	// TODO: test with flattenresults.
	want := []Value{ // the row is a list of length 1, containing an entry for the repeated record.
		[]Value{ // the repeated record is a list of length 2, containing an entry for each repetition.
			[]Value{ // record contains a string followed by a nested record.
				"first repeated record",
				[]Value{
					1,
					"two",
				},
			},
			[]Value{ // second record.
				"second repeated record",
				[]Value{
					3,
					"four",
				},
			},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("converting repeated records containing record : got:\n%v\nwant:\n%v", got, want)
	}
}
