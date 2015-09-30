/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package types

import (
	"encoding/json"
	ff "github.com/pquerna/ffjson/tests/types/ff"
	"reflect"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	var record ff.Everything
	var recordTripped ff.Everything
	ff.NewEverything(&record)

	buf1, err := json.Marshal(&record)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	recordTripped.MySweetInterface = &ff.Cats{}
	err = json.Unmarshal(buf1, &recordTripped)
	if err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	good := reflect.DeepEqual(record.FooStruct, recordTripped.FooStruct)
	if !good {
		t.Fatalf("Expected: %v\n Got: %v", *record.FooStruct, *recordTripped.FooStruct)
	}

	record.FooStruct = nil
	recordTripped.FooStruct = nil

	good = reflect.DeepEqual(record, recordTripped)
	if !good {
		t.Fatalf("Expected: %v\n Got: %v", record, recordTripped)
	}

	if recordTripped.SuperBool != true {
		t.Fatal("Embeded struct didn't Unmarshal")
	}

	if recordTripped.Something != 99 {
		t.Fatal("Embeded nonexported-struct didn't Unmarshal")
	}
}

func TestUnmarshalEmpty(t *testing.T) {
	record := ff.Everything{}
	err := record.UnmarshalJSON([]byte(`{}`))
	if err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
}

const (
	everythingJson = `{
  "Bool": true,
  "Int": 1,
  "Int8": 2,
  "Int16": 3,
  "Int32": -4,
  "Int64": 57,
  "Uint": 100,
  "Uint8": 101,
  "Uint16": 102,
  "Uint32": 0,
  "Uint64": 103,
  "Uintptr": 104,
  "Float32": 3.14,
  "Float64": 3.15,
  "Array": [
    1,
    2,
    3
  ],
  "Map": {
    "bar": 2,
    "foo": 1
  },
  "String": "snowman☃\uD801\uDC37",
  "StringPointer": null,
  "Int64Pointer": null,
  "FooStruct": {
    "Bar": 1
  },
  "Something": 99
}`
)

func TestUnmarshalFull(t *testing.T) {
	record := ff.Everything{}
	// TODO(pquerna): add unicode snowman
	// TODO(pquerna): handle Bar subtype
	err := record.UnmarshalJSON([]byte(everythingJson))
	if err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}

	expect := "snowman☃𐐷"
	if record.String != expect {
		t.Fatalf("record.String decoding problem, expected: %v got: %v", expect, record.String)
	}

	if record.Something != 99 {
		t.Fatalf("record.Something decoding problem, expected: 99 got: %v", record.Something)
	}
}

func TestUnmarshalNullPointer(t *testing.T) {
	record := ff.Everything{}
	err := record.UnmarshalJSON([]byte(`{"FooStruct": null,"Something":99}`))
	if err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if record.FooStruct != nil {
		t.Fatalf("record.Something decoding problem, expected: nil got: %v", record.FooStruct)
	}
}
