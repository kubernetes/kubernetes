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

package goser

import (
	"encoding/json"
	"fmt"
	base "github.com/pquerna/ffjson/tests/goser/base"
	ff "github.com/pquerna/ffjson/tests/goser/ff"
	"reflect"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	var record ff.Log
	var recordTripped ff.Log
	ff.NewLog(&record)

	buf1, err := json.Marshal(&record)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	err = json.Unmarshal(buf1, &recordTripped)
	if err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	good := reflect.DeepEqual(record, recordTripped)
	if !good {
		t.Fatalf("Expected: %v\n Got: %v", record, recordTripped)
	}
}

func BenchmarkMarshalJSON(b *testing.B) {
	var record base.Log
	base.NewLog(&record)

	buf, err := json.Marshal(&record)
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := json.Marshal(&record)
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
	}
}

func BenchmarkFFMarshalJSON(b *testing.B) {
	var record ff.Log
	ff.NewLog(&record)

	buf, err := record.MarshalJSON()
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := record.MarshalJSON()
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
	}
}

type fatalF interface {
	Fatalf(format string, args ...interface{})
}

func getBaseData(b fatalF) []byte {
	var record base.Log
	base.NewLog(&record)
	buf, err := json.MarshalIndent(&record, "", "    ")
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	return buf
}

func BenchmarkUnmarshalJSON(b *testing.B) {
	rec := base.Log{}
	buf := getBaseData(b)
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := json.Unmarshal(buf, &rec)
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
	}
}

func BenchmarkFFUnmarshalJSON(b *testing.B) {
	rec := ff.Log{}
	buf := getBaseData(b)
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := rec.UnmarshalJSON(buf)
		if err != nil {
			b.Fatalf("UnmarshalJSON: %v", err)
		}
	}
}

func TestUnmarshal(t *testing.T) {
	rec := ff.Log{}
	buf := getBaseData(t)

	err := rec.UnmarshalJSON(buf)
	if err != nil {
		t.Fatalf("Unmarshal: %v from %s", err, string(buf))
	}

	rec2 := base.Log{}
	json.Unmarshal(buf, &rec2)

	a := fmt.Sprintf("%v", rec)
	b := fmt.Sprintf("%v", rec2)
	if a != b {
		t.Fatalf("Expected: %v\n Got: %v\n from: %s", rec2, rec, string(buf))
	}
}
