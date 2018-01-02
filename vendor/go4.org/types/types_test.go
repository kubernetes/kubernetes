/*
Copyright 2013 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package types

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestTime3339(t *testing.T) {
	tm := time.Unix(123, 456)
	t3 := Time3339(tm)
	type O struct {
		SomeTime Time3339 `json:"someTime"`
	}
	o := &O{SomeTime: t3}
	got, err := json.Marshal(o)
	if err != nil {
		t.Fatal(err)
	}
	goodEnc := "{\"someTime\":\"1970-01-01T00:02:03.000000456Z\"}"
	if string(got) != goodEnc {
		t.Errorf("Encoding wrong.\n Got: %q\nWant: %q", got, goodEnc)
	}
	ogot := &O{}
	err = json.Unmarshal([]byte(goodEnc), ogot)
	if err != nil {
		t.Fatal(err)
	}
	if !tm.Equal(ogot.SomeTime.Time()) {
		t.Errorf("Unmarshal got time %v; want %v", ogot.SomeTime.Time(), tm)
	}
}

func TestTime3339_Marshal(t *testing.T) {
	tests := []struct {
		in   time.Time
		want string
	}{
		{time.Time{}, "null"},
		{time.Unix(1, 0), `"1970-01-01T00:00:01Z"`},
	}
	for i, tt := range tests {
		got, err := Time3339(tt.in).MarshalJSON()
		if err != nil {
			t.Errorf("%d. marshal(%v) got error: %v", i, tt.in, err)
			continue
		}
		if string(got) != tt.want {
			t.Errorf("%d. marshal(%v) = %q; want %q", i, tt.in, got, tt.want)
		}
	}
}

func TestTime3339_empty(t *testing.T) {
	tests := []struct {
		enc string
		z   bool
	}{
		{enc: "null", z: true},
		{enc: `""`, z: true},
		{enc: "0000-00-00T00:00:00Z", z: true},
		{enc: "0001-01-01T00:00:00Z", z: true},
		{enc: "1970-01-01T00:00:00Z", z: true},
		{enc: "2001-02-03T04:05:06Z", z: false},
		{enc: "2001-02-03T04:05:06+06:00", z: false},
		{enc: "2001-02-03T04:05:06-06:00", z: false},
		{enc: "2001-02-03T04:05:06.123456789Z", z: false},
		{enc: "2001-02-03T04:05:06.123456789+06:00", z: false},
		{enc: "2001-02-03T04:05:06.123456789-06:00", z: false},
	}
	for _, tt := range tests {
		var tm Time3339
		enc := tt.enc
		if strings.Contains(enc, "T") {
			enc = "\"" + enc + "\""
		}
		err := json.Unmarshal([]byte(enc), &tm)
		if err != nil {
			t.Errorf("unmarshal %q = %v", enc, err)
		}
		if tm.IsAnyZero() != tt.z {
			t.Errorf("unmarshal %q = %v (%d), %v; zero=%v; want %v", tt.enc, tm.Time(), tm.Time().Unix(), err,
				!tt.z, tt.z)
		}
	}
}
