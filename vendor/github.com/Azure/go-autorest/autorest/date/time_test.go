package date

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
	"time"
)

func ExampleParseTime() {
	d, _ := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	fmt.Println(d)
	// Output: 2001-02-03 04:05:06 +0000 UTC
}

func ExampleTime_MarshalBinary() {
	ti, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	if err != nil {
		fmt.Println(err)
	}
	d := Time{ti}
	t, err := d.MarshalBinary()
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(t))
	// Output: 2001-02-03T04:05:06Z
}

func ExampleTime_UnmarshalBinary() {
	d := Time{}
	t := "2001-02-03T04:05:06Z"

	if err := d.UnmarshalBinary([]byte(t)); err != nil {
		fmt.Println(err)
	}
	fmt.Println(d)
	// Output: 2001-02-03T04:05:06Z
}

func ExampleTime_MarshalJSON() {
	d, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	if err != nil {
		fmt.Println(err)
	}
	j, err := json.Marshal(d)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(j))
	// Output: "2001-02-03T04:05:06Z"
}

func ExampleTime_UnmarshalJSON() {
	var d struct {
		Time Time `json:"datetime"`
	}
	j := `{"datetime" : "2001-02-03T04:05:06Z"}`

	if err := json.Unmarshal([]byte(j), &d); err != nil {
		fmt.Println(err)
	}
	fmt.Println(d.Time)
	// Output: 2001-02-03T04:05:06Z
}

func ExampleTime_MarshalText() {
	d, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	if err != nil {
		fmt.Println(err)
	}
	t, err := d.MarshalText()
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(t))
	// Output: 2001-02-03T04:05:06Z
}

func ExampleTime_UnmarshalText() {
	d := Time{}
	t := "2001-02-03T04:05:06Z"

	if err := d.UnmarshalText([]byte(t)); err != nil {
		fmt.Println(err)
	}
	fmt.Println(d)
	// Output: 2001-02-03T04:05:06Z
}

func TestUnmarshalTextforInvalidDate(t *testing.T) {
	d := Time{}
	dt := "2001-02-03T04:05:06AAA"

	if err := d.UnmarshalText([]byte(dt)); err == nil {
		t.Fatalf("date: Time#Unmarshal was expecting error for invalid date")
	}
}

func TestUnmarshalJSONforInvalidDate(t *testing.T) {
	d := Time{}
	dt := `"2001-02-03T04:05:06AAA"`

	if err := d.UnmarshalJSON([]byte(dt)); err == nil {
		t.Fatalf("date: Time#Unmarshal was expecting error for invalid date")
	}
}

func TestTimeString(t *testing.T) {
	ti, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	if err != nil {
		fmt.Println(err)
	}
	d := Time{ti}
	if d.String() != "2001-02-03T04:05:06Z" {
		t.Fatalf("date: Time#String failed (%v)", d.String())
	}
}

func TestTimeStringReturnsEmptyStringForError(t *testing.T) {
	d := Time{Time: time.Date(20000, 01, 01, 01, 01, 01, 01, time.UTC)}
	if d.String() != "" {
		t.Fatalf("date: Time#String failed empty string for an error")
	}
}

func TestTimeBinaryRoundTrip(t *testing.T) {
	ti, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	if err != nil {
		t.Fatalf("date: Time#ParseTime failed (%v)", err)
	}
	d1 := Time{ti}
	t1, err := d1.MarshalBinary()
	if err != nil {
		t.Fatalf("date: Time#MarshalBinary failed (%v)", err)
	}

	d2 := Time{}
	if err = d2.UnmarshalBinary(t1); err != nil {
		t.Fatalf("date: Time#UnmarshalBinary failed (%v)", err)
	}

	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("date:Round-trip Binary failed (%v, %v)", d1, d2)
	}
}

func TestTimeJSONRoundTrip(t *testing.T) {
	type s struct {
		Time Time `json:"datetime"`
	}

	ti, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	if err != nil {
		t.Fatalf("date: Time#ParseTime failed (%v)", err)
	}

	d1 := s{Time: Time{ti}}
	j, err := json.Marshal(d1)
	if err != nil {
		t.Fatalf("date: Time#MarshalJSON failed (%v)", err)
	}

	d2 := s{}
	if err = json.Unmarshal(j, &d2); err != nil {
		t.Fatalf("date: Time#UnmarshalJSON failed (%v)", err)
	}

	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("date: Round-trip JSON failed (%v, %v)", d1, d2)
	}
}

func TestTimeTextRoundTrip(t *testing.T) {
	ti, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	if err != nil {
		t.Fatalf("date: Time#ParseTime failed (%v)", err)
	}
	d1 := Time{Time: ti}
	t1, err := d1.MarshalText()
	if err != nil {
		t.Fatalf("date: Time#MarshalText failed (%v)", err)
	}

	d2 := Time{}
	if err = d2.UnmarshalText(t1); err != nil {
		t.Fatalf("date: Time#UnmarshalText failed (%v)", err)
	}

	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("date: Round-trip Text failed (%v, %v)", d1, d2)
	}
}

func TestTimeToTime(t *testing.T) {
	ti, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	d := Time{ti}
	if err != nil {
		t.Fatalf("date: Time#ParseTime failed (%v)", err)
	}
	var _ time.Time = d.ToTime()
}

func TestUnmarshalJSONNoOffset(t *testing.T) {
	var d struct {
		Time Time `json:"datetime"`
	}
	j := `{"datetime" : "2001-02-03T04:05:06.789"}`

	if err := json.Unmarshal([]byte(j), &d); err != nil {
		t.Fatalf("date: Time#Unmarshal failed (%v)", err)
	}
}

func TestUnmarshalJSONPosOffset(t *testing.T) {
	var d struct {
		Time Time `json:"datetime"`
	}
	j := `{"datetime" : "1980-01-02T00:11:35.01+01:00"}`

	if err := json.Unmarshal([]byte(j), &d); err != nil {
		t.Fatalf("date: Time#Unmarshal failed (%v)", err)
	}
}

func TestUnmarshalJSONNegOffset(t *testing.T) {
	var d struct {
		Time Time `json:"datetime"`
	}
	j := `{"datetime" : "1492-10-12T10:15:01.789-08:00"}`

	if err := json.Unmarshal([]byte(j), &d); err != nil {
		t.Fatalf("date: Time#Unmarshal failed (%v)", err)
	}
}

func TestUnmarshalTextNoOffset(t *testing.T) {
	d := Time{}
	t1 := "2001-02-03T04:05:06"

	if err := d.UnmarshalText([]byte(t1)); err != nil {
		t.Fatalf("date: Time#UnmarshalText failed (%v)", err)
	}
}

func TestUnmarshalTextPosOffset(t *testing.T) {
	d := Time{}
	t1 := "2001-02-03T04:05:06+00:30"

	if err := d.UnmarshalText([]byte(t1)); err != nil {
		t.Fatalf("date: Time#UnmarshalText failed (%v)", err)
	}
}

func TestUnmarshalTextNegOffset(t *testing.T) {
	d := Time{}
	t1 := "2001-02-03T04:05:06-11:00"

	if err := d.UnmarshalText([]byte(t1)); err != nil {
		t.Fatalf("date: Time#UnmarshalText failed (%v)", err)
	}
}
