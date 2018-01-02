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

func ExampleParseDate() {
	d, err := ParseDate("2001-02-03")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(d)
	// Output: 2001-02-03
}

func ExampleDate() {
	d, err := ParseDate("2001-02-03")
	if err != nil {
		fmt.Println(err)
	}

	t, err := time.Parse(time.RFC3339, "2001-02-04T00:00:00Z")
	if err != nil {
		fmt.Println(err)
	}

	// Date acts as time.Time when the receiver
	if d.Before(t) {
		fmt.Printf("Before ")
	} else {
		fmt.Printf("After ")
	}

	// Convert Date when needing a time.Time
	if t.After(d.ToTime()) {
		fmt.Printf("After")
	} else {
		fmt.Printf("Before")
	}
	// Output: Before After
}

func ExampleDate_MarshalBinary() {
	d, err := ParseDate("2001-02-03")
	if err != nil {
		fmt.Println(err)
	}
	t, err := d.MarshalBinary()
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(t))
	// Output: 2001-02-03
}

func ExampleDate_UnmarshalBinary() {
	d := Date{}
	t := "2001-02-03"

	if err := d.UnmarshalBinary([]byte(t)); err != nil {
		fmt.Println(err)
	}
	fmt.Println(d)
	// Output: 2001-02-03
}

func ExampleDate_MarshalJSON() {
	d, err := ParseDate("2001-02-03")
	if err != nil {
		fmt.Println(err)
	}
	j, err := json.Marshal(d)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(j))
	// Output: "2001-02-03"
}

func ExampleDate_UnmarshalJSON() {
	var d struct {
		Date Date `json:"date"`
	}
	j := `{"date" : "2001-02-03"}`

	if err := json.Unmarshal([]byte(j), &d); err != nil {
		fmt.Println(err)
	}
	fmt.Println(d.Date)
	// Output: 2001-02-03
}

func ExampleDate_MarshalText() {
	d, err := ParseDate("2001-02-03")
	if err != nil {
		fmt.Println(err)
	}
	t, err := d.MarshalText()
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(t))
	// Output: 2001-02-03
}

func ExampleDate_UnmarshalText() {
	d := Date{}
	t := "2001-02-03"

	if err := d.UnmarshalText([]byte(t)); err != nil {
		fmt.Println(err)
	}
	fmt.Println(d)
	// Output: 2001-02-03
}

func TestDateString(t *testing.T) {
	d, err := ParseDate("2001-02-03")
	if err != nil {
		t.Fatalf("date: String failed (%v)", err)
	}
	if d.String() != "2001-02-03" {
		t.Fatalf("date: String failed (%v)", d.String())
	}
}

func TestDateBinaryRoundTrip(t *testing.T) {
	d1, err := ParseDate("2001-02-03")
	if err != nil {
		t.Fatalf("date: ParseDate failed (%v)", err)
	}
	t1, err := d1.MarshalBinary()
	if err != nil {
		t.Fatalf("date: MarshalBinary failed (%v)", err)
	}

	d2 := Date{}
	if err = d2.UnmarshalBinary(t1); err != nil {
		t.Fatalf("date: UnmarshalBinary failed (%v)", err)
	}

	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("date: Round-trip Binary failed (%v, %v)", d1, d2)
	}
}

func TestDateJSONRoundTrip(t *testing.T) {
	type s struct {
		Date Date `json:"date"`
	}
	var err error
	d1 := s{}
	d1.Date, err = ParseDate("2001-02-03")
	if err != nil {
		t.Fatalf("date: ParseDate failed (%v)", err)
	}

	j, err := json.Marshal(d1)
	if err != nil {
		t.Fatalf("date: MarshalJSON failed (%v)", err)
	}

	d2 := s{}
	if err = json.Unmarshal(j, &d2); err != nil {
		t.Fatalf("date: UnmarshalJSON failed (%v)", err)
	}

	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("date: Round-trip JSON failed (%v, %v)", d1, d2)
	}
}

func TestDateTextRoundTrip(t *testing.T) {
	d1, err := ParseDate("2001-02-03")
	if err != nil {
		t.Fatalf("date: ParseDate failed (%v)", err)
	}
	t1, err := d1.MarshalText()
	if err != nil {
		t.Fatalf("date: MarshalText failed (%v)", err)
	}
	d2 := Date{}
	if err = d2.UnmarshalText(t1); err != nil {
		t.Fatalf("date: UnmarshalText failed (%v)", err)
	}

	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("date: Round-trip Text failed (%v, %v)", d1, d2)
	}
}

func TestDateToTime(t *testing.T) {
	var d Date
	d, err := ParseDate("2001-02-03")
	if err != nil {
		t.Fatalf("date: ParseDate failed (%v)", err)
	}
	var _ time.Time = d.ToTime()
}

func TestDateUnmarshalJSONReturnsError(t *testing.T) {
	var d struct {
		Date Date `json:"date"`
	}
	j := `{"date" : "February 3, 2001"}`

	if err := json.Unmarshal([]byte(j), &d); err == nil {
		t.Fatal("date: Date failed to return error for malformed JSON date")
	}
}

func TestDateUnmarshalTextReturnsError(t *testing.T) {
	d := Date{}
	txt := "February 3, 2001"

	if err := d.UnmarshalText([]byte(txt)); err == nil {
		t.Fatal("date: Date failed to return error for malformed Text date")
	}
}
