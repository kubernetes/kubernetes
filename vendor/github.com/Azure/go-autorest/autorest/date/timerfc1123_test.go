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

func ExampleTimeRFC1123() {
	d, err := ParseTime(rfc1123, "Mon, 02 Jan 2006 15:04:05 MST")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(d)
	// Output: 2006-01-02 15:04:05 +0000 MST
}

func ExampleTimeRFC1123_MarshalBinary() {
	ti, err := ParseTime(rfc1123, "Mon, 02 Jan 2006 15:04:05 MST")
	if err != nil {
		fmt.Println(err)
	}
	d := TimeRFC1123{ti}
	b, err := d.MarshalBinary()
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(b))
	// Output: Mon, 02 Jan 2006 15:04:05 MST
}

func ExampleTimeRFC1123_UnmarshalBinary() {
	d := TimeRFC1123{}
	t := "Mon, 02 Jan 2006 15:04:05 MST"
	if err := d.UnmarshalBinary([]byte(t)); err != nil {
		fmt.Println(err)
	}
	fmt.Println(d)
	// Output: Mon, 02 Jan 2006 15:04:05 MST
}

func ExampleTimeRFC1123_MarshalJSON() {
	ti, err := ParseTime(rfc1123, "Mon, 02 Jan 2006 15:04:05 MST")
	if err != nil {
		fmt.Println(err)
	}
	d := TimeRFC1123{ti}
	j, err := json.Marshal(d)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(j))
	// Output: "Mon, 02 Jan 2006 15:04:05 MST"
}

func TestTimeRFC1123MarshalJSONInvalid(t *testing.T) {
	ti := time.Date(20000, 01, 01, 00, 00, 00, 00, time.UTC)
	d := TimeRFC1123{ti}
	if _, err := json.Marshal(d); err == nil {
		t.Fatalf("date: TimeRFC1123#Marshal failed for invalid date")
	}
}

func ExampleTimeRFC1123_UnmarshalJSON() {
	var d struct {
		Time TimeRFC1123 `json:"datetime"`
	}
	j := `{"datetime" : "Mon, 02 Jan 2006 15:04:05 MST"}`

	if err := json.Unmarshal([]byte(j), &d); err != nil {
		fmt.Println(err)
	}
	fmt.Println(d.Time)
	// Output: Mon, 02 Jan 2006 15:04:05 MST
}

func ExampleTimeRFC1123_MarshalText() {
	ti, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	if err != nil {
		fmt.Println(err)
	}
	d := TimeRFC1123{ti}
	t, err := d.MarshalText()
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(t))
	// Output: Sat, 03 Feb 2001 04:05:06 UTC
}

func ExampleTimeRFC1123_UnmarshalText() {
	d := TimeRFC1123{}
	t := "Sat, 03 Feb 2001 04:05:06 UTC"

	if err := d.UnmarshalText([]byte(t)); err != nil {
		fmt.Println(err)
	}
	fmt.Println(d)
	// Output: Sat, 03 Feb 2001 04:05:06 UTC
}

func TestUnmarshalJSONforInvalidDateRfc1123(t *testing.T) {
	dt := `"Mon, 02 Jan 2000000 15:05 MST"`
	d := TimeRFC1123{}
	if err := d.UnmarshalJSON([]byte(dt)); err == nil {
		t.Fatalf("date: TimeRFC1123#Unmarshal failed for invalid date")
	}
}

func TestUnmarshalTextforInvalidDateRfc1123(t *testing.T) {
	dt := "Mon, 02 Jan 2000000 15:05 MST"
	d := TimeRFC1123{}
	if err := d.UnmarshalText([]byte(dt)); err == nil {
		t.Fatalf("date: TimeRFC1123#Unmarshal failed for invalid date")
	}
}

func TestTimeStringRfc1123(t *testing.T) {
	ti, err := ParseTime(rfc1123, "Mon, 02 Jan 2006 15:04:05 MST")
	if err != nil {
		fmt.Println(err)
	}
	d := TimeRFC1123{ti}
	if d.String() != "Mon, 02 Jan 2006 15:04:05 MST" {
		t.Fatalf("date: TimeRFC1123#String failed (%v)", d.String())
	}
}

func TestTimeStringReturnsEmptyStringForErrorRfc1123(t *testing.T) {
	d := TimeRFC1123{Time: time.Date(20000, 01, 01, 01, 01, 01, 01, time.UTC)}
	if d.String() != "" {
		t.Fatalf("date: TimeRFC1123#String failed empty string for an error")
	}
}

func TestTimeBinaryRoundTripRfc1123(t *testing.T) {
	ti, err := ParseTime(rfc3339, "2001-02-03T04:05:06Z")
	if err != nil {
		t.Fatalf("date: TimeRFC1123#ParseTime failed (%v)", err)
	}
	d1 := TimeRFC1123{ti}
	t1, err := d1.MarshalBinary()
	if err != nil {
		t.Fatalf("date: TimeRFC1123#MarshalBinary failed (%v)", err)
	}

	d2 := TimeRFC1123{}
	if err = d2.UnmarshalBinary(t1); err != nil {
		t.Fatalf("date: TimeRFC1123#UnmarshalBinary failed (%v)", err)
	}

	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("date: Round-trip Binary failed (%v, %v)", d1, d2)
	}
}

func TestTimeJSONRoundTripRfc1123(t *testing.T) {
	type s struct {
		Time TimeRFC1123 `json:"datetime"`
	}
	var err error
	ti, err := ParseTime(rfc1123, "Mon, 02 Jan 2006 15:04:05 MST")
	if err != nil {
		t.Fatalf("date: TimeRFC1123#ParseTime failed (%v)", err)
	}
	d1 := s{Time: TimeRFC1123{ti}}
	j, err := json.Marshal(d1)
	if err != nil {
		t.Fatalf("date: TimeRFC1123#MarshalJSON failed (%v)", err)
	}

	d2 := s{}
	if err = json.Unmarshal(j, &d2); err != nil {
		t.Fatalf("date: TimeRFC1123#UnmarshalJSON failed (%v)", err)
	}

	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("date: Round-trip JSON failed (%v, %v)", d1, d2)
	}
}

func TestTimeTextRoundTripRfc1123(t *testing.T) {
	ti, err := ParseTime(rfc1123, "Mon, 02 Jan 2006 15:04:05 MST")
	if err != nil {
		t.Fatalf("date: TimeRFC1123#ParseTime failed (%v)", err)
	}
	d1 := TimeRFC1123{Time: ti}
	t1, err := d1.MarshalText()
	if err != nil {
		t.Fatalf("date: TimeRFC1123#MarshalText failed (%v)", err)
	}

	d2 := TimeRFC1123{}
	if err = d2.UnmarshalText(t1); err != nil {
		t.Fatalf("date: TimeRFC1123#UnmarshalText failed (%v)", err)
	}

	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("date: Round-trip Text failed (%v, %v)", d1, d2)
	}
}

func TestTimeToTimeRFC1123(t *testing.T) {
	ti, err := ParseTime(rfc1123, "Mon, 02 Jan 2006 15:04:05 MST")
	d := TimeRFC1123{ti}
	if err != nil {
		t.Fatalf("date: TimeRFC1123#ParseTime failed (%v)", err)
	}
	var _ time.Time = d.ToTime()
}
