// +build go1.7

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
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"testing"
	"time"
)

func ExampleUnixTime_MarshalJSON() {
	epoch := UnixTime(UnixEpoch())
	text, _ := json.Marshal(epoch)
	fmt.Print(string(text))
	// Output: 0
}

func ExampleUnixTime_UnmarshalJSON() {
	var myTime UnixTime
	json.Unmarshal([]byte("1.3e2"), &myTime)
	fmt.Printf("%v", time.Time(myTime))
	// Output: 1970-01-01 00:02:10 +0000 UTC
}

func TestUnixTime_MarshalJSON(t *testing.T) {
	testCases := []time.Time{
		UnixEpoch().Add(-1 * time.Second),                        // One second befote the Unix Epoch
		time.Date(2017, time.April, 14, 20, 27, 47, 0, time.UTC), // The time this test was written
		UnixEpoch(),
		time.Date(1800, 01, 01, 0, 0, 0, 0, time.UTC),
		time.Date(2200, 12, 29, 00, 01, 37, 82, time.UTC),
	}

	for _, tc := range testCases {
		t.Run(tc.String(), func(subT *testing.T) {
			var actual, expected float64
			var marshaled []byte

			target := UnixTime(tc)
			expected = float64(target.Duration().Nanoseconds()) / 1e9

			if temp, err := json.Marshal(target); err == nil {
				marshaled = temp
			} else {
				subT.Error(err)
				return
			}

			dec := json.NewDecoder(bytes.NewReader(marshaled))
			if err := dec.Decode(&actual); err != nil {
				subT.Error(err)
				return
			}

			diff := math.Abs(actual - expected)
			subT.Logf("\ngot :\t%g\nwant:\t%g\ndiff:\t%g", actual, expected, diff)
			if diff > 1e-9 { //Must be within 1 nanosecond of one another
				subT.Fail()
			}
		})
	}
}

func TestUnixTime_UnmarshalJSON(t *testing.T) {
	testCases := []struct {
		text     string
		expected time.Time
	}{
		{"1", UnixEpoch().Add(time.Second)},
		{"0", UnixEpoch()},
		{"1492203742", time.Date(2017, time.April, 14, 21, 02, 22, 0, time.UTC)}, // The time this test was written
		{"-1", time.Date(1969, time.December, 31, 23, 59, 59, 0, time.UTC)},
		{"1.5", UnixEpoch().Add(1500 * time.Millisecond)},
		{"0e1", UnixEpoch()}, // See http://json.org for 'number' format definition.
		{"1.3e+2", UnixEpoch().Add(130 * time.Second)},
		{"1.6E-10", UnixEpoch()}, // This is so small, it should get truncated into the UnixEpoch
		{"2E-6", UnixEpoch().Add(2 * time.Microsecond)},
		{"1.289345e9", UnixEpoch().Add(1289345000 * time.Second)},
		{"1e-9", UnixEpoch().Add(time.Nanosecond)},
	}

	for _, tc := range testCases {
		t.Run(tc.text, func(subT *testing.T) {
			var rehydrated UnixTime
			if err := json.Unmarshal([]byte(tc.text), &rehydrated); err != nil {
				subT.Error(err)
				return
			}

			if time.Time(rehydrated) != tc.expected {
				subT.Logf("\ngot: \t%v\nwant:\t%v\ndiff:\t%v", time.Time(rehydrated), tc.expected, time.Time(rehydrated).Sub(tc.expected))
				subT.Fail()
			}
		})
	}
}

func TestUnixTime_JSONRoundTrip(t *testing.T) {
	testCases := []time.Time{
		UnixEpoch(),
		time.Date(2005, time.November, 5, 0, 0, 0, 0, time.UTC), // The day V for Vendetta (film) was released.
		UnixEpoch().Add(-6 * time.Second),
		UnixEpoch().Add(800 * time.Hour),
		UnixEpoch().Add(time.Nanosecond),
		time.Date(2015, time.September, 05, 4, 30, 12, 9992, time.UTC),
	}

	for _, tc := range testCases {
		t.Run(tc.String(), func(subT *testing.T) {
			subject := UnixTime(tc)
			var marshaled []byte
			if temp, err := json.Marshal(subject); err == nil {
				marshaled = temp
			} else {
				subT.Error(err)
				return
			}

			var unmarshaled UnixTime
			if err := json.Unmarshal(marshaled, &unmarshaled); err != nil {
				subT.Error(err)
			}

			actual := time.Time(unmarshaled)
			diff := actual.Sub(tc)
			subT.Logf("\ngot :\t%s\nwant:\t%s\ndiff:\t%s", actual.String(), tc.String(), diff.String())

			if diff > time.Duration(100) { // We lose some precision be working in floats. We shouldn't lose more than 100 nanoseconds.
				subT.Fail()
			}
		})
	}
}

func TestUnixTime_MarshalBinary(t *testing.T) {
	testCases := []struct {
		expected int64
		subject  time.Time
	}{
		{0, UnixEpoch()},
		{-15 * int64(time.Second), UnixEpoch().Add(-15 * time.Second)},
		{54, UnixEpoch().Add(54 * time.Nanosecond)},
	}

	for _, tc := range testCases {
		t.Run("", func(subT *testing.T) {
			var marshaled []byte

			if temp, err := UnixTime(tc.subject).MarshalBinary(); err == nil {
				marshaled = temp
			} else {
				subT.Error(err)
				return
			}

			var unmarshaled int64
			if err := binary.Read(bytes.NewReader(marshaled), binary.LittleEndian, &unmarshaled); err != nil {
				subT.Error(err)
				return
			}

			if unmarshaled != tc.expected {
				subT.Logf("\ngot: \t%d\nwant:\t%d", unmarshaled, tc.expected)
				subT.Fail()
			}
		})
	}
}

func TestUnixTime_BinaryRoundTrip(t *testing.T) {
	testCases := []time.Time{
		UnixEpoch(),
		UnixEpoch().Add(800 * time.Minute),
		UnixEpoch().Add(7 * time.Hour),
		UnixEpoch().Add(-1 * time.Nanosecond),
	}

	for _, tc := range testCases {
		t.Run(tc.String(), func(subT *testing.T) {
			original := UnixTime(tc)
			var marshaled []byte

			if temp, err := original.MarshalBinary(); err == nil {
				marshaled = temp
			} else {
				subT.Error(err)
				return
			}

			var traveled UnixTime
			if err := traveled.UnmarshalBinary(marshaled); err != nil {
				subT.Error(err)
				return
			}

			if traveled != original {
				subT.Logf("\ngot: \t%s\nwant:\t%s", time.Time(original).String(), time.Time(traveled).String())
				subT.Fail()
			}
		})
	}
}

func TestUnixTime_MarshalText(t *testing.T) {
	testCases := []time.Time{
		UnixEpoch(),
		UnixEpoch().Add(45 * time.Second),
		UnixEpoch().Add(time.Nanosecond),
		UnixEpoch().Add(-100000 * time.Second),
	}

	for _, tc := range testCases {
		expected, _ := tc.MarshalText()
		t.Run("", func(subT *testing.T) {
			var marshaled []byte

			if temp, err := UnixTime(tc).MarshalText(); err == nil {
				marshaled = temp
			} else {
				subT.Error(err)
				return
			}

			if string(marshaled) != string(expected) {
				subT.Logf("\ngot: \t%s\nwant:\t%s", string(marshaled), string(expected))
				subT.Fail()
			}
		})
	}
}

func TestUnixTime_TextRoundTrip(t *testing.T) {
	testCases := []time.Time{
		UnixEpoch(),
		UnixEpoch().Add(-1 * time.Nanosecond),
		UnixEpoch().Add(1 * time.Nanosecond),
		time.Date(2017, time.April, 17, 21, 00, 00, 00, time.UTC),
	}

	for _, tc := range testCases {
		t.Run(tc.String(), func(subT *testing.T) {
			unixTC := UnixTime(tc)

			var marshaled []byte

			if temp, err := unixTC.MarshalText(); err == nil {
				marshaled = temp
			} else {
				subT.Error(err)
				return
			}

			var unmarshaled UnixTime
			if err := unmarshaled.UnmarshalText(marshaled); err != nil {
				subT.Error(err)
				return
			}

			if unmarshaled != unixTC {
				t.Logf("\ngot: \t%s\nwant:\t%s", time.Time(unmarshaled).String(), tc.String())
				t.Fail()
			}
		})
	}
}
