// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package strfmt

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestDuration(t *testing.T) {
	pp := Duration(0)

	err := pp.UnmarshalText([]byte("0ms"))
	assert.NoError(t, err)
	err = pp.UnmarshalText([]byte("yada"))
	assert.Error(t, err)

	orig := "2ms"
	b := []byte(orig)
	bj := []byte("\"" + orig + "\"")

	err = pp.UnmarshalText(b)
	assert.NoError(t, err)

	txt, err := pp.MarshalText()
	assert.NoError(t, err)
	assert.Equal(t, orig, string(txt))

	err = pp.UnmarshalJSON(bj)
	assert.NoError(t, err)
	assert.EqualValues(t, orig, pp.String())

	b, err = pp.MarshalJSON()
	assert.NoError(t, err)
	assert.Equal(t, bj, b)
}

func testDurationParser(t *testing.T, toParse string, expected time.Duration) {
	r, e := ParseDuration(toParse)
	assert.NoError(t, e)
	assert.Equal(t, expected, r)
}

func testDurationSQLScanner(t *testing.T, dur time.Duration) {
	values := []interface{}{int64(dur), float64(dur)}
	for _, value := range values {
		var result Duration
		err := result.Scan(value)
		assert.NoError(t, err)
		assert.Equal(t, dur, time.Duration(result))
	}
}

func TestDurationParser(t *testing.T) {
	testcases := map[string]time.Duration{

		// parse the short forms without spaces
		"1ns": 1 * time.Nanosecond,
		"1us": 1 * time.Microsecond,
		"1µs": 1 * time.Microsecond,
		"1ms": 1 * time.Millisecond,
		"1s":  1 * time.Second,
		"1m":  1 * time.Minute,
		"1h":  1 * time.Hour,
		"1hr": 1 * time.Hour,
		"1d":  24 * time.Hour,
		"1w":  7 * 24 * time.Hour,
		"1wk": 7 * 24 * time.Hour,

		// parse the long forms without spaces
		"1nanoseconds":  1 * time.Nanosecond,
		"1nanos":        1 * time.Nanosecond,
		"1microseconds": 1 * time.Microsecond,
		"1micros":       1 * time.Microsecond,
		"1millis":       1 * time.Millisecond,
		"1milliseconds": 1 * time.Millisecond,
		"1second":       1 * time.Second,
		"1sec":          1 * time.Second,
		"1min":          1 * time.Minute,
		"1minute":       1 * time.Minute,
		"1hour":         1 * time.Hour,
		"1day":          24 * time.Hour,
		"1week":         7 * 24 * time.Hour,

		// parse the short forms with spaces
		"1  ns": 1 * time.Nanosecond,
		"1  us": 1 * time.Microsecond,
		"1  µs": 1 * time.Microsecond,
		"1  ms": 1 * time.Millisecond,
		"1  s":  1 * time.Second,
		"1  m":  1 * time.Minute,
		"1  h":  1 * time.Hour,
		"1  hr": 1 * time.Hour,
		"1  d":  24 * time.Hour,
		"1  w":  7 * 24 * time.Hour,
		"1  wk": 7 * 24 * time.Hour,

		// parse the long forms without spaces
		"1  nanoseconds":  1 * time.Nanosecond,
		"1  nanos":        1 * time.Nanosecond,
		"1  microseconds": 1 * time.Microsecond,
		"1  micros":       1 * time.Microsecond,
		"1  millis":       1 * time.Millisecond,
		"1  milliseconds": 1 * time.Millisecond,
		"1  second":       1 * time.Second,
		"1  sec":          1 * time.Second,
		"1  min":          1 * time.Minute,
		"1  minute":       1 * time.Minute,
		"1  hour":         1 * time.Hour,
		"1  day":          24 * time.Hour,
		"1  week":         7 * 24 * time.Hour,
	}

	for str, dur := range testcases {
		testDurationParser(t, str, dur)
		testDurationSQLScanner(t, dur)
	}
}
