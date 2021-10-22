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
	"bytes"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

var (
	p, _ = time.Parse(time.RFC3339Nano, "2011-08-18T19:03:37.000000000+01:00")

	testCases = []struct {
		in   []byte    // externally sourced data -- to be unmarshalled
		time time.Time // its representation in time.Time
		str  string    // its marshalled representation
	}{
		{[]byte("2014-12-15T08:00:00"), time.Date(2014, 12, 15, 8, 0, 0, 0, time.UTC), "2014-12-15T08:00:00.000Z"},
		{[]byte("2014-12-15T08:00:00.000Z"), time.Date(2014, 12, 15, 8, 0, 0, 0, time.UTC), "2014-12-15T08:00:00.000Z"},
		{[]byte("2011-08-18T19:03:37.000000000+01:00"), time.Date(2011, 8, 18, 19, 3, 37, 0, p.Location()), "2011-08-18T19:03:37.000+01:00"},
		{[]byte("2014-12-15T19:30:20Z"), time.Date(2014, 12, 15, 19, 30, 20, 0, time.UTC), "2014-12-15T19:30:20.000Z"},
		{[]byte("0001-01-01T00:00:00Z"), time.Time{}.UTC(), "0001-01-01T00:00:00.000Z"},
		{[]byte(""), time.Unix(0, 0).UTC(), "1970-01-01T00:00:00.000Z"},
		{[]byte(nil), time.Unix(0, 0).UTC(), "1970-01-01T00:00:00.000Z"},
	}
)

func TestNewDateTime(t *testing.T) {
	assert.EqualValues(t, time.Unix(0, 0).UTC(), NewDateTime())
}

func TestParseDateTime_errorCases(t *testing.T) {
	_, err := ParseDateTime("yada")
	assert.Error(t, err)
}

// TestParseDateTime tests the full cycle:
// parsing -> marshalling -> unmarshalling / scanning
func TestParseDateTime_fullCycle(t *testing.T) {
	for caseNum, example := range testCases {
		t.Logf("Case #%d", caseNum)

		parsed, err := ParseDateTime(example.str)
		assert.NoError(t, err)
		assert.EqualValues(t, example.time, parsed)

		mt, err := parsed.MarshalText()
		assert.NoError(t, err)
		assert.Equal(t, []byte(example.str), mt)

		if example.str != "" {
			v := IsDateTime(example.str)
			assert.True(t, v)
		} else {
			t.Logf("IsDateTime() skipped for empty testcases")
		}

		pp := NewDateTime()
		err = pp.UnmarshalText(mt)
		assert.NoError(t, err)
		assert.EqualValues(t, example.time, pp)

		pp = NewDateTime()
		err = pp.Scan(mt)
		assert.NoError(t, err)
		assert.Equal(t, DateTime(example.time), pp)
	}
}

func TestDateTime_IsDateTime_errorCases(t *testing.T) {
	v := IsDateTime("zor")
	assert.False(t, v)

	v = IsDateTime("zorg")
	assert.False(t, v)

	v = IsDateTime("zorgTx")
	assert.False(t, v)

	v = IsDateTime("1972-12-31Tx")
	assert.False(t, v)

	v = IsDateTime("1972-12-31T24:40:00.000Z")
	assert.False(t, v)

	v = IsDateTime("1972-12-31T23:63:00.000Z")
	assert.False(t, v)

	v = IsDateTime("1972-12-31T23:59:60.000Z")
	assert.False(t, v)

}
func TestDateTime_UnmarshalText_errorCases(t *testing.T) {
	pp := NewDateTime()
	err := pp.UnmarshalText([]byte("yada"))
	assert.Error(t, err)
	err = pp.UnmarshalJSON([]byte("yada"))
	assert.Error(t, err)
}

func TestDateTime_UnmarshalText(t *testing.T) {
	for caseNum, example := range testCases {
		t.Logf("Case #%d", caseNum)
		pp := NewDateTime()
		err := pp.UnmarshalText(example.in)
		assert.NoError(t, err)
		assert.EqualValues(t, example.time, pp)
	}
}
func TestDateTime_UnmarshalJSON(t *testing.T) {
	for caseNum, example := range testCases {
		t.Logf("Case #%d", caseNum)
		pp := NewDateTime()
		err := pp.UnmarshalJSON(esc(example.in))
		assert.NoError(t, err)
		assert.EqualValues(t, example.time, pp)
	}

	// Check UnmarshalJSON failure with no lexed items
	pp := NewDateTime()
	err := pp.UnmarshalJSON([]byte("zorg emperor"))
	assert.Error(t, err)

	// Check lexer failure
	err = pp.UnmarshalJSON([]byte(`"zorg emperor"`))
	assert.Error(t, err)

	// Check null case
	err = pp.UnmarshalJSON([]byte("null"))
	assert.Nil(t, err)
}

func esc(v []byte) []byte {
	var buf bytes.Buffer
	buf.WriteByte('"')
	buf.Write(v)
	buf.WriteByte('"')
	return buf.Bytes()
}

func TestDateTime_MarshalText(t *testing.T) {
	for caseNum, example := range testCases {
		t.Logf("Case #%d", caseNum)
		dt := DateTime(example.time)
		mt, err := dt.MarshalText()
		assert.NoError(t, err)
		assert.Equal(t, []byte(example.str), mt)
	}
}
func TestDateTime_MarshalJSON(t *testing.T) {
	for caseNum, example := range testCases {
		t.Logf("Case #%d", caseNum)
		dt := DateTime(example.time)
		bb, err := dt.MarshalJSON()
		assert.NoError(t, err)
		assert.EqualValues(t, esc([]byte(example.str)), bb)
	}
}

func TestDateTime_Scan(t *testing.T) {
	for caseNum, example := range testCases {
		t.Logf("Case #%d", caseNum)

		pp := NewDateTime()
		err := pp.Scan(example.in)
		assert.NoError(t, err)
		assert.Equal(t, DateTime(example.time), pp)

		pp = NewDateTime()
		err = pp.Scan(string(example.in))
		assert.NoError(t, err)
		assert.Equal(t, DateTime(example.time), pp)

		pp = NewDateTime()
		err = pp.Scan(example.time)
		assert.NoError(t, err)
		assert.Equal(t, DateTime(example.time), pp)
	}
}

func TestDateTime_Scan_Failed(t *testing.T) {
	pp := NewDateTime()
	zero := NewDateTime()

	err := pp.Scan(nil)
	assert.NoError(t, err)
	// Zero values differ...
	//assert.Equal(t, zero, pp)
	assert.Equal(t, DateTime{}, pp)

	err = pp.Scan("")
	assert.NoError(t, err)
	assert.Equal(t, zero, pp)

	err = pp.Scan(int64(0))
	assert.Error(t, err)

	err = pp.Scan(float64(0))
	assert.Error(t, err)
}

func TestDeepCopyDateTime(t *testing.T) {
	p, err := ParseDateTime("2011-08-18T19:03:37.000000000+01:00")
	assert.NoError(t, err)
	in := &p

	out := new(DateTime)
	in.DeepCopyInto(out)
	assert.Equal(t, in, out)

	out2 := in.DeepCopy()
	assert.Equal(t, in, out2)

	var inNil *DateTime
	out3 := inNil.DeepCopy()
	assert.Nil(t, out3)
}
