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
	"database/sql"
	"database/sql/driver"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

var _ sql.Scanner = &Date{}
var _ driver.Valuer = Date{}

func TestDate(t *testing.T) {
	pp := Date{}
	err := pp.UnmarshalText([]byte{})
	assert.NoError(t, err)
	err = pp.UnmarshalText([]byte("yada"))
	assert.Error(t, err)
	orig := "2014-12-15"
	b := []byte(orig)
	bj := []byte("\"" + orig + "\"")
	err = pp.UnmarshalText([]byte(orig))
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

func TestDate_Scan(t *testing.T) {
	ref := time.Now().Truncate(24 * time.Hour).UTC()
	date, str := Date(ref), ref.Format(RFC3339FullDate)

	values := []interface{}{str, []byte(str), ref}
	for _, value := range values {
		result := Date{}
		(&result).Scan(value)
		assert.Equal(t, date, result, "value: %#v", value)
	}
}

func TestDate_Value(t *testing.T) {
	ref := time.Now().Truncate(24 * time.Hour).UTC()
	date := Date(ref)
	dbv, err := date.Value()
	assert.NoError(t, err)
	assert.EqualValues(t, dbv, ref)
}
