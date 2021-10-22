package any_tests

import (
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

var stringConvertMap = map[string]string{
	"null":                   "",
	"321.1":                  "321.1",
	`"1.1"`:                  "1.1",
	`"-123.1"`:               "-123.1",
	"0.0":                    "0.0",
	"0":                      "0",
	`"0"`:                    "0",
	`"0.0"`:                  "0.0",
	`"00.0"`:                 "00.0",
	"true":                   "true",
	"false":                  "false",
	`"true"`:                 "true",
	`"false"`:                "false",
	`"true123"`:              "true123",
	`"+1"`:                   "+1",
	"[]":                     "[]",
	"[1,2]":                  "[1,2]",
	"{}":                     "{}",
	`{"a":1, "stream":true}`: `{"a":1, "stream":true}`,
}

func Test_read_any_to_string(t *testing.T) {
	should := require.New(t)
	for k, v := range stringConvertMap {
		any := jsoniter.Get([]byte(k))
		should.Equal(v, any.ToString(), "original val "+k)
	}
}

func Test_read_string_as_any(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte(`"hello"`))
	should.Equal("hello", any.ToString())
	should.True(any.ToBool())
	any = jsoniter.Get([]byte(`" "`))
	should.False(any.ToBool())
	any = jsoniter.Get([]byte(`"false"`))
	should.True(any.ToBool())
	any = jsoniter.Get([]byte(`"123"`))
	should.Equal(123, any.ToInt())
}

func Test_wrap_string(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("-32000")).MustBeValid()
	should.Equal(-32000, any.ToInt())
	should.NoError(any.LastError())
}
