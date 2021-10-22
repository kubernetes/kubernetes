package any_tests

import (
	"fmt"
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
)

var boolConvertMap = map[string]bool{
	"null":  false,
	"true":  true,
	"false": false,

	`"true"`:  true,
	`"false"`: true,

	"123":   true,
	`"123"`: true,
	"0":     false,
	`"0"`:   false,
	"-1":    true,
	`"-1"`:  true,

	"1.1":       true,
	"0.0":       false,
	"-1.1":      true,
	`""`:        false,
	"[1,2]":     true,
	"[]":        false,
	"{}":        true,
	`{"abc":1}`: true,
}

func Test_read_bool_as_any(t *testing.T) {
	should := require.New(t)

	var any jsoniter.Any
	for k, v := range boolConvertMap {
		any = jsoniter.Get([]byte(k))
		if v {
			should.True(any.ToBool(), fmt.Sprintf("origin val is %v", k))
		} else {
			should.False(any.ToBool(), fmt.Sprintf("origin val is %v", k))
		}
	}

}

func Test_write_bool_to_stream(t *testing.T) {
	should := require.New(t)
	any := jsoniter.Get([]byte("true"))
	stream := jsoniter.NewStream(jsoniter.ConfigDefault, nil, 32)
	any.WriteTo(stream)
	should.Equal("true", string(stream.Buffer()))
	should.Equal(any.ValueType(), jsoniter.BoolValue)

	any = jsoniter.Get([]byte("false"))
	stream = jsoniter.NewStream(jsoniter.ConfigDefault, nil, 32)
	any.WriteTo(stream)
	should.Equal("false", string(stream.Buffer()))

	should.Equal(any.ValueType(), jsoniter.BoolValue)
}
