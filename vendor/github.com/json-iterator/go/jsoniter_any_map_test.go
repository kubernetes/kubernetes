package jsoniter

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func Test_wrap_map(t *testing.T) {
	should := require.New(t)
	any := Wrap(map[string]string{"Field1": "hello"})
	should.Equal("hello", any.Get("Field1").ToString())
	any = Wrap(map[string]string{"Field1": "hello"})
	should.Equal(1, any.Size())
}
