package jsoniter

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

type MyEnum int64

const (
	MyEnumA MyEnum = iota
	MyEnumB
)

func (m *MyEnum) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"foo-%d"`, int(*m))), nil
}

func (m *MyEnum) UnmarshalJSON(jb []byte) error {
	switch string(jb) {
	case `"foo-1"`:
		*m = MyEnumB
	default:
		*m = MyEnumA
	}
	return nil
}

func Test_custom_marshaler_on_enum(t *testing.T) {
	type Wrapper struct {
		Payload interface{}
	}
	type Wrapper2 struct {
		Payload MyEnum
	}
	should := require.New(t)

	w := Wrapper{Payload: MyEnumB}

	jb, err := Marshal(w)
	should.NoError(err)
	should.Equal(`{"Payload":"foo-1"}`, string(jb))

	var w2 Wrapper2
	err = Unmarshal(jb, &w2)
	should.NoError(err)
	should.Equal(MyEnumB, w2.Payload)
}
