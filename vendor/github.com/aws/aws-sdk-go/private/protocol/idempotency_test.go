package protocol_test

import (
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/stretchr/testify/assert"
)

func TestCanSetIdempotencyToken(t *testing.T) {
	cases := []struct {
		CanSet bool
		Case   interface{}
	}{
		{
			true,
			struct {
				Field *string `idempotencyToken:"true"`
			}{},
		},
		{
			true,
			struct {
				Field string `idempotencyToken:"true"`
			}{},
		},
		{
			false,
			struct {
				Field *string `idempotencyToken:"true"`
			}{Field: new(string)},
		},
		{
			false,
			struct {
				Field string `idempotencyToken:"true"`
			}{Field: "value"},
		},
		{
			false,
			struct {
				Field *int `idempotencyToken:"true"`
			}{},
		},
		{
			false,
			struct {
				Field *string
			}{},
		},
	}

	for i, c := range cases {
		v := reflect.Indirect(reflect.ValueOf(c.Case))
		ty := v.Type()
		canSet := protocol.CanSetIdempotencyToken(v.Field(0), ty.Field(0))
		assert.Equal(t, c.CanSet, canSet, "Expect case %d can set to match", i)
	}
}

func TestSetIdempotencyToken(t *testing.T) {
	cases := []struct {
		Case interface{}
	}{
		{
			&struct {
				Field *string `idempotencyToken:"true"`
			}{},
		},
		{
			&struct {
				Field string `idempotencyToken:"true"`
			}{},
		},
		{
			&struct {
				Field *string `idempotencyToken:"true"`
			}{Field: new(string)},
		},
		{
			&struct {
				Field string `idempotencyToken:"true"`
			}{Field: ""},
		},
	}

	for i, c := range cases {
		v := reflect.Indirect(reflect.ValueOf(c.Case))

		protocol.SetIdempotencyToken(v.Field(0))
		assert.NotEmpty(t, v.Field(0).Interface(), "Expect case %d to be set", i)
	}
}

func TestUUIDVersion4(t *testing.T) {
	uuid := protocol.UUIDVersion4(make([]byte, 16))
	assert.Equal(t, `00000000-0000-4000-8000-000000000000`, uuid)

	b := make([]byte, 16)
	for i := 0; i < len(b); i++ {
		b[i] = 1
	}
	uuid = protocol.UUIDVersion4(b)
	assert.Equal(t, `01010101-0101-4101-8101-010101010101`, uuid)
}
