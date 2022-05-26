package protocol

import (
	"crypto/rand"
	"fmt"
	"reflect"
)

// RandReader is the random reader the protocol package will use to read
// random bytes from. This is exported for testing, and should not be used.
var RandReader = rand.Reader

const idempotencyTokenFillTag = `idempotencyToken`

// CanSetIdempotencyToken returns true if the struct field should be
// automatically populated with a Idempotency token.
//
// Only *string and string type fields that are tagged with idempotencyToken
// which are not already set can be auto filled.
func CanSetIdempotencyToken(v reflect.Value, f reflect.StructField) bool {
	switch u := v.Interface().(type) {
	// To auto fill an Idempotency token the field must be a string,
	// tagged for auto fill, and have a zero value.
	case *string:
		return u == nil && len(f.Tag.Get(idempotencyTokenFillTag)) != 0
	case string:
		return len(u) == 0 && len(f.Tag.Get(idempotencyTokenFillTag)) != 0
	}

	return false
}

// GetIdempotencyToken returns a randomly generated idempotency token.
func GetIdempotencyToken() string {
	b := make([]byte, 16)
	RandReader.Read(b)

	return UUIDVersion4(b)
}

// SetIdempotencyToken will set the value provided with a Idempotency Token.
// Given that the value can be set. Will panic if value is not setable.
func SetIdempotencyToken(v reflect.Value) {
	if v.Kind() == reflect.Ptr {
		if v.IsNil() && v.CanSet() {
			v.Set(reflect.New(v.Type().Elem()))
		}
		v = v.Elem()
	}
	v = reflect.Indirect(v)

	if !v.CanSet() {
		panic(fmt.Sprintf("unable to set idempotnecy token %v", v))
	}

	b := make([]byte, 16)
	_, err := rand.Read(b)
	if err != nil {
		// TODO handle error
		return
	}

	v.Set(reflect.ValueOf(UUIDVersion4(b)))
}

// UUIDVersion4 returns a Version 4 random UUID from the byte slice provided
func UUIDVersion4(u []byte) string {
	// https://en.wikipedia.org/wiki/Universally_unique_identifier#Version_4_.28random.29
	// 13th character is "4"
	u[6] = (u[6] | 0x40) & 0x4F
	// 17th character is "8", "9", "a", or "b"
	u[8] = (u[8] | 0x80) & 0xBF

	return fmt.Sprintf(`%X-%X-%X-%X-%X`, u[0:4], u[4:6], u[6:8], u[8:10], u[10:])
}
