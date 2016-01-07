package dbus

import (
	"errors"
	"reflect"
	"strings"
)

var (
	byteType        = reflect.TypeOf(byte(0))
	boolType        = reflect.TypeOf(false)
	uint8Type       = reflect.TypeOf(uint8(0))
	int16Type       = reflect.TypeOf(int16(0))
	uint16Type      = reflect.TypeOf(uint16(0))
	int32Type       = reflect.TypeOf(int32(0))
	uint32Type      = reflect.TypeOf(uint32(0))
	int64Type       = reflect.TypeOf(int64(0))
	uint64Type      = reflect.TypeOf(uint64(0))
	float64Type     = reflect.TypeOf(float64(0))
	stringType      = reflect.TypeOf("")
	signatureType   = reflect.TypeOf(Signature{""})
	objectPathType  = reflect.TypeOf(ObjectPath(""))
	variantType     = reflect.TypeOf(Variant{Signature{""}, nil})
	interfacesType  = reflect.TypeOf([]interface{}{})
	unixFDType      = reflect.TypeOf(UnixFD(0))
	unixFDIndexType = reflect.TypeOf(UnixFDIndex(0))
)

// An InvalidTypeError signals that a value which cannot be represented in the
// D-Bus wire format was passed to a function.
type InvalidTypeError struct {
	Type reflect.Type
}

func (e InvalidTypeError) Error() string {
	return "dbus: invalid type " + e.Type.String()
}

// Store copies the values contained in src to dest, which must be a slice of
// pointers. It converts slices of interfaces from src to corresponding structs
// in dest. An error is returned if the lengths of src and dest or the types of
// their elements don't match.
func Store(src []interface{}, dest ...interface{}) error {
	if len(src) != len(dest) {
		return errors.New("dbus.Store: length mismatch")
	}

	for i := range src {
		if err := store(src[i], dest[i]); err != nil {
			return err
		}
	}
	return nil
}

func store(src, dest interface{}) error {
	if reflect.TypeOf(dest).Elem() == reflect.TypeOf(src) {
		reflect.ValueOf(dest).Elem().Set(reflect.ValueOf(src))
		return nil
	} else if hasStruct(dest) {
		rv := reflect.ValueOf(dest).Elem()
		switch rv.Kind() {
		case reflect.Struct:
			vs, ok := src.([]interface{})
			if !ok {
				return errors.New("dbus.Store: type mismatch")
			}
			t := rv.Type()
			ndest := make([]interface{}, 0, rv.NumField())
			for i := 0; i < rv.NumField(); i++ {
				field := t.Field(i)
				if field.PkgPath == "" && field.Tag.Get("dbus") != "-" {
					ndest = append(ndest, rv.Field(i).Addr().Interface())
				}
			}
			if len(vs) != len(ndest) {
				return errors.New("dbus.Store: type mismatch")
			}
			err := Store(vs, ndest...)
			if err != nil {
				return errors.New("dbus.Store: type mismatch")
			}
		case reflect.Slice:
			sv := reflect.ValueOf(src)
			if sv.Kind() != reflect.Slice {
				return errors.New("dbus.Store: type mismatch")
			}
			rv.Set(reflect.MakeSlice(rv.Type(), sv.Len(), sv.Len()))
			for i := 0; i < sv.Len(); i++ {
				if err := store(sv.Index(i).Interface(), rv.Index(i).Addr().Interface()); err != nil {
					return err
				}
			}
		case reflect.Map:
			sv := reflect.ValueOf(src)
			if sv.Kind() != reflect.Map {
				return errors.New("dbus.Store: type mismatch")
			}
			keys := sv.MapKeys()
			rv.Set(reflect.MakeMap(sv.Type()))
			for _, key := range keys {
				v := reflect.New(sv.Type().Elem())
				if err := store(v, sv.MapIndex(key).Interface()); err != nil {
					return err
				}
				rv.SetMapIndex(key, v.Elem())
			}
		default:
			return errors.New("dbus.Store: type mismatch")
		}
		return nil
	} else {
		return errors.New("dbus.Store: type mismatch")
	}
}

func hasStruct(v interface{}) bool {
	t := reflect.TypeOf(v)
	for {
		switch t.Kind() {
		case reflect.Struct:
			return true
		case reflect.Slice, reflect.Ptr, reflect.Map:
			t = t.Elem()
		default:
			return false
		}
	}
}

// An ObjectPath is an object path as defined by the D-Bus spec.
type ObjectPath string

// IsValid returns whether the object path is valid.
func (o ObjectPath) IsValid() bool {
	s := string(o)
	if len(s) == 0 {
		return false
	}
	if s[0] != '/' {
		return false
	}
	if s[len(s)-1] == '/' && len(s) != 1 {
		return false
	}
	// probably not used, but technically possible
	if s == "/" {
		return true
	}
	split := strings.Split(s[1:], "/")
	for _, v := range split {
		if len(v) == 0 {
			return false
		}
		for _, c := range v {
			if !isMemberChar(c) {
				return false
			}
		}
	}
	return true
}

// A UnixFD is a Unix file descriptor sent over the wire. See the package-level
// documentation for more information about Unix file descriptor passsing.
type UnixFD int32

// A UnixFDIndex is the representation of a Unix file descriptor in a message.
type UnixFDIndex uint32

// alignment returns the alignment of values of type t.
func alignment(t reflect.Type) int {
	switch t {
	case variantType:
		return 1
	case objectPathType:
		return 4
	case signatureType:
		return 1
	case interfacesType: // sometimes used for structs
		return 8
	}
	switch t.Kind() {
	case reflect.Uint8:
		return 1
	case reflect.Uint16, reflect.Int16:
		return 2
	case reflect.Uint32, reflect.Int32, reflect.String, reflect.Array, reflect.Slice, reflect.Map:
		return 4
	case reflect.Uint64, reflect.Int64, reflect.Float64, reflect.Struct:
		return 8
	case reflect.Ptr:
		return alignment(t.Elem())
	}
	return 1
}

// isKeyType returns whether t is a valid type for a D-Bus dict.
func isKeyType(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Int16, reflect.Int32, reflect.Int64, reflect.Float64,
		reflect.String:

		return true
	}
	return false
}

// isValidInterface returns whether s is a valid name for an interface.
func isValidInterface(s string) bool {
	if len(s) == 0 || len(s) > 255 || s[0] == '.' {
		return false
	}
	elem := strings.Split(s, ".")
	if len(elem) < 2 {
		return false
	}
	for _, v := range elem {
		if len(v) == 0 {
			return false
		}
		if v[0] >= '0' && v[0] <= '9' {
			return false
		}
		for _, c := range v {
			if !isMemberChar(c) {
				return false
			}
		}
	}
	return true
}

// isValidMember returns whether s is a valid name for a member.
func isValidMember(s string) bool {
	if len(s) == 0 || len(s) > 255 {
		return false
	}
	i := strings.Index(s, ".")
	if i != -1 {
		return false
	}
	if s[0] >= '0' && s[0] <= '9' {
		return false
	}
	for _, c := range s {
		if !isMemberChar(c) {
			return false
		}
	}
	return true
}

func isMemberChar(c rune) bool {
	return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
		(c >= 'a' && c <= 'z') || c == '_'
}
