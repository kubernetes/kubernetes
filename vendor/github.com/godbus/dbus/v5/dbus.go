package dbus

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
)

var (
	byteType        = reflect.TypeOf(byte(0))
	boolType        = reflect.TypeOf(false)
	uint8Type       = reflect.TypeOf(uint8(0))
	int16Type       = reflect.TypeOf(int16(0))
	uint16Type      = reflect.TypeOf(uint16(0))
	intType         = reflect.TypeOf(int(0))
	uintType        = reflect.TypeOf(uint(0))
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
	interfaceType   = reflect.TypeOf((*interface{})(nil)).Elem()
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
		if err := storeInterfaces(src[i], dest[i]); err != nil {
			return err
		}
	}
	return nil
}

func storeInterfaces(src, dest interface{}) error {
	return store(reflect.ValueOf(dest), reflect.ValueOf(src))
}

func store(dest, src reflect.Value) error {
	if dest.Kind() == reflect.Ptr {
		return store(dest.Elem(), src)
	}
	switch src.Kind() {
	case reflect.Slice:
		return storeSlice(dest, src)
	case reflect.Map:
		return storeMap(dest, src)
	default:
		return storeBase(dest, src)
	}
}

func storeBase(dest, src reflect.Value) error {
	return setDest(dest, src)
}

func setDest(dest, src reflect.Value) error {
	if !isVariant(src.Type()) && isVariant(dest.Type()) {
		//special conversion for dbus.Variant
		dest.Set(reflect.ValueOf(MakeVariant(src.Interface())))
		return nil
	}
	if isVariant(src.Type()) && !isVariant(dest.Type()) {
		src = getVariantValue(src)
		return store(dest, src)
	}
	if !src.Type().ConvertibleTo(dest.Type()) {
		return fmt.Errorf(
			"dbus.Store: type mismatch: cannot convert %s to %s",
			src.Type(), dest.Type())
	}
	dest.Set(src.Convert(dest.Type()))
	return nil
}

func kindsAreCompatible(dest, src reflect.Type) bool {
	switch {
	case isVariant(dest):
		return true
	case dest.Kind() == reflect.Interface:
		return true
	default:
		return dest.Kind() == src.Kind()
	}
}

func isConvertibleTo(dest, src reflect.Type) bool {
	switch {
	case isVariant(dest):
		return true
	case dest.Kind() == reflect.Interface:
		return true
	case dest.Kind() == reflect.Slice:
		return src.Kind() == reflect.Slice &&
			isConvertibleTo(dest.Elem(), src.Elem())
	case dest.Kind() == reflect.Struct:
		return src == interfacesType
	default:
		return src.ConvertibleTo(dest)
	}
}

func storeMap(dest, src reflect.Value) error {
	switch {
	case !kindsAreCompatible(dest.Type(), src.Type()):
		return fmt.Errorf(
			"dbus.Store: type mismatch: "+
				"map: cannot store a value of %s into %s",
			src.Type(), dest.Type())
	case isVariant(dest.Type()):
		return storeMapIntoVariant(dest, src)
	case dest.Kind() == reflect.Interface:
		return storeMapIntoInterface(dest, src)
	case isConvertibleTo(dest.Type().Key(), src.Type().Key()) &&
		isConvertibleTo(dest.Type().Elem(), src.Type().Elem()):
		return storeMapIntoMap(dest, src)
	default:
		return fmt.Errorf(
			"dbus.Store: type mismatch: "+
				"map: cannot convert a value of %s into %s",
			src.Type(), dest.Type())
	}
}

func storeMapIntoVariant(dest, src reflect.Value) error {
	dv := reflect.MakeMap(src.Type())
	err := store(dv, src)
	if err != nil {
		return err
	}
	return storeBase(dest, dv)
}

func storeMapIntoInterface(dest, src reflect.Value) error {
	var dv reflect.Value
	if isVariant(src.Type().Elem()) {
		//Convert variants to interface{} recursively when converting
		//to interface{}
		dv = reflect.MakeMap(
			reflect.MapOf(src.Type().Key(), interfaceType))
	} else {
		dv = reflect.MakeMap(src.Type())
	}
	err := store(dv, src)
	if err != nil {
		return err
	}
	return storeBase(dest, dv)
}

func storeMapIntoMap(dest, src reflect.Value) error {
	if dest.IsNil() {
		dest.Set(reflect.MakeMap(dest.Type()))
	}
	keys := src.MapKeys()
	for _, key := range keys {
		dkey := key.Convert(dest.Type().Key())
		dval := reflect.New(dest.Type().Elem()).Elem()
		err := store(dval, getVariantValue(src.MapIndex(key)))
		if err != nil {
			return err
		}
		dest.SetMapIndex(dkey, dval)
	}
	return nil
}

func storeSlice(dest, src reflect.Value) error {
	switch {
	case src.Type() == interfacesType && dest.Kind() == reflect.Struct:
		//The decoder always decodes structs as slices of interface{}
		return storeStruct(dest, src)
	case !kindsAreCompatible(dest.Type(), src.Type()):
		return fmt.Errorf(
			"dbus.Store: type mismatch: "+
				"slice: cannot store a value of %s into %s",
			src.Type(), dest.Type())
	case isVariant(dest.Type()):
		return storeSliceIntoVariant(dest, src)
	case dest.Kind() == reflect.Interface:
		return storeSliceIntoInterface(dest, src)
	case isConvertibleTo(dest.Type().Elem(), src.Type().Elem()):
		return storeSliceIntoSlice(dest, src)
	default:
		return fmt.Errorf(
			"dbus.Store: type mismatch: "+
				"slice: cannot convert a value of %s into %s",
			src.Type(), dest.Type())
	}
}

func storeStruct(dest, src reflect.Value) error {
	if isVariant(dest.Type()) {
		return storeBase(dest, src)
	}
	dval := make([]interface{}, 0, dest.NumField())
	dtype := dest.Type()
	for i := 0; i < dest.NumField(); i++ {
		field := dest.Field(i)
		ftype := dtype.Field(i)
		if ftype.PkgPath != "" {
			continue
		}
		if ftype.Tag.Get("dbus") == "-" {
			continue
		}
		dval = append(dval, field.Addr().Interface())
	}
	if src.Len() != len(dval) {
		return fmt.Errorf(
			"dbus.Store: type mismatch: "+
				"destination struct does not have "+
				"enough fields need: %d have: %d",
			src.Len(), len(dval))
	}
	return Store(src.Interface().([]interface{}), dval...)
}

func storeSliceIntoVariant(dest, src reflect.Value) error {
	dv := reflect.MakeSlice(src.Type(), src.Len(), src.Cap())
	err := store(dv, src)
	if err != nil {
		return err
	}
	return storeBase(dest, dv)
}

func storeSliceIntoInterface(dest, src reflect.Value) error {
	var dv reflect.Value
	if isVariant(src.Type().Elem()) {
		//Convert variants to interface{} recursively when converting
		//to interface{}
		dv = reflect.MakeSlice(reflect.SliceOf(interfaceType),
			src.Len(), src.Cap())
	} else {
		dv = reflect.MakeSlice(src.Type(), src.Len(), src.Cap())
	}
	err := store(dv, src)
	if err != nil {
		return err
	}
	return storeBase(dest, dv)
}

func storeSliceIntoSlice(dest, src reflect.Value) error {
	if dest.IsNil() || dest.Len() < src.Len() {
		dest.Set(reflect.MakeSlice(dest.Type(), src.Len(), src.Cap()))
	}
	if dest.Len() != src.Len() {
		return fmt.Errorf(
			"dbus.Store: type mismatch: "+
				"slices are different lengths "+
				"need: %d have: %d",
			src.Len(), dest.Len())
	}
	for i := 0; i < src.Len(); i++ {
		err := store(dest.Index(i), getVariantValue(src.Index(i)))
		if err != nil {
			return err
		}
	}
	return nil
}

func getVariantValue(in reflect.Value) reflect.Value {
	if isVariant(in.Type()) {
		return reflect.ValueOf(in.Interface().(Variant).Value())
	}
	return in
}

func isVariant(t reflect.Type) bool {
	return t == variantType
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
	case interfacesType:
		return 4
	}
	switch t.Kind() {
	case reflect.Uint8:
		return 1
	case reflect.Uint16, reflect.Int16:
		return 2
	case reflect.Uint, reflect.Int, reflect.Uint32, reflect.Int32, reflect.String, reflect.Array, reflect.Slice, reflect.Map:
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
		reflect.String, reflect.Uint, reflect.Int:

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
