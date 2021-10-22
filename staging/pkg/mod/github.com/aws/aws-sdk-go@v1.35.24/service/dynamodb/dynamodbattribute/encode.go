package dynamodbattribute

import (
	"fmt"
	"reflect"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

// An UnixTime provides aliasing of time.Time into a type that when marshaled
// and unmarshaled with DynamoDB AttributeValues it will be done so as number
// instead of string in seconds since January 1, 1970 UTC.
//
// This type is useful as an alternative to the struct tag `unixtime` when you
// want to have your time value marshaled as Unix time in seconds intead of
// the default time.RFC3339.
//
// Important to note that zero value time as unixtime is not 0 seconds
// from January 1, 1970 UTC, but -62135596800. Which is seconds between
// January 1, 0001 UTC, and January 1, 0001 UTC.
type UnixTime time.Time

// MarshalDynamoDBAttributeValue implements the Marshaler interface so that
// the UnixTime can be marshaled from to a DynamoDB AttributeValue number
// value encoded in the number of seconds since January 1, 1970 UTC.
func (e UnixTime) MarshalDynamoDBAttributeValue(av *dynamodb.AttributeValue) error {
	t := time.Time(e)
	s := strconv.FormatInt(t.Unix(), 10)
	av.N = &s

	return nil
}

// UnmarshalDynamoDBAttributeValue implements the Unmarshaler interface so that
// the UnixTime can be unmarshaled from a DynamoDB AttributeValue number representing
// the number of seconds since January 1, 1970 UTC.
//
// If an error parsing the AttributeValue number occurs UnmarshalError will be
// returned.
func (e *UnixTime) UnmarshalDynamoDBAttributeValue(av *dynamodb.AttributeValue) error {
	t, err := decodeUnixTime(aws.StringValue(av.N))
	if err != nil {
		return err
	}

	*e = UnixTime(t)
	return nil
}

// A Marshaler is an interface to provide custom marshaling of Go value types
// to AttributeValues. Use this to provide custom logic determining how a
// Go Value type should be marshaled.
//
//		type ExampleMarshaler struct {
//			Value int
//		}
//		func (m *ExampleMarshaler) 	MarshalDynamoDBAttributeValue(av *dynamodb.AttributeValue) error {
//			n := fmt.Sprintf("%v", m.Value)
//			av.N = &n
//			return nil
//		}
//
type Marshaler interface {
	MarshalDynamoDBAttributeValue(*dynamodb.AttributeValue) error
}

// Marshal will serialize the passed in Go value type into a DynamoDB AttributeValue
// type. This value can be used in DynamoDB API operations to simplify marshaling
// your Go value types into AttributeValues.
//
// Marshal will recursively transverse the passed in value marshaling its
// contents into a AttributeValue. Marshal supports basic scalars
// (int,uint,float,bool,string), maps, slices, and structs. Anonymous
// nested types are flattened based on Go anonymous type visibility.
//
// Marshaling slices to AttributeValue will default to a List for all
// types except for []byte and [][]byte. []byte will be marshaled as
// Binary data (B), and [][]byte will be marshaled as binary data set
// (BS).
//
// `dynamodbav` struct tag can be used to control how the value will be
// marshaled into a AttributeValue.
//
//		// Field is ignored
//		Field int `dynamodbav:"-"`
//
//		// Field AttributeValue map key "myName"
//		Field int `dynamodbav:"myName"`
//
//		// Field AttributeValue map key "myName", and
//		// Field is omitted if it is empty
//		Field int `dynamodbav:"myName,omitempty"`
//
//		// Field AttributeValue map key "Field", and
//		// Field is omitted if it is empty
//		Field int `dynamodbav:",omitempty"`
//
//		// Field's elems will be omitted if empty
//		// only valid for slices, and maps.
//		Field []string `dynamodbav:",omitemptyelem"`
//
//		// Field will be marshaled as a AttributeValue string
//		// only value for number types, (int,uint,float)
//		Field int `dynamodbav:",string"`
//
//		// Field will be marshaled as a binary set
//		Field [][]byte `dynamodbav:",binaryset"`
//
//		// Field will be marshaled as a number set
//		Field []int `dynamodbav:",numberset"`
//
//		// Field will be marshaled as a string set
//		Field []string `dynamodbav:",stringset"`
//
//		// Field will be marshaled as Unix time number in seconds.
//		// This tag is only valid with time.Time typed struct fields.
//		// Important to note that zero value time as unixtime is not 0 seconds
//		// from January 1, 1970 UTC, but -62135596800. Which is seconds between
//		// January 1, 0001 UTC, and January 1, 0001 UTC.
//		Field time.Time `dynamodbav:",unixtime"`
//
// The omitempty tag is only used during Marshaling and is ignored for
// Unmarshal. Any zero value or a value when marshaled results in a
// AttributeValue NULL will be added to AttributeValue Maps during struct
// marshal. The omitemptyelem tag works the same as omitempty except it
// applies to maps and slices instead of struct fields, and will not be
// included in the marshaled AttributeValue Map, List, or Set.
//
// For convenience and backwards compatibility with ConvertTo functions
// json struct tags are supported by the Marshal and Unmarshal. If
// both json and dynamodbav struct tags are provided the json tag will
// be ignored in favor of dynamodbav.
//
// All struct fields and with anonymous fields, are marshaled unless the
// any of the following conditions are meet.
//
//		- the field is not exported
//		- json or dynamodbav field tag is "-"
//		- json or dynamodbav field tag specifies "omitempty", and is empty.
//
// Pointer and interfaces values encode as the value pointed to or contained
// in the interface. A nil value encodes as the AttributeValue NULL value.
//
// Channel, complex, and function values are not encoded and will be skipped
// when walking the value to be marshaled.
//
// When marshaling any error that occurs will halt the marshal and return
// the error.
//
// Marshal cannot represent cyclic data structures and will not handle them.
// Passing cyclic structures to Marshal will result in an infinite recursion.
func Marshal(in interface{}) (*dynamodb.AttributeValue, error) {
	return NewEncoder().Encode(in)
}

// MarshalMap is an alias for Marshal func which marshals Go value
// type to a map of AttributeValues.
//
// This is useful for DynamoDB APIs such as PutItem.
func MarshalMap(in interface{}) (map[string]*dynamodb.AttributeValue, error) {
	av, err := NewEncoder().Encode(in)
	if err != nil || av == nil || av.M == nil {
		return map[string]*dynamodb.AttributeValue{}, err
	}

	return av.M, nil
}

// MarshalList is an alias for Marshal func which marshals Go value
// type to a slice of AttributeValues.
func MarshalList(in interface{}) ([]*dynamodb.AttributeValue, error) {
	av, err := NewEncoder().Encode(in)
	if err != nil || av == nil || av.L == nil {
		return []*dynamodb.AttributeValue{}, err
	}

	return av.L, nil
}

// A MarshalOptions is a collection of options shared between marshaling
// and unmarshaling
type MarshalOptions struct {
	// States that the encoding/json struct tags should be supported.
	// if a `dynamodbav` struct tag is also provided the encoding/json
	// tag will be ignored.
	//
	// Enabled by default.
	SupportJSONTags bool

	// Support other custom struct tag keys, such as `yaml` or `toml`.
	// Note that values provided with a custom TagKey must also be supported
	// by the (un)marshalers in this package.
	TagKey string

	// EnableEmptyCollections modifies how structures, maps, and slices are (un)marshalled.
	// When set to true empty collection values will be preserved as their respective
	// empty DynamoDB AttributeValue type when set to true.
	//
	// Disabled by default.
	EnableEmptyCollections bool
}

// An Encoder provides marshaling Go value types to AttributeValues.
type Encoder struct {
	MarshalOptions

	// Empty strings, "", will be marked as NULL AttributeValue types.
	// Will not apply to lists, sets, or maps. Use the struct tag `omitemptyelem`
	// to skip empty (zero) values in lists, sets and maps.
	//
	// Enabled by default.
	NullEmptyString bool

	// Empty byte slices, len([]byte{}) == 0, will be marked as NULL AttributeValue types.
	// Will not apply to lists, sets, or maps. Use the struct tag `omitemptyelem`
	// to skip empty (zero) values in lists, sets and maps.
	//
	// Enabled by default.
	NullEmptyByteSlice bool
}

// NewEncoder creates a new Encoder with default configuration. Use
// the `opts` functional options to override the default configuration.
func NewEncoder(opts ...func(*Encoder)) *Encoder {
	e := &Encoder{
		MarshalOptions: MarshalOptions{
			SupportJSONTags: true,
		},
		NullEmptyString:    true,
		NullEmptyByteSlice: true,
	}
	for _, o := range opts {
		o(e)
	}

	return e
}

// Encode will marshal a Go value type to an AttributeValue. Returning
// the AttributeValue constructed or error.
func (e *Encoder) Encode(in interface{}) (*dynamodb.AttributeValue, error) {
	av := &dynamodb.AttributeValue{}
	if err := e.encode(av, reflect.ValueOf(in), tag{}); err != nil {
		return nil, err
	}

	return av, nil
}

func (e *Encoder) encode(av *dynamodb.AttributeValue, v reflect.Value, fieldTag tag) error {
	// We should check for omitted values first before dereferencing.
	if fieldTag.OmitEmpty && emptyValue(v, e.EnableEmptyCollections) {
		encodeNull(av)
		return nil
	}

	// Handle both pointers and interface conversion into types
	v = valueElem(v)

	if v.Kind() != reflect.Invalid {
		if used, err := tryMarshaler(av, v); used {
			return err
		}
	}

	switch v.Kind() {
	case reflect.Invalid:
		encodeNull(av)
	case reflect.Struct:
		return e.encodeStruct(av, v, fieldTag)
	case reflect.Map:
		return e.encodeMap(av, v, fieldTag)
	case reflect.Slice, reflect.Array:
		return e.encodeSlice(av, v, fieldTag)
	case reflect.Chan, reflect.Func, reflect.UnsafePointer:
		// do nothing for unsupported types
	default:
		return e.encodeScalar(av, v, fieldTag)
	}

	return nil
}

func (e *Encoder) encodeStruct(av *dynamodb.AttributeValue, v reflect.Value, fieldTag tag) error {
	// To maintain backwards compatibility with ConvertTo family of methods which
	// converted time.Time structs to strings
	if v.Type().ConvertibleTo(timeType) {
		var t time.Time
		t = v.Convert(timeType).Interface().(time.Time)
		if fieldTag.AsUnixTime {
			return UnixTime(t).MarshalDynamoDBAttributeValue(av)
		}
		s := t.Format(time.RFC3339Nano)
		av.S = &s
		return nil
	}

	av.M = map[string]*dynamodb.AttributeValue{}
	fields := unionStructFields(v.Type(), e.MarshalOptions)
	for _, f := range fields.All() {
		if f.Name == "" {
			return &InvalidMarshalError{msg: "map key cannot be empty"}
		}

		fv, found := encoderFieldByIndex(v, f.Index)
		if !found {
			continue
		}
		elem := &dynamodb.AttributeValue{}
		err := e.encode(elem, fv, f.tag)
		if err != nil {
			return err
		}
		skip, err := keepOrOmitEmpty(f.OmitEmpty, elem, err)
		if err != nil {
			return err
		} else if skip {
			continue
		}

		av.M[f.Name] = elem
	}
	if len(av.M) == 0 && !e.EnableEmptyCollections {
		encodeNull(av)
	}

	return nil
}

func (e *Encoder) encodeMap(av *dynamodb.AttributeValue, v reflect.Value, fieldTag tag) error {
	av.M = map[string]*dynamodb.AttributeValue{}
	for _, key := range v.MapKeys() {
		keyName := fmt.Sprint(key.Interface())
		if keyName == "" {
			return &InvalidMarshalError{msg: "map key cannot be empty"}
		}

		elemVal := v.MapIndex(key)
		elem := &dynamodb.AttributeValue{}
		err := e.encode(elem, elemVal, tag{})
		skip, err := keepOrOmitEmpty(fieldTag.OmitEmptyElem, elem, err)
		if err != nil {
			return err
		} else if skip {
			continue
		}

		av.M[keyName] = elem
	}

	if v.IsNil() || (len(av.M) == 0 && !e.EnableEmptyCollections) {
		encodeNull(av)
	}

	return nil
}

func (e *Encoder) encodeSlice(av *dynamodb.AttributeValue, v reflect.Value, fieldTag tag) error {
	if v.Kind() == reflect.Array && v.Len() == 0 && e.EnableEmptyCollections && fieldTag.OmitEmpty {
		encodeNull(av)
		return nil
	}

	switch v.Type().Elem().Kind() {
	case reflect.Uint8:
		slice := reflect.MakeSlice(byteSliceType, v.Len(), v.Len())
		reflect.Copy(slice, v)

		b := slice.Bytes()
		if (v.Kind() == reflect.Slice && v.IsNil()) || (len(b) == 0 && !e.EnableEmptyCollections && e.NullEmptyByteSlice) {
			encodeNull(av)
			return nil
		}
		av.B = append([]byte{}, b...)
	default:
		var elemFn func(dynamodb.AttributeValue) error

		if fieldTag.AsBinSet || v.Type() == byteSliceSlicetype { // Binary Set
			av.BS = make([][]byte, 0, v.Len())
			elemFn = func(elem dynamodb.AttributeValue) error {
				if elem.B == nil {
					return &InvalidMarshalError{msg: "binary set must only contain non-nil byte slices"}
				}
				av.BS = append(av.BS, elem.B)
				return nil
			}
		} else if fieldTag.AsNumSet { // Number Set
			av.NS = make([]*string, 0, v.Len())
			elemFn = func(elem dynamodb.AttributeValue) error {
				if elem.N == nil {
					return &InvalidMarshalError{msg: "number set must only contain non-nil string numbers"}
				}
				av.NS = append(av.NS, elem.N)
				return nil
			}
		} else if fieldTag.AsStrSet { // String Set
			av.SS = make([]*string, 0, v.Len())
			elemFn = func(elem dynamodb.AttributeValue) error {
				if elem.S == nil {
					return &InvalidMarshalError{msg: "string set must only contain non-nil strings"}
				}
				av.SS = append(av.SS, elem.S)
				return nil
			}
		} else { // List
			av.L = make([]*dynamodb.AttributeValue, 0, v.Len())
			elemFn = func(elem dynamodb.AttributeValue) error {
				av.L = append(av.L, &elem)
				return nil
			}
		}

		if n, err := e.encodeList(v, fieldTag, elemFn); err != nil {
			return err
		} else if (v.Kind() == reflect.Slice && v.IsNil()) || (n == 0 && !e.EnableEmptyCollections) {
			encodeNull(av)
		}
	}

	return nil
}

func (e *Encoder) encodeList(v reflect.Value, fieldTag tag, elemFn func(dynamodb.AttributeValue) error) (int, error) {
	count := 0
	for i := 0; i < v.Len(); i++ {
		elem := dynamodb.AttributeValue{}
		err := e.encode(&elem, v.Index(i), tag{OmitEmpty: fieldTag.OmitEmptyElem})
		skip, err := keepOrOmitEmpty(fieldTag.OmitEmptyElem, &elem, err)
		if err != nil {
			return 0, err
		} else if skip {
			continue
		}

		if err := elemFn(elem); err != nil {
			return 0, err
		}
		count++
	}

	return count, nil
}

func (e *Encoder) encodeScalar(av *dynamodb.AttributeValue, v reflect.Value, fieldTag tag) error {
	if v.Type() == numberType {
		s := v.String()
		if fieldTag.AsString {
			av.S = &s
		} else {
			av.N = &s
		}
		return nil
	}

	switch v.Kind() {
	case reflect.Bool:
		av.BOOL = new(bool)
		*av.BOOL = v.Bool()
	case reflect.String:
		if err := e.encodeString(av, v); err != nil {
			return err
		}
	default:
		// Fallback to encoding numbers, will return invalid type if not supported
		if err := e.encodeNumber(av, v); err != nil {
			return err
		}
		if fieldTag.AsString && av.NULL == nil && av.N != nil {
			av.S = av.N
			av.N = nil
		}
	}

	return nil
}

func (e *Encoder) encodeNumber(av *dynamodb.AttributeValue, v reflect.Value) error {
	if used, err := tryMarshaler(av, v); used {
		return err
	}

	var out string
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		out = encodeInt(v.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		out = encodeUint(v.Uint())
	case reflect.Float32:
		out = encodeFloat(v.Float(), 32)
	case reflect.Float64:
		out = encodeFloat(v.Float(), 64)
	default:
		return &unsupportedMarshalTypeError{Type: v.Type()}
	}

	av.N = &out

	return nil
}

func (e *Encoder) encodeString(av *dynamodb.AttributeValue, v reflect.Value) error {
	if used, err := tryMarshaler(av, v); used {
		return err
	}

	switch v.Kind() {
	case reflect.String:
		s := v.String()
		if len(s) == 0 && e.NullEmptyString {
			encodeNull(av)
		} else {
			av.S = &s
		}
	default:
		return &unsupportedMarshalTypeError{Type: v.Type()}
	}

	return nil
}

func encodeInt(i int64) string {
	return strconv.FormatInt(i, 10)
}
func encodeUint(u uint64) string {
	return strconv.FormatUint(u, 10)
}
func encodeFloat(f float64, bitSize int) string {
	return strconv.FormatFloat(f, 'f', -1, bitSize)
}
func encodeNull(av *dynamodb.AttributeValue) {
	t := true
	*av = dynamodb.AttributeValue{NULL: &t}
}

// encoderFieldByIndex finds the field with the provided nested index
func encoderFieldByIndex(v reflect.Value, index []int) (reflect.Value, bool) {
	for i, x := range index {
		if i > 0 && v.Kind() == reflect.Ptr && v.Type().Elem().Kind() == reflect.Struct {
			if v.IsNil() {
				return reflect.Value{}, false
			}
			v = v.Elem()
		}
		v = v.Field(x)
	}
	return v, true
}

func valueElem(v reflect.Value) reflect.Value {
	switch v.Kind() {
	case reflect.Interface, reflect.Ptr:
		for v.Kind() == reflect.Interface || v.Kind() == reflect.Ptr {
			v = v.Elem()
		}
	}

	return v
}

func emptyValue(v reflect.Value, emptyCollections bool) bool {
	switch v.Kind() {
	case reflect.Array:
		return v.Len() == 0 && !emptyCollections
	case reflect.Map, reflect.Slice:
		return v.IsNil() || (v.Len() == 0 && !emptyCollections)
	case reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}

func tryMarshaler(av *dynamodb.AttributeValue, v reflect.Value) (bool, error) {
	if v.Kind() != reflect.Ptr && v.Type().Name() != "" && v.CanAddr() {
		v = v.Addr()
	}

	if v.Type().NumMethod() == 0 {
		return false, nil
	}

	if m, ok := v.Interface().(Marshaler); ok {
		return true, m.MarshalDynamoDBAttributeValue(av)
	}

	return false, nil
}

func keepOrOmitEmpty(omitEmpty bool, av *dynamodb.AttributeValue, err error) (bool, error) {
	if err != nil {
		if _, ok := err.(*unsupportedMarshalTypeError); ok {
			return true, nil
		}
		return false, err
	}

	if av.NULL != nil && omitEmpty {
		return true, nil
	}

	return false, nil
}

// An InvalidMarshalError is an error type representing an error
// occurring when marshaling a Go value type to an AttributeValue.
type InvalidMarshalError struct {
	emptyOrigError
	msg string
}

// Error returns the string representation of the error.
// satisfying the error interface
func (e *InvalidMarshalError) Error() string {
	return fmt.Sprintf("%s: %s", e.Code(), e.Message())
}

// Code returns the code of the error, satisfying the awserr.Error
// interface.
func (e *InvalidMarshalError) Code() string {
	return "InvalidMarshalError"
}

// Message returns the detailed message of the error, satisfying
// the awserr.Error interface.
func (e *InvalidMarshalError) Message() string {
	return e.msg
}

// An unsupportedMarshalTypeError represents a Go value type
// which cannot be marshaled into an AttributeValue and should
// be skipped by the marshaler.
type unsupportedMarshalTypeError struct {
	emptyOrigError
	Type reflect.Type
}

// Error returns the string representation of the error.
// satisfying the error interface
func (e *unsupportedMarshalTypeError) Error() string {
	return fmt.Sprintf("%s: %s", e.Code(), e.Message())
}

// Code returns the code of the error, satisfying the awserr.Error
// interface.
func (e *unsupportedMarshalTypeError) Code() string {
	return "unsupportedMarshalTypeError"
}

// Message returns the detailed message of the error, satisfying
// the awserr.Error interface.
func (e *unsupportedMarshalTypeError) Message() string {
	return "Go value type " + e.Type.String() + " is not supported"
}
