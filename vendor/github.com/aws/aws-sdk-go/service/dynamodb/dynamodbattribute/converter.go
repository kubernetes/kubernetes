// Package dynamodbattribute provides conversion utilities from dynamodb.AttributeValue
// to concrete Go types and structures. These conversion utilities allow you to
// convert a Struct, Slice, Map, or Scalar value to or from dynamodb.AttributeValue.
// These are most useful to serialize concrete types to dynamodb.AttributeValue for
// requests or unmarshalling the dynamodb.AttributeValue into a well known typed form.
//
// Convert concrete type to dynamodb.AttributeValue: See (ExampleConvertTo)
//
//     type Record struct {
//         MyField string
//         Letters []string
//         A2Num   map[string]int
//     }
//
//     ...
//
//     r := Record{
//         MyField: "dynamodbattribute.ConvertToX example",
//         Letters: []string{"a", "b", "c", "d"},
//         A2Num:   map[string]int{"a": 1, "b": 2, "c": 3},
//     }
//     av, err := dynamodbattribute.ConvertTo(r)
//     fmt.Println(av, err)
//
// Convert dynamodb.AttributeValue to Concrete type: See (ExampleConvertFrom)
//
//     r2 := Record{}
//     err = dynamodbattribute.ConvertFrom(av, &r2)
//     fmt.Println(err, reflect.DeepEqual(r, r2))
//
// Use Conversion utilities with DynamoDB.PutItem: See ()
//
//     svc := dynamodb.New(nil)
//     item, err := dynamodbattribute.ConvertToMap(r)
//     if err != nil {
//         fmt.Println("Failed to convert", err)
//         return
//     }
//     result, err := svc.PutItem(&dynamodb.PutItemInput{
//         Item:      item,
//         TableName: aws.String("exampleTable"),
//     })
package dynamodbattribute

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"runtime"
	"strconv"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

// ConvertToMap accepts a map[string]interface{} or struct and converts it to a
// map[string]*dynamodb.AttributeValue.
//
// If in contains any structs, it is first JSON encoded/decoded it to convert it
// to a map[string]interface{}, so `json` struct tags are respected.
func ConvertToMap(in interface{}) (item map[string]*dynamodb.AttributeValue, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(runtime.Error); ok {
				err = e
			} else if s, ok := r.(string); ok {
				err = fmt.Errorf(s)
			} else {
				err = r.(error)
			}
			item = nil
		}
	}()

	if in == nil {
		return nil, awserr.New("SerializationError",
			"in must be a map[string]interface{} or struct, got <nil>", nil)
	}

	v := reflect.ValueOf(in)
	if v.Kind() != reflect.Struct && !(v.Kind() == reflect.Map && v.Type().Key().Kind() == reflect.String) {
		return nil, awserr.New("SerializationError",
			fmt.Sprintf("in must be a map[string]interface{} or struct, got %s",
				v.Type().String()),
			nil)
	}

	if isTyped(reflect.TypeOf(in)) {
		var out map[string]interface{}
		in = convertToUntyped(in, out)
	}

	item = make(map[string]*dynamodb.AttributeValue)
	for k, v := range in.(map[string]interface{}) {
		item[k] = convertTo(v)
	}

	return item, nil
}

// ConvertFromMap accepts a map[string]*dynamodb.AttributeValue and converts it to a
// map[string]interface{} or struct.
//
// If v points to a struct, the result is first converted it to a
// map[string]interface{}, then JSON encoded/decoded it to convert to a struct,
// so `json` struct tags are respected.
func ConvertFromMap(item map[string]*dynamodb.AttributeValue, v interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(runtime.Error); ok {
				err = e
			} else if s, ok := r.(string); ok {
				err = fmt.Errorf(s)
			} else {
				err = r.(error)
			}
			item = nil
		}
	}()

	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return awserr.New("SerializationError",
			fmt.Sprintf("v must be a non-nil pointer to a map[string]interface{} or struct, got %s",
				rv.Type()),
			nil)
	}
	if rv.Elem().Kind() != reflect.Struct && !(rv.Elem().Kind() == reflect.Map && rv.Elem().Type().Key().Kind() == reflect.String) {
		return awserr.New("SerializationError",
			fmt.Sprintf("v must be a non-nil pointer to a map[string]interface{} or struct, got %s",
				rv.Type()),
			nil)
	}

	m := make(map[string]interface{})
	for k, v := range item {
		m[k] = convertFrom(v)
	}

	if isTyped(reflect.TypeOf(v)) {
		err = convertToTyped(m, v)
	} else {
		rv.Elem().Set(reflect.ValueOf(m))
	}

	return err
}

// ConvertToList accepts an array or slice and converts it to a
// []*dynamodb.AttributeValue.
//
// If in contains any structs, it is first JSON encoded/decoded it to convert it
// to a []interface{}, so `json` struct tags are respected.
func ConvertToList(in interface{}) (item []*dynamodb.AttributeValue, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(runtime.Error); ok {
				err = e
			} else if s, ok := r.(string); ok {
				err = fmt.Errorf(s)
			} else {
				err = r.(error)
			}
			item = nil
		}
	}()

	if in == nil {
		return nil, awserr.New("SerializationError",
			"in must be an array or slice, got <nil>",
			nil)
	}

	v := reflect.ValueOf(in)
	if v.Kind() != reflect.Array && v.Kind() != reflect.Slice {
		return nil, awserr.New("SerializationError",
			fmt.Sprintf("in must be an array or slice, got %s",
				v.Type().String()),
			nil)
	}

	if isTyped(reflect.TypeOf(in)) {
		var out []interface{}
		in = convertToUntyped(in, out)
	}

	item = make([]*dynamodb.AttributeValue, 0, len(in.([]interface{})))
	for _, v := range in.([]interface{}) {
		item = append(item, convertTo(v))
	}

	return item, nil
}

// ConvertFromList accepts a []*dynamodb.AttributeValue and converts it to an array or
// slice.
//
// If v contains any structs, the result is first converted it to a
// []interface{}, then JSON encoded/decoded it to convert to a typed array or
// slice, so `json` struct tags are respected.
func ConvertFromList(item []*dynamodb.AttributeValue, v interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(runtime.Error); ok {
				err = e
			} else if s, ok := r.(string); ok {
				err = fmt.Errorf(s)
			} else {
				err = r.(error)
			}
			item = nil
		}
	}()

	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return awserr.New("SerializationError",
			fmt.Sprintf("v must be a non-nil pointer to an array or slice, got %s",
				rv.Type()),
			nil)
	}
	if rv.Elem().Kind() != reflect.Array && rv.Elem().Kind() != reflect.Slice {
		return awserr.New("SerializationError",
			fmt.Sprintf("v must be a non-nil pointer to an array or slice, got %s",
				rv.Type()),
			nil)
	}

	l := make([]interface{}, 0, len(item))
	for _, v := range item {
		l = append(l, convertFrom(v))
	}

	if isTyped(reflect.TypeOf(v)) {
		err = convertToTyped(l, v)
	} else {
		rv.Elem().Set(reflect.ValueOf(l))
	}

	return err
}

// ConvertTo accepts any interface{} and converts it to a *dynamodb.AttributeValue.
//
// If in contains any structs, it is first JSON encoded/decoded it to convert it
// to a interface{}, so `json` struct tags are respected.
func ConvertTo(in interface{}) (item *dynamodb.AttributeValue, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(runtime.Error); ok {
				err = e
			} else if s, ok := r.(string); ok {
				err = fmt.Errorf(s)
			} else {
				err = r.(error)
			}
			item = nil
		}
	}()

	if in != nil && isTyped(reflect.TypeOf(in)) {
		var out interface{}
		in = convertToUntyped(in, out)
	}

	item = convertTo(in)
	return item, nil
}

// ConvertFrom accepts a *dynamodb.AttributeValue and converts it to any interface{}.
//
// If v contains any structs, the result is first converted it to a interface{},
// then JSON encoded/decoded it to convert to a struct, so `json` struct tags
// are respected.
func ConvertFrom(item *dynamodb.AttributeValue, v interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(runtime.Error); ok {
				err = e
			} else if s, ok := r.(string); ok {
				err = fmt.Errorf(s)
			} else {
				err = r.(error)
			}
			item = nil
		}
	}()

	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return awserr.New("SerializationError",
			fmt.Sprintf("v must be a non-nil pointer to an interface{} or struct, got %s",
				rv.Type()),
			nil)
	}
	if rv.Elem().Kind() != reflect.Interface && rv.Elem().Kind() != reflect.Struct {
		return awserr.New("SerializationError",
			fmt.Sprintf("v must be a non-nil pointer to an interface{} or struct, got %s",
				rv.Type()),
			nil)
	}

	res := convertFrom(item)

	if isTyped(reflect.TypeOf(v)) {
		err = convertToTyped(res, v)
	} else if res != nil {
		rv.Elem().Set(reflect.ValueOf(res))
	}

	return err
}

func isTyped(v reflect.Type) bool {
	switch v.Kind() {
	case reflect.Struct:
		return true
	case reflect.Array, reflect.Slice:
		if isTyped(v.Elem()) {
			return true
		}
	case reflect.Map:
		if isTyped(v.Key()) {
			return true
		}
		if isTyped(v.Elem()) {
			return true
		}
	case reflect.Ptr:
		return isTyped(v.Elem())
	}
	return false
}

func convertToUntyped(in, out interface{}) interface{} {
	b, err := json.Marshal(in)
	if err != nil {
		panic(err)
	}

	decoder := json.NewDecoder(bytes.NewReader(b))
	decoder.UseNumber()
	err = decoder.Decode(&out)
	if err != nil {
		panic(err)
	}

	return out
}

func convertToTyped(in, out interface{}) error {
	b, err := json.Marshal(in)
	if err != nil {
		return err
	}

	decoder := json.NewDecoder(bytes.NewReader(b))
	return decoder.Decode(&out)
}

func convertTo(in interface{}) *dynamodb.AttributeValue {
	a := &dynamodb.AttributeValue{}

	if in == nil {
		a.NULL = new(bool)
		*a.NULL = true
		return a
	}

	if m, ok := in.(map[string]interface{}); ok {
		a.M = make(map[string]*dynamodb.AttributeValue)
		for k, v := range m {
			a.M[k] = convertTo(v)
		}
		return a
	}

	if l, ok := in.([]interface{}); ok {
		a.L = make([]*dynamodb.AttributeValue, len(l))
		for index, v := range l {
			a.L[index] = convertTo(v)
		}
		return a
	}

	// Only primitive types should remain.
	v := reflect.ValueOf(in)
	switch v.Kind() {
	case reflect.Bool:
		a.BOOL = new(bool)
		*a.BOOL = v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		a.N = new(string)
		*a.N = strconv.FormatInt(v.Int(), 10)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		a.N = new(string)
		*a.N = strconv.FormatUint(v.Uint(), 10)
	case reflect.Float32, reflect.Float64:
		a.N = new(string)
		*a.N = strconv.FormatFloat(v.Float(), 'f', -1, 64)
	case reflect.String:
		if n, ok := in.(json.Number); ok {
			a.N = new(string)
			*a.N = n.String()
		} else {
			a.S = new(string)
			*a.S = v.String()
		}
	default:
		panic(fmt.Sprintf("the type %s is not supported", v.Type().String()))
	}

	return a
}

func convertFrom(a *dynamodb.AttributeValue) interface{} {
	if a.S != nil {
		return *a.S
	}

	if a.N != nil {
		// Number is tricky b/c we don't know which numeric type to use. Here we
		// simply try the different types from most to least restrictive.
		if n, err := strconv.ParseInt(*a.N, 10, 64); err == nil {
			return int(n)
		}
		if n, err := strconv.ParseUint(*a.N, 10, 64); err == nil {
			return uint(n)
		}
		n, err := strconv.ParseFloat(*a.N, 64)
		if err != nil {
			panic(err)
		}
		return n
	}

	if a.BOOL != nil {
		return *a.BOOL
	}

	if a.NULL != nil {
		return nil
	}

	if a.M != nil {
		m := make(map[string]interface{})
		for k, v := range a.M {
			m[k] = convertFrom(v)
		}
		return m
	}

	if a.L != nil {
		l := make([]interface{}, len(a.L))
		for index, v := range a.L {
			l[index] = convertFrom(v)
		}
		return l
	}

	panic(fmt.Sprintf("%#v is not a supported dynamodb.AttributeValue", a))
}
