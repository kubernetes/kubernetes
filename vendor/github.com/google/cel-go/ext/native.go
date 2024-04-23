// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ext

import (
	"fmt"
	"reflect"
	"strings"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/pb"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	structpb "google.golang.org/protobuf/types/known/structpb"
)

var (
	nativeObjTraitMask = traits.FieldTesterType | traits.IndexerType
	jsonValueType      = reflect.TypeOf(&structpb.Value{})
	jsonStructType     = reflect.TypeOf(&structpb.Struct{})
)

// NativeTypes creates a type provider which uses reflect.Type and reflect.Value instances
// to produce type definitions that can be used within CEL.
//
// All struct types in Go are exposed to CEL via their simple package name and struct type name:
//
// ```go
// package identity
//
//	type Account struct {
//	  ID int
//	}
//
// ```
//
// The type `identity.Account` would be exported to CEL using the same qualified name, e.g.
// `identity.Account{ID: 1234}` would create a new `Account` instance with the `ID` field
// populated.
//
// Only exported fields are exposed via NativeTypes, and the type-mapping between Go and CEL
// is as follows:
//
// | Go type                             | CEL type  |
// |-------------------------------------|-----------|
// | bool                                | bool      |
// | []byte                              | bytes     |
// | float32, float64                    | double    |
// | int, int8, int16, int32, int64      | int       |
// | string                              | string    |
// | uint, uint8, uint16, uint32, uint64 | uint      |
// | time.Duration                       | duration  |
// | time.Time                           | timestamp |
// | array, slice                        | list      |
// | map                                 | map       |
//
// Please note, if you intend to configure support for proto messages in addition to native
// types, you will need to provide the protobuf types before the golang native types. The
// same advice holds if you are using custom type adapters and type providers. The native type
// provider composes over whichever type adapter and provider is configured in the cel.Env at
// the time that it is invoked.
func NativeTypes(refTypes ...any) cel.EnvOption {
	return func(env *cel.Env) (*cel.Env, error) {
		tp, err := newNativeTypeProvider(env.CELTypeAdapter(), env.CELTypeProvider(), refTypes...)
		if err != nil {
			return nil, err
		}
		env, err = cel.CustomTypeAdapter(tp)(env)
		if err != nil {
			return nil, err
		}
		return cel.CustomTypeProvider(tp)(env)
	}
}

func newNativeTypeProvider(adapter types.Adapter, provider types.Provider, refTypes ...any) (*nativeTypeProvider, error) {
	nativeTypes := make(map[string]*nativeType, len(refTypes))
	for _, refType := range refTypes {
		switch rt := refType.(type) {
		case reflect.Type:
			result, err := newNativeTypes(rt)
			if err != nil {
				return nil, err
			}
			for idx := range result {
				nativeTypes[result[idx].TypeName()] = result[idx]
			}
		case reflect.Value:
			result, err := newNativeTypes(rt.Type())
			if err != nil {
				return nil, err
			}
			for idx := range result {
				nativeTypes[result[idx].TypeName()] = result[idx]
			}
		default:
			return nil, fmt.Errorf("unsupported native type: %v (%T) must be reflect.Type or reflect.Value", rt, rt)
		}
	}
	return &nativeTypeProvider{
		nativeTypes:  nativeTypes,
		baseAdapter:  adapter,
		baseProvider: provider,
	}, nil
}

type nativeTypeProvider struct {
	nativeTypes  map[string]*nativeType
	baseAdapter  types.Adapter
	baseProvider types.Provider
}

// EnumValue proxies to the types.Provider configured at the times the NativeTypes
// option was configured.
func (tp *nativeTypeProvider) EnumValue(enumName string) ref.Val {
	return tp.baseProvider.EnumValue(enumName)
}

// FindIdent looks up natives type instances by qualified identifier, and if not found
// proxies to the composed types.Provider.
func (tp *nativeTypeProvider) FindIdent(typeName string) (ref.Val, bool) {
	if t, found := tp.nativeTypes[typeName]; found {
		return t, true
	}
	return tp.baseProvider.FindIdent(typeName)
}

// FindStructType looks up the CEL type definition by qualified identifier, and if not found
// proxies to the composed types.Provider.
func (tp *nativeTypeProvider) FindStructType(typeName string) (*types.Type, bool) {
	if _, found := tp.nativeTypes[typeName]; found {
		return types.NewTypeTypeWithParam(types.NewObjectType(typeName)), true
	}
	if celType, found := tp.baseProvider.FindStructType(typeName); found {
		return celType, true
	}
	return tp.baseProvider.FindStructType(typeName)
}

// FindStructFieldNames looks up the type definition first from the native types, then from
// the backing provider type set. If found, a set of field names corresponding to the type
// will be returned.
func (tp *nativeTypeProvider) FindStructFieldNames(typeName string) ([]string, bool) {
	if t, found := tp.nativeTypes[typeName]; found {
		fieldCount := t.refType.NumField()
		fields := make([]string, fieldCount)
		for i := 0; i < fieldCount; i++ {
			fields[i] = t.refType.Field(i).Name
		}
		return fields, true
	}
	if celTypeFields, found := tp.baseProvider.FindStructFieldNames(typeName); found {
		return celTypeFields, true
	}
	return tp.baseProvider.FindStructFieldNames(typeName)
}

// FindStructFieldType looks up a native type's field definition, and if the type name is not a native
// type then proxies to the composed types.Provider
func (tp *nativeTypeProvider) FindStructFieldType(typeName, fieldName string) (*types.FieldType, bool) {
	t, found := tp.nativeTypes[typeName]
	if !found {
		return tp.baseProvider.FindStructFieldType(typeName, fieldName)
	}
	refField, isDefined := t.hasField(fieldName)
	if !found || !isDefined {
		return nil, false
	}
	celType, ok := convertToCelType(refField.Type)
	if !ok {
		return nil, false
	}
	return &types.FieldType{
		Type: celType,
		IsSet: func(obj any) bool {
			refVal := reflect.Indirect(reflect.ValueOf(obj))
			refField := refVal.FieldByName(fieldName)
			return !refField.IsZero()
		},
		GetFrom: func(obj any) (any, error) {
			refVal := reflect.Indirect(reflect.ValueOf(obj))
			refField := refVal.FieldByName(fieldName)
			return getFieldValue(tp, refField), nil
		},
	}, true
}

// NewValue implements the ref.TypeProvider interface method.
func (tp *nativeTypeProvider) NewValue(typeName string, fields map[string]ref.Val) ref.Val {
	t, found := tp.nativeTypes[typeName]
	if !found {
		return tp.baseProvider.NewValue(typeName, fields)
	}
	refPtr := reflect.New(t.refType)
	refVal := refPtr.Elem()
	for fieldName, val := range fields {
		refFieldDef, isDefined := t.hasField(fieldName)
		if !isDefined {
			return types.NewErr("no such field: %s", fieldName)
		}
		fieldVal, err := val.ConvertToNative(refFieldDef.Type)
		if err != nil {
			return types.NewErr(err.Error())
		}
		refField := refVal.FieldByIndex(refFieldDef.Index)
		refFieldVal := reflect.ValueOf(fieldVal)
		refField.Set(refFieldVal)
	}
	return tp.NativeToValue(refPtr.Interface())
}

// NewValue adapts native values to CEL values and will proxy to the composed type adapter
// for non-native types.
func (tp *nativeTypeProvider) NativeToValue(val any) ref.Val {
	if val == nil {
		return types.NullValue
	}
	if v, ok := val.(ref.Val); ok {
		return v
	}
	rawVal := reflect.ValueOf(val)
	refVal := rawVal
	if refVal.Kind() == reflect.Ptr {
		refVal = reflect.Indirect(refVal)
	}
	// This isn't quite right if you're also supporting proto,
	// but maybe an acceptable limitation.
	switch refVal.Kind() {
	case reflect.Array, reflect.Slice:
		switch val := val.(type) {
		case []byte:
			return tp.baseAdapter.NativeToValue(val)
		default:
			return types.NewDynamicList(tp, val)
		}
	case reflect.Map:
		return types.NewDynamicMap(tp, val)
	case reflect.Struct:
		switch val := val.(type) {
		case proto.Message, *pb.Map, protoreflect.List, protoreflect.Message, protoreflect.Value,
			time.Time:
			return tp.baseAdapter.NativeToValue(val)
		default:
			return newNativeObject(tp, val, rawVal)
		}
	default:
		return tp.baseAdapter.NativeToValue(val)
	}
}

func convertToCelType(refType reflect.Type) (*cel.Type, bool) {
	switch refType.Kind() {
	case reflect.Bool:
		return cel.BoolType, true
	case reflect.Float32, reflect.Float64:
		return cel.DoubleType, true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if refType == durationType {
			return cel.DurationType, true
		}
		return cel.IntType, true
	case reflect.String:
		return cel.StringType, true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return cel.UintType, true
	case reflect.Array, reflect.Slice:
		refElem := refType.Elem()
		if refElem == reflect.TypeOf(byte(0)) {
			return cel.BytesType, true
		}
		elemType, ok := convertToCelType(refElem)
		if !ok {
			return nil, false
		}
		return cel.ListType(elemType), true
	case reflect.Map:
		keyType, ok := convertToCelType(refType.Key())
		if !ok {
			return nil, false
		}
		// Ensure the key type is a int, bool, uint, string
		elemType, ok := convertToCelType(refType.Elem())
		if !ok {
			return nil, false
		}
		return cel.MapType(keyType, elemType), true
	case reflect.Struct:
		if refType == timestampType {
			return cel.TimestampType, true
		}
		return cel.ObjectType(
			fmt.Sprintf("%s.%s", simplePkgAlias(refType.PkgPath()), refType.Name()),
		), true
	case reflect.Pointer:
		if refType.Implements(pbMsgInterfaceType) {
			pbMsg := reflect.New(refType.Elem()).Interface().(protoreflect.ProtoMessage)
			return cel.ObjectType(string(pbMsg.ProtoReflect().Descriptor().FullName())), true
		}
		return convertToCelType(refType.Elem())
	}
	return nil, false
}

func newNativeObject(adapter types.Adapter, val any, refValue reflect.Value) ref.Val {
	valType, err := newNativeType(refValue.Type())
	if err != nil {
		return types.NewErr(err.Error())
	}
	return &nativeObj{
		Adapter:  adapter,
		val:      val,
		valType:  valType,
		refValue: refValue,
	}
}

type nativeObj struct {
	types.Adapter
	val      any
	valType  *nativeType
	refValue reflect.Value
}

// ConvertToNative implements the ref.Val interface method.
//
// CEL does not have a notion of pointers, so whether a field is a pointer or value
// is handled as part of this conversion step.
func (o *nativeObj) ConvertToNative(typeDesc reflect.Type) (any, error) {
	if o.refValue.Type() == typeDesc {
		return o.val, nil
	}
	if o.refValue.Kind() == reflect.Pointer && o.refValue.Type().Elem() == typeDesc {
		return o.refValue.Elem().Interface(), nil
	}
	if typeDesc.Kind() == reflect.Pointer && o.refValue.Type() == typeDesc.Elem() {
		ptr := reflect.New(typeDesc.Elem())
		ptr.Elem().Set(o.refValue)
		return ptr.Interface(), nil
	}
	switch typeDesc {
	case jsonValueType:
		jsonStruct, err := o.ConvertToNative(jsonStructType)
		if err != nil {
			return nil, err
		}
		return structpb.NewStructValue(jsonStruct.(*structpb.Struct)), nil
	case jsonStructType:
		refVal := reflect.Indirect(o.refValue)
		refType := refVal.Type()
		fields := make(map[string]*structpb.Value, refVal.NumField())
		for i := 0; i < refVal.NumField(); i++ {
			fieldType := refType.Field(i)
			fieldValue := refVal.Field(i)
			if !fieldValue.IsValid() || fieldValue.IsZero() {
				continue
			}
			fieldCELVal := o.NativeToValue(fieldValue.Interface())
			fieldJSONVal, err := fieldCELVal.ConvertToNative(jsonValueType)
			if err != nil {
				return nil, err
			}
			fields[fieldType.Name] = fieldJSONVal.(*structpb.Value)
		}
		return &structpb.Struct{Fields: fields}, nil
	}
	return nil, fmt.Errorf("type conversion error from '%v' to '%v'", o.Type(), typeDesc)
}

// ConvertToType implements the ref.Val interface method.
func (o *nativeObj) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case types.TypeType:
		return o.valType
	default:
		if typeVal.TypeName() == o.valType.typeName {
			return o
		}
	}
	return types.NewErr("type conversion error from '%s' to '%s'", o.Type(), typeVal)
}

// Equal implements the ref.Val interface method.
//
// Note, that in Golang a pointer to a value is not equal to the value it contains.
// In CEL pointers and values to which they point are equal.
func (o *nativeObj) Equal(other ref.Val) ref.Val {
	otherNtv, ok := other.(*nativeObj)
	if !ok {
		return types.False
	}
	val := o.val
	otherVal := otherNtv.val
	refVal := o.refValue
	otherRefVal := otherNtv.refValue
	if refVal.Kind() != otherRefVal.Kind() {
		if refVal.Kind() == reflect.Pointer {
			val = refVal.Elem().Interface()
		} else if otherRefVal.Kind() == reflect.Pointer {
			otherVal = otherRefVal.Elem().Interface()
		}
	}
	return types.Bool(reflect.DeepEqual(val, otherVal))
}

// IsZeroValue indicates whether the contained Golang value is a zero value.
//
// Golang largely follows proto3 semantics for zero values.
func (o *nativeObj) IsZeroValue() bool {
	return reflect.Indirect(o.refValue).IsZero()
}

// IsSet tests whether a field which is defined is set to a non-default value.
func (o *nativeObj) IsSet(field ref.Val) ref.Val {
	refField, refErr := o.getReflectedField(field)
	if refErr != nil {
		return refErr
	}
	return types.Bool(!refField.IsZero())
}

// Get returns the value fo a field name.
func (o *nativeObj) Get(field ref.Val) ref.Val {
	refField, refErr := o.getReflectedField(field)
	if refErr != nil {
		return refErr
	}
	return adaptFieldValue(o, refField)
}

func (o *nativeObj) getReflectedField(field ref.Val) (reflect.Value, ref.Val) {
	fieldName, ok := field.(types.String)
	if !ok {
		return reflect.Value{}, types.MaybeNoSuchOverloadErr(field)
	}
	fieldNameStr := string(fieldName)
	refField, isDefined := o.valType.hasField(fieldNameStr)
	if !isDefined {
		return reflect.Value{}, types.NewErr("no such field: %s", fieldName)
	}
	refVal := reflect.Indirect(o.refValue)
	return refVal.FieldByIndex(refField.Index), nil
}

// Type implements the ref.Val interface method.
func (o *nativeObj) Type() ref.Type {
	return o.valType
}

// Value implements the ref.Val interface method.
func (o *nativeObj) Value() any {
	return o.val
}

func newNativeTypes(rawType reflect.Type) ([]*nativeType, error) {
	nt, err := newNativeType(rawType)
	if err != nil {
		return nil, err
	}
	result := []*nativeType{nt}

	alreadySeen := make(map[string]struct{})
	var iterateStructMembers func(reflect.Type)
	iterateStructMembers = func(t reflect.Type) {
		if k := t.Kind(); k == reflect.Pointer || k == reflect.Slice || k == reflect.Array || k == reflect.Map {
			t = t.Elem()
		}
		if t.Kind() != reflect.Struct {
			return
		}
		if _, seen := alreadySeen[t.String()]; seen {
			return
		}
		alreadySeen[t.String()] = struct{}{}
		nt, ntErr := newNativeType(t)
		if ntErr != nil {
			err = ntErr
			return
		}
		result = append(result, nt)

		for idx := 0; idx < t.NumField(); idx++ {
			iterateStructMembers(t.Field(idx).Type)
		}
	}
	iterateStructMembers(rawType)

	return result, err
}

func newNativeType(rawType reflect.Type) (*nativeType, error) {
	refType := rawType
	if refType.Kind() == reflect.Pointer {
		refType = refType.Elem()
	}
	if !isValidObjectType(refType) {
		return nil, fmt.Errorf("unsupported reflect.Type %v, must be reflect.Struct", rawType)
	}
	return &nativeType{
		typeName: fmt.Sprintf("%s.%s", simplePkgAlias(refType.PkgPath()), refType.Name()),
		refType:  refType,
	}, nil
}

type nativeType struct {
	typeName string
	refType  reflect.Type
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (t *nativeType) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return nil, fmt.Errorf("type conversion error for type to '%v'", typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (t *nativeType) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case types.TypeType:
		return types.TypeType
	}
	return types.NewErr("type conversion error from '%s' to '%s'", types.TypeType, typeVal)
}

// Equal returns true of both type names are equal to each other.
func (t *nativeType) Equal(other ref.Val) ref.Val {
	otherType, ok := other.(ref.Type)
	return types.Bool(ok && t.TypeName() == otherType.TypeName())
}

// HasTrait implements the ref.Type interface method.
func (t *nativeType) HasTrait(trait int) bool {
	return nativeObjTraitMask&trait == trait
}

// String implements the strings.Stringer interface method.
func (t *nativeType) String() string {
	return t.typeName
}

// Type implements the ref.Val interface method.
func (t *nativeType) Type() ref.Type {
	return types.TypeType
}

// TypeName implements the ref.Type interface method.
func (t *nativeType) TypeName() string {
	return t.typeName
}

// Value implements the ref.Val interface method.
func (t *nativeType) Value() any {
	return t.typeName
}

// hasField returns whether a field name has a corresponding Golang reflect.StructField
func (t *nativeType) hasField(fieldName string) (reflect.StructField, bool) {
	f, found := t.refType.FieldByName(fieldName)
	if !found || !f.IsExported() || !isSupportedType(f.Type) {
		return reflect.StructField{}, false
	}
	return f, true
}

func adaptFieldValue(adapter types.Adapter, refField reflect.Value) ref.Val {
	return adapter.NativeToValue(getFieldValue(adapter, refField))
}

func getFieldValue(adapter types.Adapter, refField reflect.Value) any {
	if refField.IsZero() {
		switch refField.Kind() {
		case reflect.Array, reflect.Slice:
			return types.NewDynamicList(adapter, []ref.Val{})
		case reflect.Map:
			return types.NewDynamicMap(adapter, map[ref.Val]ref.Val{})
		case reflect.Struct:
			if refField.Type() == timestampType {
				return types.Timestamp{Time: time.Unix(0, 0)}
			}
			return reflect.New(refField.Type()).Elem().Interface()
		case reflect.Pointer:
			return reflect.New(refField.Type().Elem()).Interface()
		}
	}
	return refField.Interface()
}

func simplePkgAlias(pkgPath string) string {
	paths := strings.Split(pkgPath, "/")
	if len(paths) == 0 {
		return ""
	}
	return paths[len(paths)-1]
}

func isValidObjectType(refType reflect.Type) bool {
	return refType.Kind() == reflect.Struct
}

func isSupportedType(refType reflect.Type) bool {
	switch refType.Kind() {
	case reflect.Chan, reflect.Complex64, reflect.Complex128, reflect.Func, reflect.UnsafePointer, reflect.Uintptr:
		return false
	case reflect.Array, reflect.Slice:
		return isSupportedType(refType.Elem())
	case reflect.Map:
		return isSupportedType(refType.Key()) && isSupportedType(refType.Elem())
	}
	return true
}

var (
	pbMsgInterfaceType = reflect.TypeOf((*protoreflect.ProtoMessage)(nil)).Elem()
	timestampType      = reflect.TypeOf(time.Now())
	durationType       = reflect.TypeOf(time.Nanosecond)
)
