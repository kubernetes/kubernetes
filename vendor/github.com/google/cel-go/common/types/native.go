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
// See the License for the me.
// limitations under the License.

package types

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	structpb "google.golang.org/protobuf/types/known/structpb"
)

var (
	nativeObjTraitMask = traits.FieldTesterType | traits.IndexerType
	jsonValueType      = reflect.TypeFor[*structpb.Value]()
	jsonStructType     = reflect.TypeFor[*structpb.Struct]()

	pbMsgInterfaceType = reflect.TypeFor[protoreflect.ProtoMessage]()
	refValType         = reflect.TypeFor[ref.Val]()
	timestampType      = reflect.TypeFor[time.Time]()
	durationType       = reflect.TypeFor[time.Duration]()

	errDuplicatedFieldName = errors.New("field name already exists in struct")
)

// NewNativeType constructs a NativeType instance for a Go struct reflect.Type.
func NewNativeType(rawType reflect.Type, opts ...NativeTypeOption) (*NativeType, error) {
	tpOptions := NativeTypeOptions{}
	for _, opt := range opts {
		if err := opt(&tpOptions); err != nil {
			return nil, err
		}
	}
	return newNativeType(rawType, tpOptions.fieldNameHandler)
}

// NativeTypesFieldNameHandler is a handler for mapping a reflect.StructField to a CEL field name.
// This can be used to override the default Go struct field to CEL field name mapping.
type NativeTypesFieldNameHandler = func(field reflect.StructField) string

// NativeTypeOptions holds options for native types.
type NativeTypeOptions struct {
	fieldNameHandler NativeTypesFieldNameHandler
}

// NativeTypeOption is a functional option for configuring handling of native types.
type NativeTypeOption func(*NativeTypeOptions) error

// ParseStructTags configures if native types field names should be overridable by CEL struct tags.
// This is equivalent to ParseStructTag("cel").
func ParseStructTags(enabled bool) NativeTypeOption {
	if enabled {
		return ParseStructTag("cel")
	}
	return ParseStructField(nil)
}

// ParseStructTag configures the struct tag to parse. The 0th item in the tag is used as the name of the CEL field.
func ParseStructTag(tag string) NativeTypeOption {
	return ParseStructField(fieldNameByTag(tag))
}

// ParseStructField configures how to parse Go struct fields. It can be used to customize struct field parsing.
func ParseStructField(handler NativeTypesFieldNameHandler) NativeTypeOption {
	return func(opts *NativeTypeOptions) error {
		opts.fieldNameHandler = handler
		return nil
	}
}

func fieldNameByTag(structTagToParse string) func(field reflect.StructField) string {
	return func(field reflect.StructField) string {
		tag, found := field.Tag.Lookup(structTagToParse)
		if found {
			splits := strings.Split(tag, ",")
			if len(splits) > 0 {
				name := splits[0]
				return name
			}
		}
		return field.Name
	}
}

func isSkippedFieldName(name string) bool {
	return name == "" || name == "-"
}

// NativeType represents a CEL struct type descriptor generated from a native Go struct.
type NativeType struct {
	typeName     string
	refType      reflect.Type
	fieldsByName map[string]reflect.StructField
}

// ReflectType implements StructTypeDescriptor.
func (t *NativeType) ReflectType() reflect.Type {
	return t.refType
}

// Adapt implements StructTypeDescriptor.
func (t *NativeType) Adapt(adapter Adapter, value any) ref.Val {
	if value == nil {
		return NullValue
	}
	refVal := reflect.ValueOf(value)
	if refVal.Kind() == reflect.Ptr {
		if refVal.IsNil() {
			return NullValue
		}
		refVal = refVal.Elem()
	}
	return &nativeObj{
		Adapter:  adapter,
		val:      value,
		valType:  t,
		refValue: refVal,
	}
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (t *NativeType) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return nil, fmt.Errorf("type conversion error for type to '%v'", typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (t *NativeType) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case TypeType:
		return TypeType
	}
	return NewErr("type conversion error from '%s' to '%s'", TypeType, typeVal)
}

// Equal returns true if both type names are equal to each other.
func (t *NativeType) Equal(other ref.Val) ref.Val {
	otherType, ok := other.(ref.Type)
	return Bool(ok && t.TypeName() == otherType.TypeName())
}

// HasTrait implements the ref.Type interface method.
func (t *NativeType) HasTrait(trait int) bool {
	return nativeObjTraitMask&trait == trait
}

// String implements the fmt.Stringer interface method.
func (t *NativeType) String() string {
	return t.typeName
}

// Type implements the ref.Val interface method.
func (t *NativeType) Type() ref.Type {
	return TypeType
}

// TypeName implements the ref.Type interface method.
func (t *NativeType) TypeName() string {
	return t.typeName
}

// Value implements the ref.Val interface method.
func (t *NativeType) Value() any {
	return t.typeName
}

func (t *NativeType) hasField(fieldName string) (reflect.StructField, bool) {
	f, found := t.fieldsByName[fieldName]
	if !found {
		return reflect.StructField{}, false
	}
	return f, true
}

// FieldNames provides the list of field names for this type.
func (t *NativeType) FieldNames() []string {
	fields := make([]string, 0, len(t.fieldsByName))
	for fieldName := range t.fieldsByName {
		fields = append(fields, fieldName)
	}
	return fields
}

// FindFieldType looks up a field by name and provides the type and accessors.
func (t *NativeType) FindFieldType(fieldName string) (*FieldType, bool) {
	refField, found := t.hasField(fieldName)
	if !found {
		return nil, false
	}
	celType, ok := convertToCelType(refField.Type)
	if !ok {
		return nil, false
	}
	return &FieldType{
		Type: celType,
		IsSet: func(obj any) bool {
			refVal := reflect.Indirect(reflect.ValueOf(obj))
			refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
			return refFieldVal.IsValid() && !refFieldVal.IsZero()
		},
		GetFrom: func(obj any) (any, error) {
			refVal := reflect.Indirect(reflect.ValueOf(obj))
			refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
			return getFieldValue(refFieldVal), nil
		},
	}, true
}

// NewValue constructs a new native Go struct instance populated with given field values.
func (t *NativeType) NewValue(adapter Adapter, fields map[string]ref.Val) ref.Val {
	refPtr := reflect.New(t.refType)
	refVal := refPtr.Elem()
	for fieldName, val := range fields {
		refFieldDef, isDefined := t.hasField(fieldName)
		if !isDefined {
			return NewErr("no such field: %s", fieldName)
		}
		fieldVal, err := val.ConvertToNative(refFieldDef.Type)
		if err != nil {
			return NewErrFromString(err.Error())
		}
		refField := safeSetFieldByIndex(refVal, refFieldDef.Index)
		if !refField.IsValid() {
			return NewErr("cannot set field: %s", fieldName)
		}
		refField.Set(reflect.ValueOf(fieldVal))
	}
	return adapter.NativeToValue(refPtr.Interface())
}

type nativeObj struct {
	Adapter
	val      any
	valType  *NativeType
	refValue reflect.Value
}

func (o *nativeObj) ConvertToNative(typeDesc reflect.Type) (any, error) {
	if o.refValue.Type() == typeDesc {
		if reflect.TypeOf(o.val) == typeDesc {
			return o.val, nil
		}
		return o.refValue.Interface(), nil
	}
	if typeDesc.Kind() == reflect.Pointer && o.refValue.Type() == typeDesc.Elem() {
		if reflect.TypeOf(o.val) == typeDesc {
			return o.val, nil
		}
		ptr := reflect.New(o.refValue.Type())
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
		fields := make(map[string]*structpb.Value, refVal.NumField())
		for fieldName, fieldType := range o.valType.fieldsByName {
			fieldValue := safeGetFieldByIndex(refVal, fieldType.Index)
			if !fieldValue.IsValid() || fieldValue.IsZero() {
				continue
			}
			fieldCELVal := o.NativeToValue(fieldValue.Interface())
			fieldJSONVal, err := fieldCELVal.ConvertToNative(jsonValueType)
			if err != nil {
				return nil, err
			}
			fields[fieldName] = fieldJSONVal.(*structpb.Value)
		}
		return &structpb.Struct{Fields: fields}, nil
	}
	return nil, fmt.Errorf("type conversion error from '%v' to '%v'", o.Type(), typeDesc)
}

func (o *nativeObj) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case TypeType:
		return o.valType
	default:
		if typeVal.TypeName() == o.valType.typeName {
			return o
		}
	}
	return NewErr("type conversion error from '%s' to '%s'", o.Type(), typeVal)
}

func (o *nativeObj) Equal(other ref.Val) ref.Val {
	otherNtv, ok := other.(*nativeObj)
	if !ok {
		return False
	}
	val := o.val
	otherVal := otherNtv.val
	if reflect.TypeOf(val).Kind() != reflect.TypeOf(otherVal).Kind() {
		val = o.refValue.Interface()
		otherVal = otherNtv.refValue.Interface()
	}
	return Bool(reflect.DeepEqual(val, otherVal))
}

func (o *nativeObj) IsZeroValue() bool {
	return o.refValue.IsZero()
}

func (o *nativeObj) IsSet(field ref.Val) ref.Val {
	refField, refErr := o.getReflectedField(field)
	if refErr != nil {
		return refErr
	}
	return Bool(!refField.IsZero())
}

func (o *nativeObj) Get(field ref.Val) ref.Val {
	refField, refErr := o.getReflectedField(field)
	if refErr != nil {
		return refErr
	}
	return adaptFieldValue(o, refField)
}

func (o *nativeObj) getReflectedField(field ref.Val) (reflect.Value, ref.Val) {
	fieldName, ok := field.(String)
	if !ok {
		return reflect.Value{}, MaybeNoSuchOverloadErr(field)
	}
	fieldNameStr := string(fieldName)
	refField, isDefined := o.valType.hasField(fieldNameStr)
	if !isDefined {
		return reflect.Value{}, NewErr("no such field: %s", fieldName)
	}
	refVal := reflect.Indirect(o.refValue)
	return safeGetFieldByIndex(refVal, refField.Index), nil
}

func (o *nativeObj) Type() ref.Type {
	return o.valType
}

func (o *nativeObj) Value() any {
	return o.val
}

func newNativeTypes(rawType reflect.Type, fieldNameHandler NativeTypesFieldNameHandler) ([]*NativeType, error) {
	nt, err := newNativeType(rawType, fieldNameHandler)
	if err != nil {
		return nil, err
	}
	result := []*NativeType{nt}

	alreadySeen := make(map[string]struct{})
	var iterateStructMembers func(reflect.Type)
	iterateStructMembers = func(t reflect.Type) {
		if t.Implements(reflect.TypeFor[ref.Val]()) {
			return
		}
		if k := t.Kind(); k == reflect.Pointer || k == reflect.Slice || k == reflect.Array || k == reflect.Map {
			iterateStructMembers(t.Elem())
			return
		}
		if t.Kind() != reflect.Struct {
			return
		}
		if _, seen := alreadySeen[t.String()]; seen {
			return
		}
		alreadySeen[t.String()] = struct{}{}
		nt, ntErr := newNativeType(t, fieldNameHandler)
		if ntErr != nil {
			err = ntErr
			return
		}
		result = append(result, nt)

		for _, field := range reflect.VisibleFields(t) {
			if !field.IsExported() || !isSupportedType(field.Type) {
				continue
			}
			iterateStructMembers(field.Type)
		}
	}
	iterateStructMembers(rawType)

	return result, err
}

func toFieldName(f reflect.StructField, fieldNameHandler NativeTypesFieldNameHandler) string {
	if fieldNameHandler == nil {
		return f.Name
	}
	return fieldNameHandler(f)
}

func newNativeType(rawType reflect.Type, fieldNameHandler NativeTypesFieldNameHandler) (*NativeType, error) {
	refType := rawType
	if refType.Kind() == reflect.Pointer {
		refType = refType.Elem()
	}
	if !isValidObjectType(refType) {
		return nil, fmt.Errorf("unsupported reflect.Type %v, must be reflect.Struct", rawType)
	}

	fieldsByName := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(refType) {
		if !field.IsExported() || !isSupportedType(field.Type) {
			continue
		}
		fieldName := toFieldName(field, fieldNameHandler)
		if isSkippedFieldName(fieldName) {
			continue
		}
		if _, found := fieldsByName[fieldName]; found {
			return nil, fmt.Errorf("invalid field name `%s` in struct `%s`: %w", fieldName, refType.Name(), errDuplicatedFieldName)
		}
		fieldsByName[fieldName] = field
	}

	return &NativeType{
		typeName:     fmt.Sprintf("%s.%s", simplePkgAlias(refType.PkgPath()), refType.Name()),
		refType:      refType,
		fieldsByName: fieldsByName,
	}, nil
}

func adaptFieldValue(adapter Adapter, refField reflect.Value) ref.Val {
	return adapter.NativeToValue(getFieldValue(refField))
}

func safeSetFieldByIndex(v reflect.Value, index []int) reflect.Value {
	for _, i := range index {
		if v.Kind() == reflect.Pointer {
			if v.IsNil() {
				v.Set(reflect.New(v.Type().Elem()))
			}
			v = v.Elem()
		}
		if v.Kind() != reflect.Struct || i >= v.NumField() {
			return reflect.Value{}
		}
		v = v.Field(i)
	}
	return v
}

func safeGetFieldByIndex(v reflect.Value, index []int) reflect.Value {
	for _, i := range index {
		if v.Kind() == reflect.Pointer {
			if v.IsNil() {
				v = reflect.New(v.Type().Elem()).Elem()
			} else {
				v = v.Elem()
			}
		}
		if v.Kind() != reflect.Struct || i >= v.NumField() {
			return reflect.Value{}
		}
		v = v.Field(i)
	}
	return v
}

func getFieldValue(refField reflect.Value) any {
	if !refField.IsValid() {
		return nil
	}
	if refField.IsZero() {
		switch refField.Kind() {
		case reflect.Struct:
			if refField.Type() == timestampType {
				return time.Unix(0, 0)
			}
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

func convertToCelType(refType reflect.Type) (*Type, bool) {
	switch refType.Kind() {
	case reflect.Bool:
		return BoolType, true
	case reflect.Float32, reflect.Float64:
		return DoubleType, true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if refType == durationType {
			return DurationType, true
		}
		return IntType, true
	case reflect.String:
		return StringType, true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return UintType, true
	case reflect.Array, reflect.Slice:
		refElem := refType.Elem()
		if refElem == reflect.TypeOf(byte(0)) {
			return BytesType, true
		}
		elemType, ok := convertToCelType(refElem)
		if !ok {
			return nil, false
		}
		return NewListType(elemType), true
	case reflect.Map:
		keyType, ok := convertToCelType(refType.Key())
		if !ok {
			return nil, false
		}
		elemType, ok := convertToCelType(refType.Elem())
		if !ok {
			return nil, false
		}
		return NewMapType(keyType, elemType), true
	case reflect.Struct:
		if refType == timestampType {
			return TimestampType, true
		}
		if refType.Implements(refValType) {
			emptyCelVal := reflect.New(refType).Elem().Interface().(ref.Val)
			return emptyCelVal.Type().(*Type), true
		}
		return NewObjectType(
			fmt.Sprintf("%s.%s", simplePkgAlias(refType.PkgPath()), refType.Name()),
		), true
	case reflect.Pointer:
		if refType.Implements(refValType) {
			emptyCelVal := reflect.New(refType.Elem()).Interface().(ref.Val)
			return emptyCelVal.Type().(*Type), true
		}
		if refType.Implements(pbMsgInterfaceType) {
			pbMsg := reflect.New(refType.Elem()).Interface().(protoreflect.ProtoMessage)
			return NewObjectType(string(pbMsg.ProtoReflect().Descriptor().FullName())), true
		}
		return convertToCelType(refType.Elem())
	}
	return nil, false
}
