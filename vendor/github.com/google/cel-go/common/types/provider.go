// Copyright 2018 Google LLC
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

package types

import (
	"fmt"
	"reflect"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/google/cel-go/common/types/pb"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	anypb "google.golang.org/protobuf/types/known/anypb"
	dpb "google.golang.org/protobuf/types/known/durationpb"
	structpb "google.golang.org/protobuf/types/known/structpb"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

// Adapter converts native Go values of varying type and complexity to equivalent CEL values.
type Adapter = ref.TypeAdapter

// Provider specifies functions for creating new object instances and for resolving
// enum values by name.
type Provider interface {
	// EnumValue returns the numeric value of the given enum value name.
	EnumValue(enumName string) ref.Val

	// FindIdent takes a qualified identifier name and returns a ref.Val if one exists.
	FindIdent(identName string) (ref.Val, bool)

	// FindStructType returns the Type give a qualified type name.
	//
	// For historical reasons, only struct types are expected to be returned through this
	// method, and the type values are expected to be wrapped in a TypeType instance using
	// TypeTypeWithParam(<structType>).
	//
	// Returns false if not found.
	FindStructType(structType string) (*Type, bool)

	// FieldStructFieldType returns the field type for a checked type value. Returns
	// false if the field could not be found.
	FindStructFieldType(structType, fieldName string) (*FieldType, bool)

	// NewValue creates a new type value from a qualified name and map of field
	// name to value.
	//
	// Note, for each value, the Val.ConvertToNative function will be invoked
	// to convert the Val to the field's native type. If an error occurs during
	// conversion, the NewValue will be a types.Err.
	NewValue(structType string, fields map[string]ref.Val) ref.Val
}

// FieldType represents a field's type value and whether that field supports presence detection.
type FieldType struct {
	// Type of the field as a CEL native type value.
	Type *Type

	// IsSet indicates whether the field is set on an input object.
	IsSet ref.FieldTester

	// GetFrom retrieves the field value on the input object, if set.
	GetFrom ref.FieldGetter
}

// Registry provides type information for a set of registered types.
type Registry struct {
	revTypeMap map[string]*Type
	pbdb       *pb.Db
}

// NewRegistry accepts a list of proto message instances and returns a type
// provider which can create new instances of the provided message or any
// message that proto depends upon in its FileDescriptor.
func NewRegistry(types ...proto.Message) (*Registry, error) {
	p := &Registry{
		revTypeMap: make(map[string]*Type),
		pbdb:       pb.NewDb(),
	}
	err := p.RegisterType(
		BoolType,
		BytesType,
		DoubleType,
		DurationType,
		IntType,
		ListType,
		MapType,
		NullType,
		StringType,
		TimestampType,
		TypeType,
		UintType)
	if err != nil {
		return nil, err
	}
	// This block ensures that the well-known protobuf types are registered by default.
	for _, fd := range p.pbdb.FileDescriptions() {
		err = p.registerAllTypes(fd)
		if err != nil {
			return nil, err
		}
	}
	for _, msgType := range types {
		err = p.RegisterMessage(msgType)
		if err != nil {
			return nil, err
		}
	}
	return p, nil
}

// NewEmptyRegistry returns a registry which is completely unconfigured.
func NewEmptyRegistry() *Registry {
	return &Registry{
		revTypeMap: make(map[string]*Type),
		pbdb:       pb.NewDb(),
	}
}

// Copy copies the current state of the registry into its own memory space.
func (p *Registry) Copy() *Registry {
	copy := &Registry{
		revTypeMap: make(map[string]*Type),
		pbdb:       p.pbdb.Copy(),
	}
	for k, v := range p.revTypeMap {
		copy.revTypeMap[k] = v
	}
	return copy
}

// EnumValue returns the numeric value of the given enum value name.
func (p *Registry) EnumValue(enumName string) ref.Val {
	enumVal, found := p.pbdb.DescribeEnum(enumName)
	if !found {
		return NewErr("unknown enum name '%s'", enumName)
	}
	return Int(enumVal.Value())
}

// FieldFieldType returns the field type for a checked type value. Returns false if
// the field could not be found.
//
// Deprecated: use FindStructFieldType
func (p *Registry) FindFieldType(structType, fieldName string) (*ref.FieldType, bool) {
	msgType, found := p.pbdb.DescribeType(structType)
	if !found {
		return nil, false
	}
	field, found := msgType.FieldByName(fieldName)
	if !found {
		return nil, false
	}
	return &ref.FieldType{
		Type:    field.CheckedType(),
		IsSet:   field.IsSet,
		GetFrom: field.GetFrom}, true
}

// FieldStructFieldType returns the field type for a checked type value. Returns
// false if the field could not be found.
func (p *Registry) FindStructFieldType(structType, fieldName string) (*FieldType, bool) {
	msgType, found := p.pbdb.DescribeType(structType)
	if !found {
		return nil, false
	}
	field, found := msgType.FieldByName(fieldName)
	if !found {
		return nil, false
	}
	return &FieldType{
		Type:    fieldDescToCELType(field),
		IsSet:   field.IsSet,
		GetFrom: field.GetFrom}, true
}

// FindIdent takes a qualified identifier name and returns a ref.Val if one exists.
func (p *Registry) FindIdent(identName string) (ref.Val, bool) {
	if t, found := p.revTypeMap[identName]; found {
		return t, true
	}
	if enumVal, found := p.pbdb.DescribeEnum(identName); found {
		return Int(enumVal.Value()), true
	}
	return nil, false
}

// FindType looks up the Type given a qualified typeName. Returns false if not found.
//
// Deprecated: use FindStructType
func (p *Registry) FindType(structType string) (*exprpb.Type, bool) {
	if _, found := p.pbdb.DescribeType(structType); !found {
		return nil, false
	}
	if structType != "" && structType[0] == '.' {
		structType = structType[1:]
	}
	return &exprpb.Type{
		TypeKind: &exprpb.Type_Type{
			Type: &exprpb.Type{
				TypeKind: &exprpb.Type_MessageType{
					MessageType: structType}}}}, true
}

// FindStructType returns the Type give a qualified type name.
//
// For historical reasons, only struct types are expected to be returned through this
// method, and the type values are expected to be wrapped in a TypeType instance using
// TypeTypeWithParam(<structType>).
//
// Returns false if not found.
func (p *Registry) FindStructType(structType string) (*Type, bool) {
	if _, found := p.pbdb.DescribeType(structType); !found {
		return nil, false
	}
	if structType != "" && structType[0] == '.' {
		structType = structType[1:]
	}
	return NewTypeTypeWithParam(NewObjectType(structType)), true
}

// NewValue creates a new type value from a qualified name and map of field
// name to value.
//
// Note, for each value, the Val.ConvertToNative function will be invoked
// to convert the Val to the field's native type. If an error occurs during
// conversion, the NewValue will be a types.Err.
func (p *Registry) NewValue(structType string, fields map[string]ref.Val) ref.Val {
	td, found := p.pbdb.DescribeType(structType)
	if !found {
		return NewErr("unknown type '%s'", structType)
	}
	msg := td.New()
	fieldMap := td.FieldMap()
	for name, value := range fields {
		field, found := fieldMap[name]
		if !found {
			return NewErr("no such field: %s", name)
		}
		err := msgSetField(msg, field, value)
		if err != nil {
			return &Err{err}
		}
	}
	return p.NativeToValue(msg.Interface())
}

// RegisterDescriptor registers the contents of a protocol buffer `FileDescriptor`.
func (p *Registry) RegisterDescriptor(fileDesc protoreflect.FileDescriptor) error {
	fd, err := p.pbdb.RegisterDescriptor(fileDesc)
	if err != nil {
		return err
	}
	return p.registerAllTypes(fd)
}

// RegisterMessage registers a protocol buffer message and its dependencies.
func (p *Registry) RegisterMessage(message proto.Message) error {
	fd, err := p.pbdb.RegisterMessage(message)
	if err != nil {
		return err
	}
	return p.registerAllTypes(fd)
}

// RegisterType registers a type value with the provider which ensures the provider is aware of how to
// map the type to an identifier.
//
// If the `ref.Type` value is a `*types.Type` it will be registered directly by its runtime type name.
// If the `ref.Type` value is not a `*types.Type` instance, a `*types.Type` instance which reflects the
// traits present on the input and the runtime type name. By default this foreign type will be treated
// as a types.StructKind. To avoid potential issues where the `ref.Type` values does not match the
// generated `*types.Type` instance, consider always using the `*types.Type` to represent type extensions
// to CEL, even when they're not based on protobuf types.
func (p *Registry) RegisterType(types ...ref.Type) error {
	for _, t := range types {
		celType := maybeForeignType(t)
		existing, found := p.revTypeMap[t.TypeName()]
		if !found {
			p.revTypeMap[t.TypeName()] = celType
			continue
		}
		if !existing.IsEquivalentType(celType) {
			return fmt.Errorf("type registration conflict. found: %v, input: %v", existing, celType)
		}
		if existing.traitMask != celType.traitMask {
			return fmt.Errorf(
				"type registered with conflicting traits: %v with traits %v, input: %v",
				existing.TypeName(), existing.traitMask, celType.traitMask)
		}
	}
	return nil
}

// NativeToValue converts various "native" types to ref.Val with this specific implementation
// providing support for custom proto-based types.
//
// This method should be the inverse of ref.Val.ConvertToNative.
func (p *Registry) NativeToValue(value any) ref.Val {
	if val, found := nativeToValue(p, value); found {
		return val
	}
	switch v := value.(type) {
	case proto.Message:
		typeName := string(v.ProtoReflect().Descriptor().FullName())
		td, found := p.pbdb.DescribeType(typeName)
		if !found {
			return NewErr("unknown type: '%s'", typeName)
		}
		unwrapped, isUnwrapped, err := td.MaybeUnwrap(v)
		if err != nil {
			return UnsupportedRefValConversionErr(v)
		}
		if isUnwrapped {
			return p.NativeToValue(unwrapped)
		}
		typeVal, found := p.FindIdent(typeName)
		if !found {
			return NewErr("unknown type: '%s'", typeName)
		}
		return NewObject(p, td, typeVal, v)
	case *pb.Map:
		return NewProtoMap(p, v)
	case protoreflect.List:
		return NewProtoList(p, v)
	case protoreflect.Message:
		return p.NativeToValue(v.Interface())
	case protoreflect.Value:
		return p.NativeToValue(v.Interface())
	}
	return UnsupportedRefValConversionErr(value)
}

func (p *Registry) registerAllTypes(fd *pb.FileDescription) error {
	for _, typeName := range fd.GetTypeNames() {
		// skip well-known type names since they're automatically sanitized
		// during NewObjectType() calls.
		if _, found := checkedWellKnowns[typeName]; found {
			continue
		}
		err := p.RegisterType(NewObjectTypeValue(typeName))
		if err != nil {
			return err
		}
	}
	return nil
}

func fieldDescToCELType(field *pb.FieldDescription) *Type {
	if field.IsMap() {
		return NewMapType(
			singularFieldDescToCELType(field.KeyType),
			singularFieldDescToCELType(field.ValueType))
	}
	if field.IsList() {
		return NewListType(singularFieldDescToCELType(field))
	}
	return singularFieldDescToCELType(field)
}

func singularFieldDescToCELType(field *pb.FieldDescription) *Type {
	if field.IsMessage() {
		return NewObjectType(string(field.Descriptor().Message().FullName()))
	}
	if field.IsEnum() {
		return IntType
	}
	return ProtoCELPrimitives[field.ProtoKind()]
}

// defaultTypeAdapter converts go native types to CEL values.
type defaultTypeAdapter struct{}

var (
	// DefaultTypeAdapter adapts canonical CEL types from their equivalent Go values.
	DefaultTypeAdapter = &defaultTypeAdapter{}
)

// NativeToValue implements the ref.TypeAdapter interface.
func (a *defaultTypeAdapter) NativeToValue(value any) ref.Val {
	if val, found := nativeToValue(a, value); found {
		return val
	}
	return UnsupportedRefValConversionErr(value)
}

// nativeToValue returns the converted (ref.Val, true) of a conversion is found,
// otherwise (nil, false)
func nativeToValue(a Adapter, value any) (ref.Val, bool) {
	switch v := value.(type) {
	case nil:
		return NullValue, true
	case *Bool:
		if v != nil {
			return *v, true
		}
	case *Bytes:
		if v != nil {
			return *v, true
		}
	case *Double:
		if v != nil {
			return *v, true
		}
	case *Int:
		if v != nil {
			return *v, true
		}
	case *String:
		if v != nil {
			return *v, true
		}
	case *Uint:
		if v != nil {
			return *v, true
		}
	case bool:
		return Bool(v), true
	case int:
		return Int(v), true
	case int32:
		return Int(v), true
	case int64:
		return Int(v), true
	case uint:
		return Uint(v), true
	case uint32:
		return Uint(v), true
	case uint64:
		return Uint(v), true
	case float32:
		return Double(v), true
	case float64:
		return Double(v), true
	case string:
		return String(v), true
	case *dpb.Duration:
		return Duration{Duration: v.AsDuration()}, true
	case time.Duration:
		return Duration{Duration: v}, true
	case *tpb.Timestamp:
		return Timestamp{Time: v.AsTime()}, true
	case time.Time:
		return Timestamp{Time: v}, true
	case *bool:
		if v != nil {
			return Bool(*v), true
		}
	case *float32:
		if v != nil {
			return Double(*v), true
		}
	case *float64:
		if v != nil {
			return Double(*v), true
		}
	case *int:
		if v != nil {
			return Int(*v), true
		}
	case *int32:
		if v != nil {
			return Int(*v), true
		}
	case *int64:
		if v != nil {
			return Int(*v), true
		}
	case *string:
		if v != nil {
			return String(*v), true
		}
	case *uint:
		if v != nil {
			return Uint(*v), true
		}
	case *uint32:
		if v != nil {
			return Uint(*v), true
		}
	case *uint64:
		if v != nil {
			return Uint(*v), true
		}
	case []byte:
		return Bytes(v), true
	// specializations for common lists types.
	case []string:
		return NewStringList(a, v), true
	case []ref.Val:
		return NewRefValList(a, v), true
	// specializations for common map types.
	case map[string]string:
		return NewStringStringMap(a, v), true
	case map[string]any:
		return NewStringInterfaceMap(a, v), true
	case map[ref.Val]ref.Val:
		return NewRefValMap(a, v), true
	// additional specializations may be added upon request / need.
	case *anypb.Any:
		if v == nil {
			return UnsupportedRefValConversionErr(v), true
		}
		unpackedAny, err := v.UnmarshalNew()
		if err != nil {
			return NewErr("anypb.UnmarshalNew() failed for type %q: %v", v.GetTypeUrl(), err), true
		}
		return a.NativeToValue(unpackedAny), true
	case *structpb.NullValue, structpb.NullValue:
		return NullValue, true
	case *structpb.ListValue:
		return NewJSONList(a, v), true
	case *structpb.Struct:
		return NewJSONStruct(a, v), true
	case ref.Val:
		return v, true
	case protoreflect.EnumNumber:
		return Int(v), true
	case proto.Message:
		if v == nil {
			return UnsupportedRefValConversionErr(v), true
		}
		typeName := string(v.ProtoReflect().Descriptor().FullName())
		td, found := pb.DefaultDb.DescribeType(typeName)
		if !found {
			return nil, false
		}
		val, unwrapped, err := td.MaybeUnwrap(v)
		if err != nil {
			return UnsupportedRefValConversionErr(v), true
		}
		if !unwrapped {
			return nil, false
		}
		return a.NativeToValue(val), true
	// Note: dynamicpb.Message implements the proto.Message _and_ protoreflect.Message interfaces
	// which means that this case must appear after handling a proto.Message type.
	case protoreflect.Message:
		return a.NativeToValue(v.Interface()), true
	default:
		refValue := reflect.ValueOf(v)
		if refValue.Kind() == reflect.Ptr {
			if refValue.IsNil() {
				return UnsupportedRefValConversionErr(v), true
			}
			refValue = refValue.Elem()
		}
		refKind := refValue.Kind()
		switch refKind {
		case reflect.Array, reflect.Slice:
			return NewDynamicList(a, v), true
		case reflect.Map:
			return NewDynamicMap(a, v), true
		// type aliases of primitive types cannot be asserted as that type, but rather need
		// to be downcast to int32 before being converted to a CEL representation.
		case reflect.Int32:
			intType := reflect.TypeOf(int32(0))
			return Int(refValue.Convert(intType).Interface().(int32)), true
		case reflect.Int64:
			intType := reflect.TypeOf(int64(0))
			return Int(refValue.Convert(intType).Interface().(int64)), true
		case reflect.Uint32:
			uintType := reflect.TypeOf(uint32(0))
			return Uint(refValue.Convert(uintType).Interface().(uint32)), true
		case reflect.Uint64:
			uintType := reflect.TypeOf(uint64(0))
			return Uint(refValue.Convert(uintType).Interface().(uint64)), true
		case reflect.Float32:
			doubleType := reflect.TypeOf(float32(0))
			return Double(refValue.Convert(doubleType).Interface().(float32)), true
		case reflect.Float64:
			doubleType := reflect.TypeOf(float64(0))
			return Double(refValue.Convert(doubleType).Interface().(float64)), true
		}
	}
	return nil, false
}

func msgSetField(target protoreflect.Message, field *pb.FieldDescription, val ref.Val) error {
	if field.IsList() {
		lv := target.NewField(field.Descriptor())
		list, ok := val.(traits.Lister)
		if !ok {
			return unsupportedTypeConversionError(field, val)
		}
		err := msgSetListField(lv.List(), field, list)
		if err != nil {
			return err
		}
		target.Set(field.Descriptor(), lv)
		return nil
	}
	if field.IsMap() {
		mv := target.NewField(field.Descriptor())
		mp, ok := val.(traits.Mapper)
		if !ok {
			return unsupportedTypeConversionError(field, val)
		}
		err := msgSetMapField(mv.Map(), field, mp)
		if err != nil {
			return err
		}
		target.Set(field.Descriptor(), mv)
		return nil
	}
	v, err := val.ConvertToNative(field.ReflectType())
	if err != nil {
		return fieldTypeConversionError(field, err)
	}
	if v == nil {
		return nil
	}
	switch pv := v.(type) {
	case proto.Message:
		v = pv.ProtoReflect()
	}
	target.Set(field.Descriptor(), protoreflect.ValueOf(v))
	return nil
}

func msgSetListField(target protoreflect.List, listField *pb.FieldDescription, listVal traits.Lister) error {
	elemReflectType := listField.ReflectType().Elem()
	for i := Int(0); i < listVal.Size().(Int); i++ {
		elem := listVal.Get(i)
		elemVal, err := elem.ConvertToNative(elemReflectType)
		if err != nil {
			return fieldTypeConversionError(listField, err)
		}
		if elemVal == nil {
			continue
		}
		switch ev := elemVal.(type) {
		case proto.Message:
			elemVal = ev.ProtoReflect()
		}
		target.Append(protoreflect.ValueOf(elemVal))
	}
	return nil
}

func msgSetMapField(target protoreflect.Map, mapField *pb.FieldDescription, mapVal traits.Mapper) error {
	targetKeyType := mapField.KeyType.ReflectType()
	targetValType := mapField.ValueType.ReflectType()
	it := mapVal.Iterator()
	for it.HasNext() == True {
		key := it.Next()
		val := mapVal.Get(key)
		k, err := key.ConvertToNative(targetKeyType)
		if err != nil {
			return fieldTypeConversionError(mapField, err)
		}
		v, err := val.ConvertToNative(targetValType)
		if err != nil {
			return fieldTypeConversionError(mapField, err)
		}
		if v == nil {
			continue
		}
		switch pv := v.(type) {
		case proto.Message:
			v = pv.ProtoReflect()
		}
		target.Set(protoreflect.ValueOf(k).MapKey(), protoreflect.ValueOf(v))
	}
	return nil
}

func unsupportedTypeConversionError(field *pb.FieldDescription, val ref.Val) error {
	msgName := field.Descriptor().ContainingMessage().FullName()
	return fmt.Errorf("unsupported field type for %v.%v: %v", msgName, field.Name(), val.Type())
}

func fieldTypeConversionError(field *pb.FieldDescription, err error) error {
	msgName := field.Descriptor().ContainingMessage().FullName()
	return fmt.Errorf("field type conversion error for %v.%v value type: %v", msgName, field.Name(), err)
}

var (
	// ProtoCELPrimitives provides a map from the protoreflect Kind to the equivalent CEL type.
	ProtoCELPrimitives = map[protoreflect.Kind]*Type{
		protoreflect.BoolKind:     BoolType,
		protoreflect.BytesKind:    BytesType,
		protoreflect.DoubleKind:   DoubleType,
		protoreflect.FloatKind:    DoubleType,
		protoreflect.Int32Kind:    IntType,
		protoreflect.Int64Kind:    IntType,
		protoreflect.Sint32Kind:   IntType,
		protoreflect.Sint64Kind:   IntType,
		protoreflect.Uint32Kind:   UintType,
		protoreflect.Uint64Kind:   UintType,
		protoreflect.Fixed32Kind:  UintType,
		protoreflect.Fixed64Kind:  UintType,
		protoreflect.Sfixed32Kind: IntType,
		protoreflect.Sfixed64Kind: IntType,
		protoreflect.StringKind:   StringType,
	}
)
