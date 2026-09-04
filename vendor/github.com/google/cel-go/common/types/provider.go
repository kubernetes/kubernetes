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
	"maps"
	"reflect"
	"sync/atomic"
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

	// FindStructFieldNames returns the field names associated with the type, if the type
	// is found.
	FindStructFieldNames(structType string) ([]string, bool)

	// FindStructFieldType returns the field type for a checked type value. Returns
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

	// IsJSONField
	IsJSONField bool
}

// Registry provides type information for a set of registered types.
type Registry struct {
	revTypeMap    map[string]*Type
	structTypes   map[string]StructTypeDescriptor
	reflectTypes  map[reflect.Type]StructTypeDescriptor
	shared        atomic.Bool
	pbdb          *pb.Db
	provider      Provider
	adapter       Adapter
	nativeOptions NativeTypeOptions
}

// NewRegistry accepts a list of proto message instances, ref.Type instances, or RegistryOption
// functions and returns a type provider.
func NewRegistry(types ...any) (*Registry, error) {
	r, err := NewProtoRegistry()
	if err != nil {
		return nil, err
	}
	if err := registerTypeItems(r, types...); err != nil {
		return nil, err
	}
	return r, nil
}

// RegistryOption configures the behavior of the registry.
type RegistryOption func(r *Registry) (*Registry, error)

// JSONFieldNames configures JSON field name support within the protobuf types in the registry.
func JSONFieldNames(enabled bool) RegistryOption {
	return func(r *Registry) (*Registry, error) {
		err := r.WithJSONFieldNames(enabled)
		return r, err
	}
}

// ProtoTypeDefs creates a RegistryOption which registers the individual proto messages with the registry.
func ProtoTypeDefs(types ...proto.Message) RegistryOption {
	return func(r *Registry) (*Registry, error) {
		for _, msgType := range types {
			err := r.RegisterMessage(msgType)
			if err != nil {
				return nil, err
			}
		}
		return r, nil
	}
}

// Types creates a RegistryOption which registers individual custom type references or descriptors with the registry.
func Types(types ...ref.Type) RegistryOption {
	return func(r *Registry) (*Registry, error) {
		err := r.RegisterType(types...)
		if err != nil {
			return nil, err
		}
		return r, nil
	}
}

// NewProtoRegistry creates a proto-based registry with a set of configurable options.
func NewProtoRegistry(opts ...RegistryOption) (*Registry, error) {
	r := NewEmptyRegistry()
	err := r.RegisterType(
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
	for _, fd := range r.pbdb.FileDescriptions() {
		err = r.registerAllTypes(fd)
		if err != nil {
			return nil, err
		}
	}
	for _, opt := range opts {
		r, err = opt(r)
		if err != nil {
			return nil, err
		}
	}
	return r, nil
}

// NewEmptyRegistry returns a registry which is completely unconfigured.
func NewEmptyRegistry() *Registry {
	return &Registry{
		revTypeMap:   make(map[string]*Type),
		structTypes:  make(map[string]StructTypeDescriptor),
		reflectTypes: make(map[reflect.Type]StructTypeDescriptor),
		pbdb:         pb.NewDb(),
	}
}

// ComposeTypes accepts a provider, adapter, and a list of types (ref.Type, proto.Message, protoreflect.FileDescriptor, or RegistryOption)
// and either:
//   - Determines the provider and adapter are the same instance and a *Registry and registers the listed types via RegisterType or
//     one of the other registration methods as appropriate.
//   - Determines the provider and adapter are not the same, or not a *Registry and creates a new composed *Registry which references
//     the new type information first and then proxies to the underlying provider and adapter methods as appropriate.
func ComposeTypes(provider Provider, adapter Adapter, types ...any) (Provider, Adapter, error) {
	reg, isReg := provider.(*Registry)
	aReg, isAdapterReg := adapter.(*Registry)
	if isReg && isAdapterReg && reg == aReg {
		if err := registerTypeItems(reg, types...); err != nil {
			return nil, nil, err
		}
		return reg, reg, nil
	}

	composedReg, err := NewRegistry(types...)
	if err != nil {
		return nil, nil, err
	}
	composedReg.provider = provider
	composedReg.adapter = adapter
	return composedReg, composedReg, nil
}

// Copy copies the current state of the registry into its own memory space.
func (p *Registry) Copy() *Registry {
	if p == nil {
		return nil
	}
	p.shared.Store(true)
	cpy := &Registry{
		revTypeMap:    p.revTypeMap,
		structTypes:   p.structTypes,
		reflectTypes:  p.reflectTypes,
		nativeOptions: p.nativeOptions,
		pbdb:          p.pbdb,
		provider:      p.provider,
		adapter:       p.adapter,
	}
	cpy.shared.Store(true)
	return cpy
}

func (p *Registry) ensureMutable() {
	if p.shared.Load() {
		p.revTypeMap = maps.Clone(p.revTypeMap)
		p.structTypes = maps.Clone(p.structTypes)
		p.reflectTypes = maps.Clone(p.reflectTypes)
		p.pbdb = p.pbdb.Copy()
		p.shared.Store(false)
	}
}

// JSONFieldNames returns whether json field names are enabled in this registry.
func (p *Registry) JSONFieldNames() bool {
	return p.pbdb.JSONFieldNames()
}

// WithJSONFieldNames configures the registry with the JSON field name support enabled or disabled.
func (p *Registry) WithJSONFieldNames(enabled bool) error {
	if enabled == p.pbdb.JSONFieldNames() {
		return nil
	}
	p.ensureMutable()
	newDB := pb.NewDb(pb.JSONFieldNames(enabled))
	files := p.pbdb.FileDescriptions()
	for _, fd := range files {
		_, err := newDB.RegisterDescriptor(fd.FileDescriptor())
		if err != nil {
			return err
		}
	}
	p.pbdb = newDB
	return nil
}

// EnumValue returns the numeric value of the given enum value name.
func (p *Registry) EnumValue(enumName string) ref.Val {
	enumVal, found := p.pbdb.DescribeEnum(enumName)
	if !found {
		if p.provider != nil {
			return p.provider.EnumValue(enumName)
		}
		return NewErr("unknown enum name '%s'", enumName)
	}
	return Int(enumVal.Value())
}

// FindFieldType returns the field type for a checked type value. Returns false if
// the field could not be found.
//
// Deprecated: use FindStructFieldType
func (p *Registry) FindFieldType(structType, fieldName string) (*ref.FieldType, bool) {
	structType = sanitizeStructTypeName(structType)
	if st, found := p.structTypes[structType]; found {
		if ft, found := st.FindFieldType(fieldName); found {
			exprType, err := TypeToExprType(ft.Type)
			if err != nil {
				return nil, false
			}
			return makeRefFieldType(exprType, ft.IsSet, ft.GetFrom, ft.IsJSONField), true
		}
	}
	if msgType, found := p.pbdb.DescribeType(structType); found {
		if field, found := msgType.FieldByName(fieldName); found {
			return makeRefFieldType(field.CheckedType(), field.IsSet, field.GetFrom, p.pbdb.JSONFieldNames() && fieldName == field.JSONName()), true
		}
	}
	if p.provider != nil {
		if ft, ok := p.provider.FindStructFieldType(structType, fieldName); ok && ft != nil {
			exprType, err := TypeToExprType(ft.Type)
			if err != nil {
				return nil, false
			}
			return makeRefFieldType(exprType, ft.IsSet, ft.GetFrom, ft.IsJSONField), true
		}
	}
	return nil, false
}

// FindStructFieldNames returns the set of field names for the given struct type,
// if the type exists in the registry.
func (p *Registry) FindStructFieldNames(structType string) ([]string, bool) {
	structType = sanitizeStructTypeName(structType)
	if st, found := p.structTypes[structType]; found {
		return st.FieldNames(), true
	}
	if msgType, found := p.pbdb.DescribeType(structType); found {
		fieldMap := msgType.FieldMap()
		fields := make([]string, len(fieldMap))
		idx := 0
		for f := range fieldMap {
			fields[idx] = f
			idx++
		}
		return fields, true
	}
	if p.provider != nil {
		return p.provider.FindStructFieldNames(structType)
	}
	return []string{}, false
}

// FindStructFieldType returns the field type for a checked type value. Returns
// false if the field could not be found.
func (p *Registry) FindStructFieldType(structType, fieldName string) (*FieldType, bool) {
	structType = sanitizeStructTypeName(structType)
	if st, found := p.structTypes[structType]; found {
		if ft, found := st.FindFieldType(fieldName); found {
			return ft, true
		}
	}
	if msgType, found := p.pbdb.DescribeType(structType); found {
		if field, found := msgType.FieldByName(fieldName); found {
			return &FieldType{
				Type:        fieldDescToCELType(field),
				IsSet:       field.IsSet,
				GetFrom:     field.GetFrom,
				IsJSONField: p.pbdb.JSONFieldNames() && fieldName == field.JSONName(),
			}, true
		}
	}
	if p.provider != nil {
		return p.provider.FindStructFieldType(structType, fieldName)
	}
	return nil, false
}

// FindStructFieldDescription returns documentation for a field if available.
// Returns false if the field could not be found.
func (p *Registry) FindStructFieldDescription(structType, fieldName string) (string, bool) {
	structType = sanitizeStructTypeName(structType)
	if msgType, found := p.pbdb.DescribeType(structType); found {
		if field, found := msgType.FieldByName(fieldName); found {
			return field.Documentation(), true
		}
	}
	if p.provider != nil {
		if pd, ok := p.provider.(interface {
			FindStructFieldDescription(string, string) (string, bool)
		}); ok {
			return pd.FindStructFieldDescription(structType, fieldName)
		}
	}
	return "", false
}

// FindIdent takes a qualified identifier name and returns a ref.Val if one exists.
func (p *Registry) FindIdent(identName string) (ref.Val, bool) {
	if t, found := p.revTypeMap[identName]; found {
		return t, true
	}
	if enumVal, found := p.pbdb.DescribeEnum(identName); found {
		return Int(enumVal.Value()), true
	}
	if p.provider != nil {
		return p.provider.FindIdent(identName)
	}
	return nil, false
}

// FindType looks up the Type given a qualified typeName. Returns false if not found.
//
// Deprecated: use FindStructType
func (p *Registry) FindType(structType string) (*exprpb.Type, bool) {
	structType = sanitizeStructTypeName(structType)
	if p.hasStructType(structType) {
		return makeExprMessageType(structType), true
	}
	if p.provider != nil {
		if tp, ok := p.provider.(ref.TypeProvider); ok {
			return tp.FindType(structType)
		}
		if _, ok := p.provider.FindStructType(structType); ok {
			return makeExprMessageType(structType), true
		}
	}
	return nil, false
}

// FindStructType returns the Type give a qualified type name.
//
// For historical reasons, only struct types are expected to be returned through this
// method, and the type values are expected to be wrapped in a TypeType instance using
// TypeTypeWithParam(<structType>).
//
// Returns false if not found.
func (p *Registry) FindStructType(structType string) (*Type, bool) {
	structType = sanitizeStructTypeName(structType)
	if p.hasStructType(structType) {
		return NewTypeTypeWithParam(NewObjectType(structType)), true
	}
	if p.provider != nil {
		return p.provider.FindStructType(structType)
	}
	return nil, false
}

// NewValue creates a new type value from a qualified name and map of field
// name to value.
//
// Note, for each value, the Val.ConvertToNative function will be invoked
// to convert the Val to the field's native type. If an error occurs during
// conversion, the NewValue will be a types.Err.
func (p *Registry) NewValue(structType string, fields map[string]ref.Val) ref.Val {
	structType = sanitizeStructTypeName(structType)
	if st, found := p.structTypes[structType]; found {
		return st.NewValue(p, fields)
	}
	td, found := p.pbdb.DescribeType(structType)
	if !found {
		if p.provider != nil {
			return p.provider.NewValue(structType, fields)
		}
		return NewErr("unknown type '%s'", structType)
	}
	msg := td.New()
	for name, value := range fields {
		field, found := td.FieldByName(name)
		if !found {
			return NewErr("no such field: %s", name)
		}
		err := msgSetField(msg, field, value)
		if err != nil {
			return &Err{error: err}
		}
	}
	return p.NativeToValue(msg.Interface())
}

// RegisterDescriptor registers the contents of a protocol buffer `FileDescriptor`.
func (p *Registry) RegisterDescriptor(fileDesc protoreflect.FileDescriptor) error {
	p.ensureMutable()
	fd, err := p.pbdb.RegisterDescriptor(fileDesc)
	if err != nil {
		return err
	}
	return p.registerAllTypes(fd)
}

// RegisterMessage registers a protocol buffer message and its dependencies.
func (p *Registry) RegisterMessage(message proto.Message) error {
	p.ensureMutable()
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
		existing, found := p.revTypeMap[t.TypeName()]
		celType := maybeForeignType(t)
		if found {
			if !existing.IsEquivalentType(celType) {
				return fmt.Errorf("type registration conflict. found: %v, input: %v", existing, celType)
			}
			if existing.traitMask != celType.traitMask {
				return fmt.Errorf(
					"type registered with conflicting traits: %v with traits %v, input: %v",
					existing.TypeName(), existing.traitMask, celType.traitMask)
			}
			continue
		}

		p.ensureMutable()
		typeName := t.TypeName()
		p.revTypeMap[typeName] = celType
		if st, ok := t.(StructTypeDescriptor); ok {
			// Conflicts are gated above so if we see a struct here, it's safe to register.
			p.structTypes[typeName] = st
			if rt := st.ReflectType(); rt != nil {
				p.reflectTypes[rt] = st
				if rt.Kind() == reflect.Ptr {
					p.reflectTypes[rt.Elem()] = st
				} else {
					p.reflectTypes[reflect.PointerTo(rt)] = st
				}
			}
		}
	}
	return nil
}

// RegisterNativeType creates nativeType instances for the given reflect.Type and registers them.
func (p *Registry) RegisterNativeType(refType reflect.Type) error {
	result, err := newNativeTypes(refType, p.nativeOptions.fieldNameHandler)
	if err != nil {
		return err
	}
	for _, nt := range result {
		if err := p.RegisterType(nt); err != nil {
			return err
		}
	}
	return nil
}

func (p *Registry) findStructDescriptorByReflectType(rt reflect.Type) (StructTypeDescriptor, bool) {
	if rt == nil {
		return nil, false
	}
	if st, found := p.reflectTypes[rt]; found {
		return st, true
	}
	if rt.Kind() == reflect.Ptr {
		if st, found := p.reflectTypes[rt.Elem()]; found {
			return st, true
		}
	} else {
		if st, found := p.reflectTypes[reflect.PointerTo(rt)]; found {
			return st, true
		}
	}
	return nil, false
}

// NativeToValue converts various "native" types to ref.Val with this specific implementation
// providing support for custom proto-based types.
//
// This method should be the inverse of ref.Val.ConvertToNative.
func (p *Registry) NativeToValue(value any) ref.Val {
	switch v := value.(type) {
	case nil:
		return NullValue
	case *Bool:
		if v != nil {
			return *v
		}
	case *Bytes:
		if v != nil {
			return *v
		}
	case *Double:
		if v != nil {
			return *v
		}
	case *Int:
		if v != nil {
			return *v
		}
	case *String:
		if v != nil {
			return *v
		}
	case *Uint:
		if v != nil {
			return *v
		}
	case ref.Val:
		return v
	case bool:
		return Bool(v)
	case int:
		return Int(v)
	case int32:
		return Int(v)
	case int64:
		return Int(v)
	case uint:
		return Uint(v)
	case uint32:
		return Uint(v)
	case uint64:
		return Uint(v)
	case float32:
		return Double(v)
	case float64:
		return Double(v)
	case string:
		return String(v)
	case *dpb.Duration:
		return Duration{Duration: v.AsDuration()}
	case time.Duration:
		return Duration{Duration: v}
	case *tpb.Timestamp:
		return Timestamp{Time: v.AsTime()}
	case time.Time:
		return Timestamp{Time: v}
	case *bool:
		if v != nil {
			return Bool(*v)
		}
	case *float32:
		if v != nil {
			return Double(*v)
		}
	case *float64:
		if v != nil {
			return Double(*v)
		}
	case *int:
		if v != nil {
			return Int(*v)
		}
	case *int32:
		if v != nil {
			return Int(*v)
		}
	case *int64:
		if v != nil {
			return Int(*v)
		}
	case *string:
		if v != nil {
			return String(*v)
		}
	case *uint:
		if v != nil {
			return Uint(*v)
		}
	case *uint32:
		if v != nil {
			return Uint(*v)
		}
	case *uint64:
		if v != nil {
			return Uint(*v)
		}
	case []byte:
		return Bytes(v)
	// specializations for common lists types.
	case []string:
		return NewStringList(p, v)
	case []ref.Val:
		return NewRefValList(p, v)
	// specializations for common map types.
	case map[string]string:
		return NewStringStringMap(p, v)
	case map[string]any:
		return NewStringInterfaceMap(p, v)
	case map[ref.Val]ref.Val:
		return NewRefValMap(p, v)
	// additional specializations may be added upon request / need.
	case *anypb.Any:
		if v == nil {
			return UnsupportedRefValConversionErr(v)
		}
		unpackedAny, err := v.UnmarshalNew()
		if err != nil {
			return NewErr("anypb.UnmarshalNew() failed for type %q: %v", v.GetTypeUrl(), err)
		}
		return p.NativeToValue(unpackedAny)
	case *structpb.NullValue, structpb.NullValue:
		return NullValue
	case *structpb.ListValue:
		return NewJSONList(p, v)
	case *structpb.Struct:
		return NewJSONStruct(p, v)
	case protoreflect.EnumNumber:
		return Int(v)
	case proto.Message:
		if v == nil {
			return UnsupportedRefValConversionErr(v)
		}
		typeName := string(v.ProtoReflect().Descriptor().FullName())
		pbdb := p.pbdb
		if pbdb == nil {
			pbdb = pb.DefaultDb
		}
		td, found := pbdb.DescribeType(typeName)
		if !found {
			if p.adapter != nil {
				return p.adapter.NativeToValue(value)
			}
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
	default:
		rt := reflect.TypeOf(value)
		if len(p.reflectTypes) > 0 {
			if st, found := p.findStructDescriptorByReflectType(rt); found {
				return st.Adapt(p, value)
			}
		}
		refVal := reflect.ValueOf(v)
		if refVal.Kind() == reflect.Ptr {
			if refVal.IsNil() {
				break
			}
			refVal = refVal.Elem()
		}
		switch refVal.Kind() {
		case reflect.Array, reflect.Slice:
			if refVal.Type().Elem() == reflect.TypeOf(byte(0)) {
				if refVal.CanAddr() {
					return Bytes(refVal.Bytes())
				}
				tmp := reflect.New(refVal.Type())
				tmp.Elem().Set(refVal)
				return Bytes(tmp.Elem().Bytes())
			}
			return NewDynamicList(p, v)
		case reflect.Map:
			return NewDynamicMap(p, v)
		case reflect.Bool:
			return Bool(refVal.Bool())
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return Int(refVal.Int())
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return Uint(refVal.Uint())
		case reflect.Float32, reflect.Float64:
			return Double(refVal.Float())
		case reflect.String:
			return String(refVal.String())
		}
	}
	if p.adapter != nil {
		return p.adapter.NativeToValue(value)
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

func (p *Registry) hasStructType(structType string) bool {
	if _, found := p.structTypes[structType]; found {
		return true
	}
	_, found := p.pbdb.DescribeType(structType)
	return found
}

func sanitizeStructTypeName(structType string) string {
	if len(structType) > 0 && structType[0] == '.' {
		return structType[1:]
	}
	return structType
}

func registerTypeItems(r *Registry, types ...any) error {
	opts := make([]any, 0, len(types))
	items := make([]any, 0, len(types))
	for _, t := range types {
		switch t.(type) {
		case NativeTypeOption:
			opts = append(opts, t)
		default:
			items = append(items, t)
		}
	}
	for _, opt := range opts {
		if err := registerTypeItem(r, opt); err != nil {
			return err
		}
	}
	for _, item := range items {
		if err := registerTypeItem(r, item); err != nil {
			return err
		}
	}
	return nil
}

func registerTypeItem(r *Registry, t any) error {
	switch v := t.(type) {
	case proto.Message:
		return r.RegisterMessage(v)
	case protoreflect.FileDescriptor:
		return r.RegisterDescriptor(v)
	case ref.Type:
		return r.RegisterType(v)
	case reflect.Type:
		return r.RegisterNativeType(v)
	case reflect.Value:
		return r.RegisterNativeType(v.Type())
	case NativeTypeOption:
		return v(&r.nativeOptions)
	case RegistryOption:
		_, err := v(r)
		return err
	default:
		return fmt.Errorf("unsupported type: %v (%T) must be reflect.Type or reflect.Value", t, t)
	}
}

func makeExprMessageType(structType string) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_Type{
			Type: &exprpb.Type{
				TypeKind: &exprpb.Type_MessageType{
					MessageType: structType,
				},
			},
		},
	}
}

func makeRefFieldType(t *exprpb.Type, isSet ref.FieldTester, getFrom ref.FieldGetter, isJSONField bool) *ref.FieldType {
	return &ref.FieldType{
		Type:        t,
		IsSet:       isSet,
		GetFrom:     getFrom,
		IsJSONField: isJSONField,
	}
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
	emptyRegistry      = &Registry{pbdb: pb.DefaultDb}
)

// NativeToValue implements the ref.TypeAdapter interface.
func (a *defaultTypeAdapter) NativeToValue(value any) ref.Val {
	return emptyRegistry.NativeToValue(value)
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
