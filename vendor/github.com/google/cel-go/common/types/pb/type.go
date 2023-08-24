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

package pb

import (
	"fmt"
	"reflect"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	dynamicpb "google.golang.org/protobuf/types/dynamicpb"
	anypb "google.golang.org/protobuf/types/known/anypb"
	dpb "google.golang.org/protobuf/types/known/durationpb"
	structpb "google.golang.org/protobuf/types/known/structpb"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
	wrapperspb "google.golang.org/protobuf/types/known/wrapperspb"
)

// description is a private interface used to make it convenient to perform type unwrapping at
// the TypeDescription or FieldDescription level.
type description interface {
	// Zero returns an empty immutable protobuf message when the description is a protobuf message
	// type.
	Zero() proto.Message
}

// newTypeDescription produces a TypeDescription value for the fully-qualified proto type name
// with a given descriptor.
func newTypeDescription(typeName string, desc protoreflect.MessageDescriptor, extensions extensionMap) *TypeDescription {
	msgType := dynamicpb.NewMessageType(desc)
	msgZero := dynamicpb.NewMessage(desc)
	fieldMap := map[string]*FieldDescription{}
	fields := desc.Fields()
	for i := 0; i < fields.Len(); i++ {
		f := fields.Get(i)
		fieldMap[string(f.Name())] = newFieldDescription(f)
	}
	return &TypeDescription{
		typeName:    typeName,
		desc:        desc,
		msgType:     msgType,
		fieldMap:    fieldMap,
		extensions:  extensions,
		reflectType: reflectTypeOf(msgZero),
		zeroMsg:     zeroValueOf(msgZero),
	}
}

// TypeDescription is a collection of type metadata relevant to expression
// checking and evaluation.
type TypeDescription struct {
	typeName    string
	desc        protoreflect.MessageDescriptor
	msgType     protoreflect.MessageType
	fieldMap    map[string]*FieldDescription
	extensions  extensionMap
	reflectType reflect.Type
	zeroMsg     proto.Message
}

// Copy copies the type description with updated references to the Db.
func (td *TypeDescription) Copy(pbdb *Db) *TypeDescription {
	return &TypeDescription{
		typeName:    td.typeName,
		desc:        td.desc,
		msgType:     td.msgType,
		fieldMap:    td.fieldMap,
		extensions:  pbdb.extensions,
		reflectType: td.reflectType,
		zeroMsg:     td.zeroMsg,
	}
}

// FieldMap returns a string field name to FieldDescription map.
func (td *TypeDescription) FieldMap() map[string]*FieldDescription {
	return td.fieldMap
}

// FieldByName returns (FieldDescription, true) if the field name is declared within the type.
func (td *TypeDescription) FieldByName(name string) (*FieldDescription, bool) {
	fd, found := td.fieldMap[name]
	if found {
		return fd, true
	}
	extFieldMap, found := td.extensions[td.typeName]
	if !found {
		return nil, false
	}
	fd, found = extFieldMap[name]
	return fd, found
}

// MaybeUnwrap accepts a proto message as input and unwraps it to a primitive CEL type if possible.
//
// This method returns the unwrapped value and 'true', else the original value and 'false'.
func (td *TypeDescription) MaybeUnwrap(msg proto.Message) (any, bool, error) {
	return unwrap(td, msg)
}

// Name returns the fully-qualified name of the type.
func (td *TypeDescription) Name() string {
	return string(td.desc.FullName())
}

// New returns a mutable proto message
func (td *TypeDescription) New() protoreflect.Message {
	return td.msgType.New()
}

// ReflectType returns the Golang reflect.Type for this type.
func (td *TypeDescription) ReflectType() reflect.Type {
	return td.reflectType
}

// Zero returns the zero proto.Message value for this type.
func (td *TypeDescription) Zero() proto.Message {
	return td.zeroMsg
}

// newFieldDescription creates a new field description from a protoreflect.FieldDescriptor.
func newFieldDescription(fieldDesc protoreflect.FieldDescriptor) *FieldDescription {
	var reflectType reflect.Type
	var zeroMsg proto.Message
	switch fieldDesc.Kind() {
	case protoreflect.EnumKind:
		reflectType = reflectTypeOf(protoreflect.EnumNumber(0))
	case protoreflect.GroupKind, protoreflect.MessageKind:
		zeroMsg = dynamicpb.NewMessage(fieldDesc.Message())
		reflectType = reflectTypeOf(zeroMsg)
	default:
		reflectType = reflectTypeOf(fieldDesc.Default().Interface())
		if fieldDesc.IsList() {
			var elemValue protoreflect.Value
			if fieldDesc.IsExtension() {
				et := dynamicpb.NewExtensionType(fieldDesc)
				elemValue = et.New().List().NewElement()
			} else {
				parentMsgType := fieldDesc.ContainingMessage()
				parentMsg := dynamicpb.NewMessage(parentMsgType)
				listField := parentMsg.NewField(fieldDesc).List()
				elemValue = listField.NewElement()
			}
			elem := elemValue.Interface()
			switch elemType := elem.(type) {
			case protoreflect.Message:
				elem = elemType.Interface()
			}
			reflectType = reflectTypeOf(elem)
		}
	}
	// Ensure the list type is appropriately reflected as a Go-native list.
	if fieldDesc.IsList() {
		reflectType = reflect.SliceOf(reflectType)
	}
	var keyType, valType *FieldDescription
	if fieldDesc.IsMap() {
		keyType = newFieldDescription(fieldDesc.MapKey())
		valType = newFieldDescription(fieldDesc.MapValue())
	}
	return &FieldDescription{
		desc:        fieldDesc,
		KeyType:     keyType,
		ValueType:   valType,
		reflectType: reflectType,
		zeroMsg:     zeroValueOf(zeroMsg),
	}
}

// FieldDescription holds metadata related to fields declared within a type.
type FieldDescription struct {
	// KeyType holds the key FieldDescription for map fields.
	KeyType *FieldDescription
	// ValueType holds the value FieldDescription for map fields.
	ValueType *FieldDescription

	desc        protoreflect.FieldDescriptor
	reflectType reflect.Type
	zeroMsg     proto.Message
}

// CheckedType returns the type-definition used at type-check time.
func (fd *FieldDescription) CheckedType() *exprpb.Type {
	if fd.desc.IsMap() {
		return &exprpb.Type{
			TypeKind: &exprpb.Type_MapType_{
				MapType: &exprpb.Type_MapType{
					KeyType:   fd.KeyType.typeDefToType(),
					ValueType: fd.ValueType.typeDefToType(),
				},
			},
		}
	}
	if fd.desc.IsList() {
		return &exprpb.Type{
			TypeKind: &exprpb.Type_ListType_{
				ListType: &exprpb.Type_ListType{
					ElemType: fd.typeDefToType()}}}
	}
	return fd.typeDefToType()
}

// Descriptor returns the protoreflect.FieldDescriptor for this type.
func (fd *FieldDescription) Descriptor() protoreflect.FieldDescriptor {
	return fd.desc
}

// IsSet returns whether the field is set on the target value, per the proto presence conventions
// of proto2 or proto3 accordingly.
//
// This function implements the FieldType.IsSet function contract which can be used to operate on
// more than just protobuf field accesses; however, the target here must be a protobuf.Message.
func (fd *FieldDescription) IsSet(target any) bool {
	switch v := target.(type) {
	case proto.Message:
		pbRef := v.ProtoReflect()
		pbDesc := pbRef.Descriptor()
		if pbDesc == fd.desc.ContainingMessage() {
			// When the target protobuf shares the same message descriptor instance as the field
			// descriptor, use the cached field descriptor value.
			return pbRef.Has(fd.desc)
		}
		// Otherwise, fallback to a dynamic lookup of the field descriptor from the target
		// instance as an attempt to use the cached field descriptor will result in a panic.
		return pbRef.Has(pbDesc.Fields().ByName(protoreflect.Name(fd.Name())))
	default:
		return false
	}
}

// GetFrom returns the accessor method associated with the field on the proto generated struct.
//
// If the field is not set, the proto default value is returned instead.
//
// This function implements the FieldType.GetFrom function contract which can be used to operate
// on more than just protobuf field accesses; however, the target here must be a protobuf.Message.
func (fd *FieldDescription) GetFrom(target any) (any, error) {
	v, ok := target.(proto.Message)
	if !ok {
		return nil, fmt.Errorf("unsupported field selection target: (%T)%v", target, target)
	}
	pbRef := v.ProtoReflect()
	pbDesc := pbRef.Descriptor()
	var fieldVal any
	if pbDesc == fd.desc.ContainingMessage() {
		// When the target protobuf shares the same message descriptor instance as the field
		// descriptor, use the cached field descriptor value.
		fieldVal = pbRef.Get(fd.desc).Interface()
	} else {
		// Otherwise, fallback to a dynamic lookup of the field descriptor from the target
		// instance as an attempt to use the cached field descriptor will result in a panic.
		fieldVal = pbRef.Get(pbDesc.Fields().ByName(protoreflect.Name(fd.Name()))).Interface()
	}
	switch fv := fieldVal.(type) {
	// Fast-path return for primitive types.
	case bool, []byte, float32, float64, int32, int64, string, uint32, uint64, protoreflect.List:
		return fv, nil
	case protoreflect.EnumNumber:
		return int64(fv), nil
	case protoreflect.Map:
		// Return a wrapper around the protobuf-reflected Map types which carries additional
		// information about the key and value definitions of the map.
		return &Map{Map: fv, KeyType: fd.KeyType, ValueType: fd.ValueType}, nil
	case protoreflect.Message:
		// Make sure to unwrap well-known protobuf types before returning.
		unwrapped, _, err := fd.MaybeUnwrapDynamic(fv)
		return unwrapped, err
	default:
		return fv, nil
	}
}

// IsEnum returns true if the field type refers to an enum value.
func (fd *FieldDescription) IsEnum() bool {
	return fd.ProtoKind() == protoreflect.EnumKind
}

// IsMap returns true if the field is of map type.
func (fd *FieldDescription) IsMap() bool {
	return fd.desc.IsMap()
}

// IsMessage returns true if the field is of message type.
func (fd *FieldDescription) IsMessage() bool {
	kind := fd.ProtoKind()
	return kind == protoreflect.MessageKind || kind == protoreflect.GroupKind
}

// IsOneof returns true if the field is declared within a oneof block.
func (fd *FieldDescription) IsOneof() bool {
	return fd.desc.ContainingOneof() != nil
}

// IsList returns true if the field is a repeated value.
//
// This method will also return true for map values, so check whether the
// field is also a map.
func (fd *FieldDescription) IsList() bool {
	return fd.desc.IsList()
}

// MaybeUnwrapDynamic takes the reflected protoreflect.Message and determines whether the
// value can be unwrapped to a more primitive CEL type.
//
// This function returns the unwrapped value and 'true' on success, or the original value
// and 'false' otherwise.
func (fd *FieldDescription) MaybeUnwrapDynamic(msg protoreflect.Message) (any, bool, error) {
	return unwrapDynamic(fd, msg)
}

// Name returns the CamelCase name of the field within the proto-based struct.
func (fd *FieldDescription) Name() string {
	return string(fd.desc.Name())
}

// ProtoKind returns the protobuf reflected kind of the field.
func (fd *FieldDescription) ProtoKind() protoreflect.Kind {
	return fd.desc.Kind()
}

// ReflectType returns the Golang reflect.Type for this field.
func (fd *FieldDescription) ReflectType() reflect.Type {
	return fd.reflectType
}

// String returns the fully qualified name of the field within its type as well as whether the
// field occurs within a oneof.
func (fd *FieldDescription) String() string {
	return fmt.Sprintf("%v.%s `oneof=%t`", fd.desc.ContainingMessage().FullName(), fd.Name(), fd.IsOneof())
}

// Zero returns the zero value for the protobuf message represented by this field.
//
// If the field is not a proto.Message type, the zero value is nil.
func (fd *FieldDescription) Zero() proto.Message {
	return fd.zeroMsg
}

func (fd *FieldDescription) typeDefToType() *exprpb.Type {
	if fd.IsMessage() {
		msgType := string(fd.desc.Message().FullName())
		if wk, found := CheckedWellKnowns[msgType]; found {
			return wk
		}
		return checkedMessageType(msgType)
	}
	if fd.IsEnum() {
		return checkedInt
	}
	return CheckedPrimitives[fd.ProtoKind()]
}

// Map wraps the protoreflect.Map object with a key and value FieldDescription for use in
// retrieving individual elements within CEL value data types.
type Map struct {
	protoreflect.Map
	KeyType   *FieldDescription
	ValueType *FieldDescription
}

func checkedMessageType(name string) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_MessageType{MessageType: name}}
}

func checkedPrimitive(primitive exprpb.Type_PrimitiveType) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_Primitive{Primitive: primitive}}
}

func checkedWellKnown(wellKnown exprpb.Type_WellKnownType) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_WellKnown{WellKnown: wellKnown}}
}

func checkedWrap(t *exprpb.Type) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_Wrapper{Wrapper: t.GetPrimitive()}}
}

// unwrap unwraps the provided proto.Message value, potentially based on the description if the
// input message is a *dynamicpb.Message which obscures the typing information from Go.
//
// Returns the unwrapped value and 'true' if unwrapped, otherwise the input value and 'false'.
func unwrap(desc description, msg proto.Message) (any, bool, error) {
	switch v := msg.(type) {
	case *anypb.Any:
		dynMsg, err := v.UnmarshalNew()
		if err != nil {
			return v, false, err
		}
		return unwrapDynamic(desc, dynMsg.ProtoReflect())
	case *dynamicpb.Message:
		return unwrapDynamic(desc, v)
	case *dpb.Duration:
		return v.AsDuration(), true, nil
	case *tpb.Timestamp:
		return v.AsTime(), true, nil
	case *structpb.Value:
		switch v.GetKind().(type) {
		case *structpb.Value_BoolValue:
			return v.GetBoolValue(), true, nil
		case *structpb.Value_ListValue:
			return v.GetListValue(), true, nil
		case *structpb.Value_NullValue:
			return structpb.NullValue_NULL_VALUE, true, nil
		case *structpb.Value_NumberValue:
			return v.GetNumberValue(), true, nil
		case *structpb.Value_StringValue:
			return v.GetStringValue(), true, nil
		case *structpb.Value_StructValue:
			return v.GetStructValue(), true, nil
		default:
			return structpb.NullValue_NULL_VALUE, true, nil
		}
	case *wrapperspb.BoolValue:
		return v.GetValue(), true, nil
	case *wrapperspb.BytesValue:
		return v.GetValue(), true, nil
	case *wrapperspb.DoubleValue:
		return v.GetValue(), true, nil
	case *wrapperspb.FloatValue:
		return float64(v.GetValue()), true, nil
	case *wrapperspb.Int32Value:
		return int64(v.GetValue()), true, nil
	case *wrapperspb.Int64Value:
		return v.GetValue(), true, nil
	case *wrapperspb.StringValue:
		return v.GetValue(), true, nil
	case *wrapperspb.UInt32Value:
		return uint64(v.GetValue()), true, nil
	case *wrapperspb.UInt64Value:
		return v.GetValue(), true, nil
	}
	return msg, false, nil
}

// unwrapDynamic unwraps a reflected protobuf Message value.
//
// Returns the unwrapped value and 'true' if unwrapped, otherwise the input value and 'false'.
func unwrapDynamic(desc description, refMsg protoreflect.Message) (any, bool, error) {
	msg := refMsg.Interface()
	if !refMsg.IsValid() {
		msg = desc.Zero()
	}
	// In order to ensure that these wrapped types match the expectations of the CEL type system
	// the dynamicpb.Message must be merged with an protobuf instance of the well-known type value.
	typeName := string(refMsg.Descriptor().FullName())
	switch typeName {
	case "google.protobuf.Any":
		// Note, Any values require further unwrapping; however, this unwrapping may or may not
		// be to a well-known type. If the unwrapped value is a well-known type it will be further
		// unwrapped before being returned to the caller. Otherwise, the dynamic protobuf object
		// represented by the Any will be returned.
		unwrappedAny := &anypb.Any{}
		err := Merge(unwrappedAny, msg)
		if err != nil {
			return nil, false, fmt.Errorf("unwrap dynamic field failed: %v", err)
		}
		dynMsg, err := unwrappedAny.UnmarshalNew()
		if err != nil {
			// Allow the error to move further up the stack as it should result in an type
			// conversion error if the caller does not recover it somehow.
			return nil, false, fmt.Errorf("unmarshal dynamic any failed: %v", err)
		}
		// Attempt to unwrap the dynamic type, otherwise return the dynamic message.
		unwrapped, nested, err := unwrapDynamic(desc, dynMsg.ProtoReflect())
		if err == nil && nested {
			return unwrapped, true, nil
		}
		return dynMsg, true, err
	case "google.protobuf.BoolValue",
		"google.protobuf.BytesValue",
		"google.protobuf.DoubleValue",
		"google.protobuf.FloatValue",
		"google.protobuf.Int32Value",
		"google.protobuf.Int64Value",
		"google.protobuf.StringValue",
		"google.protobuf.UInt32Value",
		"google.protobuf.UInt64Value":
		// The msg value is ignored when dealing with wrapper types as they have a null or value
		// behavior, rather than the standard zero value behavior of other proto message types.
		if !refMsg.IsValid() {
			return structpb.NullValue_NULL_VALUE, true, nil
		}
		valueField := refMsg.Descriptor().Fields().ByName("value")
		return refMsg.Get(valueField).Interface(), true, nil
	case "google.protobuf.Duration":
		unwrapped := &dpb.Duration{}
		err := Merge(unwrapped, msg)
		if err != nil {
			return nil, false, err
		}
		return unwrapped.AsDuration(), true, nil
	case "google.protobuf.ListValue":
		unwrapped := &structpb.ListValue{}
		err := Merge(unwrapped, msg)
		if err != nil {
			return nil, false, err
		}
		return unwrapped, true, nil
	case "google.protobuf.NullValue":
		return structpb.NullValue_NULL_VALUE, true, nil
	case "google.protobuf.Struct":
		unwrapped := &structpb.Struct{}
		err := Merge(unwrapped, msg)
		if err != nil {
			return nil, false, err
		}
		return unwrapped, true, nil
	case "google.protobuf.Timestamp":
		unwrapped := &tpb.Timestamp{}
		err := Merge(unwrapped, msg)
		if err != nil {
			return nil, false, err
		}
		return unwrapped.AsTime(), true, nil
	case "google.protobuf.Value":
		unwrapped := &structpb.Value{}
		err := Merge(unwrapped, msg)
		if err != nil {
			return nil, false, err
		}
		return unwrap(desc, unwrapped)
	}
	return msg, false, nil
}

// reflectTypeOf intercepts the reflect.Type call to ensure that dynamicpb.Message types preserve
// well-known protobuf reflected types expected by the CEL type system.
func reflectTypeOf(val any) reflect.Type {
	switch v := val.(type) {
	case proto.Message:
		return reflect.TypeOf(zeroValueOf(v))
	default:
		return reflect.TypeOf(v)
	}
}

// zeroValueOf will return the strongest possible proto.Message representing the default protobuf
// message value of the input msg type.
func zeroValueOf(msg proto.Message) proto.Message {
	if msg == nil {
		return nil
	}
	typeName := string(msg.ProtoReflect().Descriptor().FullName())
	zeroVal, found := zeroValueMap[typeName]
	if found {
		return zeroVal
	}
	return msg
}

var (
	jsonValueTypeURL = "types.googleapis.com/google.protobuf.Value"

	zeroValueMap = map[string]proto.Message{
		"google.protobuf.Any":         &anypb.Any{TypeUrl: jsonValueTypeURL},
		"google.protobuf.Duration":    &dpb.Duration{},
		"google.protobuf.ListValue":   &structpb.ListValue{},
		"google.protobuf.Struct":      &structpb.Struct{},
		"google.protobuf.Timestamp":   &tpb.Timestamp{},
		"google.protobuf.Value":       &structpb.Value{},
		"google.protobuf.BoolValue":   wrapperspb.Bool(false),
		"google.protobuf.BytesValue":  wrapperspb.Bytes([]byte{}),
		"google.protobuf.DoubleValue": wrapperspb.Double(0.0),
		"google.protobuf.FloatValue":  wrapperspb.Float(0.0),
		"google.protobuf.Int32Value":  wrapperspb.Int32(0),
		"google.protobuf.Int64Value":  wrapperspb.Int64(0),
		"google.protobuf.StringValue": wrapperspb.String(""),
		"google.protobuf.UInt32Value": wrapperspb.UInt32(0),
		"google.protobuf.UInt64Value": wrapperspb.UInt64(0),
	}
)
