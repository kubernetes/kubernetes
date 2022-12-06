/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cel

import (
	"fmt"
	"math"
	"time"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	"google.golang.org/protobuf/proto"
)

const (
	noMaxLength = math.MaxInt
)

// NewListType returns a parameterized list type with a specified element type.
func NewListType(elem *DeclType, maxItems int64) *DeclType {
	return &DeclType{
		name:         "list",
		ElemType:     elem,
		MaxElements:  maxItems,
		celType:      cel.ListType(elem.CelType()),
		defaultValue: NewListValue(),
		// a list can always be represented as [] in JSON, so hardcode the min size
		// to 2
		MinSerializedSize: 2,
	}
}

// NewMapType returns a parameterized map type with the given key and element types.
func NewMapType(key, elem *DeclType, maxProperties int64) *DeclType {
	return &DeclType{
		name:         "map",
		KeyType:      key,
		ElemType:     elem,
		MaxElements:  maxProperties,
		celType:      cel.MapType(key.CelType(), elem.CelType()),
		defaultValue: NewMapValue(),
		// a map can always be represented as {} in JSON, so hardcode the min size
		// to 2
		MinSerializedSize: 2,
	}
}

// NewObjectType creates an object type with a qualified name and a set of field declarations.
func NewObjectType(name string, fields map[string]*DeclField) *DeclType {
	t := &DeclType{
		name:      name,
		Fields:    fields,
		celType:   cel.ObjectType(name),
		traitMask: traits.FieldTesterType | traits.IndexerType,
		// an object could potentially be larger than the min size we default to here ({}),
		// but we rely upon the caller to change MinSerializedSize accordingly if they add
		// properties to the object
		MinSerializedSize: 2,
	}
	t.defaultValue = NewObjectValue(t)
	return t
}

func NewSimpleTypeWithMinSize(name string, celType *cel.Type, zeroVal ref.Val, minSize int64) *DeclType {
	return &DeclType{
		name:              name,
		celType:           celType,
		defaultValue:      zeroVal,
		MinSerializedSize: minSize,
	}
}

// DeclType represents the universal type descriptor for OpenAPIv3 types.
type DeclType struct {
	fmt.Stringer

	name string
	// Fields contains a map of escaped CEL identifier field names to field declarations.
	Fields      map[string]*DeclField
	KeyType     *DeclType
	ElemType    *DeclType
	TypeParam   bool
	Metadata    map[string]string
	MaxElements int64
	// MinSerializedSize represents the smallest possible size in bytes that
	// the DeclType could be serialized to in JSON.
	MinSerializedSize int64

	celType      *cel.Type
	traitMask    int
	defaultValue ref.Val
}

// MaybeAssignTypeName attempts to set the DeclType name to a fully qualified name, if the type
// is of `object` type.
//
// The DeclType must return true for `IsObject` or this assignment will error.
func (t *DeclType) MaybeAssignTypeName(name string) *DeclType {
	if t.IsObject() {
		objUpdated := false
		if t.name != "object" {
			name = t.name
		} else {
			objUpdated = true
		}
		fieldMap := make(map[string]*DeclField, len(t.Fields))
		for fieldName, field := range t.Fields {
			fieldType := field.Type
			fieldTypeName := fmt.Sprintf("%s.%s", name, fieldName)
			updated := fieldType.MaybeAssignTypeName(fieldTypeName)
			if updated == fieldType {
				fieldMap[fieldName] = field
				continue
			}
			objUpdated = true
			fieldMap[fieldName] = &DeclField{
				Name:         fieldName,
				Type:         updated,
				Required:     field.Required,
				enumValues:   field.enumValues,
				defaultValue: field.defaultValue,
			}
		}
		if !objUpdated {
			return t
		}
		return &DeclType{
			name:              name,
			Fields:            fieldMap,
			KeyType:           t.KeyType,
			ElemType:          t.ElemType,
			TypeParam:         t.TypeParam,
			Metadata:          t.Metadata,
			celType:           cel.ObjectType(name),
			traitMask:         t.traitMask,
			defaultValue:      t.defaultValue,
			MinSerializedSize: t.MinSerializedSize,
		}
	}
	if t.IsMap() {
		elemTypeName := fmt.Sprintf("%s.@elem", name)
		updated := t.ElemType.MaybeAssignTypeName(elemTypeName)
		if updated == t.ElemType {
			return t
		}
		return NewMapType(t.KeyType, updated, t.MaxElements)
	}
	if t.IsList() {
		elemTypeName := fmt.Sprintf("%s.@idx", name)
		updated := t.ElemType.MaybeAssignTypeName(elemTypeName)
		if updated == t.ElemType {
			return t
		}
		return NewListType(updated, t.MaxElements)
	}
	return t
}

// ExprType returns the CEL expression type of this declaration.
func (t *DeclType) ExprType() (*exprpb.Type, error) {
	return cel.TypeToExprType(t.celType)
}

// CelType returns the CEL type of this declaration.
func (t *DeclType) CelType() *cel.Type {
	return t.celType
}

// FindField returns the DeclField with the given name if present.
func (t *DeclType) FindField(name string) (*DeclField, bool) {
	f, found := t.Fields[name]
	return f, found
}

// HasTrait implements the CEL ref.Type interface making this type declaration suitable for use
// within the CEL evaluator.
func (t *DeclType) HasTrait(trait int) bool {
	if t.traitMask&trait == trait {
		return true
	}
	if t.defaultValue == nil {
		return false
	}
	_, isDecl := t.defaultValue.Type().(*DeclType)
	if isDecl {
		return false
	}
	return t.defaultValue.Type().HasTrait(trait)
}

// IsList returns whether the declaration is a `list` type which defines a parameterized element
// type, but not a parameterized key type or fields.
func (t *DeclType) IsList() bool {
	return t.KeyType == nil && t.ElemType != nil && t.Fields == nil
}

// IsMap returns whether the declaration is a 'map' type which defines parameterized key and
// element types, but not fields.
func (t *DeclType) IsMap() bool {
	return t.KeyType != nil && t.ElemType != nil && t.Fields == nil
}

// IsObject returns whether the declartion is an 'object' type which defined a set of typed fields.
func (t *DeclType) IsObject() bool {
	return t.KeyType == nil && t.ElemType == nil && t.Fields != nil
}

// String implements the fmt.Stringer interface method.
func (t *DeclType) String() string {
	return t.name
}

// TypeName returns the fully qualified type name for the DeclType.
func (t *DeclType) TypeName() string {
	return t.name
}

// DefaultValue returns the CEL ref.Val representing the default value for this object type,
// if one exists.
func (t *DeclType) DefaultValue() ref.Val {
	return t.defaultValue
}

// FieldTypeMap constructs a map of the field and object types nested within a given type.
func FieldTypeMap(path string, t *DeclType) map[string]*DeclType {
	if t.IsObject() && t.TypeName() != "object" {
		path = t.TypeName()
	}
	types := make(map[string]*DeclType)
	buildDeclTypes(path, t, types)
	return types
}

func buildDeclTypes(path string, t *DeclType, types map[string]*DeclType) {
	// Ensure object types are properly named according to where they appear in the schema.
	if t.IsObject() {
		// Hack to ensure that names are uniquely qualified and work well with the type
		// resolution steps which require fully qualified type names for field resolution
		// to function properly.
		types[t.TypeName()] = t
		for name, field := range t.Fields {
			fieldPath := fmt.Sprintf("%s.%s", path, name)
			buildDeclTypes(fieldPath, field.Type, types)
		}
	}
	// Map element properties to type names if needed.
	if t.IsMap() {
		mapElemPath := fmt.Sprintf("%s.@elem", path)
		buildDeclTypes(mapElemPath, t.ElemType, types)
		types[path] = t
	}
	// List element properties.
	if t.IsList() {
		listIdxPath := fmt.Sprintf("%s.@idx", path)
		buildDeclTypes(listIdxPath, t.ElemType, types)
		types[path] = t
	}
}

// DeclField describes the name, ordinal, and optionality of a field declaration within a type.
type DeclField struct {
	Name         string
	Type         *DeclType
	Required     bool
	enumValues   []interface{}
	defaultValue interface{}
}

func NewDeclField(name string, declType *DeclType, required bool, enumValues []interface{}, defaultValue interface{}) *DeclField {
	return &DeclField{
		Name:         name,
		Type:         declType,
		Required:     required,
		enumValues:   enumValues,
		defaultValue: defaultValue,
	}
}

// TypeName returns the string type name of the field.
func (f *DeclField) TypeName() string {
	return f.Type.TypeName()
}

// DefaultValue returns the zero value associated with the field.
func (f *DeclField) DefaultValue() ref.Val {
	if f.defaultValue != nil {
		return types.DefaultTypeAdapter.NativeToValue(f.defaultValue)
	}
	return f.Type.DefaultValue()
}

// EnumValues returns the set of values that this field may take.
func (f *DeclField) EnumValues() []ref.Val {
	if f.enumValues == nil || len(f.enumValues) == 0 {
		return []ref.Val{}
	}
	ev := make([]ref.Val, len(f.enumValues))
	for i, e := range f.enumValues {
		ev[i] = types.DefaultTypeAdapter.NativeToValue(e)
	}
	return ev
}

// NewRuleTypes returns an Open API Schema-based type-system which is CEL compatible.
func NewRuleTypes(kind string,
	declType *DeclType,
	res Resolver) (*RuleTypes, error) {
	// Note, if the schema indicates that it's actually based on another proto
	// then prefer the proto definition. For expressions in the proto, a new field
	// annotation will be needed to indicate the expected environment and type of
	// the expression.
	schemaTypes, err := newSchemaTypeProvider(kind, declType)
	if err != nil {
		return nil, err
	}
	if schemaTypes == nil {
		return nil, nil
	}
	return &RuleTypes{
		ruleSchemaDeclTypes: schemaTypes,
		resolver:            res,
	}, nil
}

// RuleTypes extends the CEL ref.TypeProvider interface and provides an Open API Schema-based
// type-system.
type RuleTypes struct {
	ref.TypeProvider
	ruleSchemaDeclTypes *schemaTypeProvider
	typeAdapter         ref.TypeAdapter
	resolver            Resolver
}

// EnvOptions returns a set of cel.EnvOption values which includes the declaration set
// as well as a custom ref.TypeProvider.
//
// Note, the standard declaration set includes 'rule' which is defined as the top-level rule-schema
// type if one is configured.
//
// If the RuleTypes value is nil, an empty []cel.EnvOption set is returned.
func (rt *RuleTypes) EnvOptions(tp ref.TypeProvider) ([]cel.EnvOption, error) {
	if rt == nil {
		return []cel.EnvOption{}, nil
	}
	var ta ref.TypeAdapter = types.DefaultTypeAdapter
	tpa, ok := tp.(ref.TypeAdapter)
	if ok {
		ta = tpa
	}
	rtWithTypes := &RuleTypes{
		TypeProvider:        tp,
		typeAdapter:         ta,
		ruleSchemaDeclTypes: rt.ruleSchemaDeclTypes,
		resolver:            rt.resolver,
	}
	for name, declType := range rt.ruleSchemaDeclTypes.types {
		tpType, found := tp.FindType(name)
		expT, err := declType.ExprType()
		if err != nil {
			return nil, fmt.Errorf("fail to get cel type: %s", err)
		}
		if found && !proto.Equal(tpType, expT) {
			return nil, fmt.Errorf(
				"type %s definition differs between CEL environment and rule", name)
		}
	}
	return []cel.EnvOption{
		cel.CustomTypeProvider(rtWithTypes),
		cel.CustomTypeAdapter(rtWithTypes),
		cel.Variable("rule", rt.ruleSchemaDeclTypes.root.CelType()),
	}, nil
}

// FindType attempts to resolve the typeName provided from the rule's rule-schema, or if not
// from the embedded ref.TypeProvider.
//
// FindType overrides the default type-finding behavior of the embedded TypeProvider.
//
// Note, when the type name is based on the Open API Schema, the name will reflect the object path
// where the type definition appears.
func (rt *RuleTypes) FindType(typeName string) (*exprpb.Type, bool) {
	if rt == nil {
		return nil, false
	}
	declType, found := rt.findDeclType(typeName)
	if found {
		expT, err := declType.ExprType()
		if err != nil {
			return expT, false
		}
		return expT, found
	}
	return rt.TypeProvider.FindType(typeName)
}

// FindDeclType returns the CPT type description which can be mapped to a CEL type.
func (rt *RuleTypes) FindDeclType(typeName string) (*DeclType, bool) {
	if rt == nil {
		return nil, false
	}
	return rt.findDeclType(typeName)
}

// FindFieldType returns a field type given a type name and field name, if found.
//
// Note, the type name for an Open API Schema type is likely to be its qualified object path.
// If, in the future an object instance rather than a type name were provided, the field
// resolution might more accurately reflect the expected type model. However, in this case
// concessions were made to align with the existing CEL interfaces.
func (rt *RuleTypes) FindFieldType(typeName, fieldName string) (*ref.FieldType, bool) {
	st, found := rt.findDeclType(typeName)
	if !found {
		return rt.TypeProvider.FindFieldType(typeName, fieldName)
	}

	f, found := st.Fields[fieldName]
	if found {
		ft := f.Type
		expT, err := ft.ExprType()
		if err != nil {
			return nil, false
		}
		return &ref.FieldType{
			Type: expT,
		}, true
	}
	// This could be a dynamic map.
	if st.IsMap() {
		et := st.ElemType
		expT, err := et.ExprType()
		if err != nil {
			return nil, false
		}
		return &ref.FieldType{
			Type: expT,
		}, true
	}
	return nil, false
}

// NativeToValue is an implementation of the ref.TypeAdapater interface which supports conversion
// of rule values to CEL ref.Val instances.
func (rt *RuleTypes) NativeToValue(val interface{}) ref.Val {
	return rt.typeAdapter.NativeToValue(val)
}

// TypeNames returns the list of type names declared within the RuleTypes object.
func (rt *RuleTypes) TypeNames() []string {
	typeNames := make([]string, len(rt.ruleSchemaDeclTypes.types))
	i := 0
	for name := range rt.ruleSchemaDeclTypes.types {
		typeNames[i] = name
		i++
	}
	return typeNames
}

func (rt *RuleTypes) findDeclType(typeName string) (*DeclType, bool) {
	declType, found := rt.ruleSchemaDeclTypes.types[typeName]
	if found {
		return declType, true
	}
	declType, found = rt.resolver.FindType(typeName)
	if found {
		return declType, true
	}
	return nil, false
}

func newSchemaTypeProvider(kind string, declType *DeclType) (*schemaTypeProvider, error) {
	if declType == nil {
		return nil, nil
	}
	root := declType.MaybeAssignTypeName(kind)
	types := FieldTypeMap(kind, root)
	return &schemaTypeProvider{
		root:  root,
		types: types,
	}, nil
}

type schemaTypeProvider struct {
	root  *DeclType
	types map[string]*DeclType
}

var (
	// AnyType is equivalent to the CEL 'protobuf.Any' type in that the value may have any of the
	// types supported.
	AnyType = NewSimpleTypeWithMinSize("any", cel.AnyType, nil, 1)

	// BoolType is equivalent to the CEL 'bool' type.
	BoolType = NewSimpleTypeWithMinSize("bool", cel.BoolType, types.False, MinBoolSize)

	// BytesType is equivalent to the CEL 'bytes' type.
	BytesType = NewSimpleTypeWithMinSize("bytes", cel.BytesType, types.Bytes([]byte{}), MinStringSize)

	// DoubleType is equivalent to the CEL 'double' type which is a 64-bit floating point value.
	DoubleType = NewSimpleTypeWithMinSize("double", cel.DoubleType, types.Double(0), MinNumberSize)

	// DurationType is equivalent to the CEL 'duration' type.
	DurationType = NewSimpleTypeWithMinSize("duration", cel.DurationType, types.Duration{Duration: time.Duration(0)}, MinDurationSizeJSON)

	// DateType is equivalent to the CEL 'date' type.
	DateType = NewSimpleTypeWithMinSize("date", cel.TimestampType, types.Timestamp{Time: time.Time{}}, JSONDateSize)

	// DynType is the equivalent of the CEL 'dyn' concept which indicates that the type will be
	// determined at runtime rather than compile time.
	DynType = NewSimpleTypeWithMinSize("dyn", cel.DynType, nil, 1)

	// IntType is equivalent to the CEL 'int' type which is a 64-bit signed int.
	IntType = NewSimpleTypeWithMinSize("int", cel.IntType, types.IntZero, MinNumberSize)

	// NullType is equivalent to the CEL 'null_type'.
	NullType = NewSimpleTypeWithMinSize("null_type", cel.NullType, types.NullValue, 4)

	// StringType is equivalent to the CEL 'string' type which is expected to be a UTF-8 string.
	// StringType values may either be string literals or expression strings.
	StringType = NewSimpleTypeWithMinSize("string", cel.StringType, types.String(""), MinStringSize)

	// TimestampType corresponds to the well-known protobuf.Timestamp type supported within CEL.
	// Note that both the OpenAPI date and date-time types map onto TimestampType, so not all types
	// labeled as Timestamp will necessarily have the same MinSerializedSize.
	TimestampType = NewSimpleTypeWithMinSize("timestamp", cel.TimestampType, types.Timestamp{Time: time.Time{}}, JSONDateSize)

	// UintType is equivalent to the CEL 'uint' type.
	UintType = NewSimpleTypeWithMinSize("uint", cel.UintType, types.Uint(0), 1)

	// ListType is equivalent to the CEL 'list' type.
	ListType = NewListType(AnyType, noMaxLength)

	// MapType is equivalent to the CEL 'map' type.
	MapType = NewMapType(AnyType, AnyType, noMaxLength)
)
