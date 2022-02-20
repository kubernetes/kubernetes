// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"fmt"
	"time"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	"google.golang.org/protobuf/proto"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// NewListType returns a parameterized list type with a specified element type.
func NewListType(elem *DeclType) *DeclType {
	return &DeclType{
		name:         "list",
		ElemType:     elem,
		exprType:     decls.NewListType(elem.ExprType()),
		defaultValue: NewListValue(),
	}
}

// NewMapType returns a parameterized map type with the given key and element types.
func NewMapType(key, elem *DeclType) *DeclType {
	return &DeclType{
		name:         "map",
		KeyType:      key,
		ElemType:     elem,
		exprType:     decls.NewMapType(key.ExprType(), elem.ExprType()),
		defaultValue: NewMapValue(),
	}
}

// NewObjectType creates an object type with a qualified name and a set of field declarations.
func NewObjectType(name string, fields map[string]*DeclField) *DeclType {
	t := &DeclType{
		name:      name,
		Fields:    fields,
		exprType:  decls.NewObjectType(name),
		traitMask: traits.FieldTesterType | traits.IndexerType,
	}
	t.defaultValue = NewObjectValue(t)
	return t
}

// NewObjectTypeRef returns a reference to an object type by name
func NewObjectTypeRef(name string) *DeclType {
	t := &DeclType{
		name:      name,
		exprType:  decls.NewObjectType(name),
		traitMask: traits.FieldTesterType | traits.IndexerType,
	}
	return t
}

// NewTypeParam creates a type parameter type with a simple name.
//
// Type parameters are resolved at compilation time to concrete types, or CEL 'dyn' type if no
// type assignment can be inferred.
func NewTypeParam(name string) *DeclType {
	return &DeclType{
		name:      name,
		TypeParam: true,
		exprType:  decls.NewTypeParamType(name),
	}
}

func newSimpleType(name string, exprType *exprpb.Type, zeroVal ref.Val) *DeclType {
	return &DeclType{
		name:         name,
		exprType:     exprType,
		defaultValue: zeroVal,
	}
}

// DeclType represents the universal type descriptor for OpenAPIv3 types.
type DeclType struct {
	fmt.Stringer

	name      string
	Fields    map[string]*DeclField
	KeyType   *DeclType
	ElemType  *DeclType
	TypeParam bool
	Metadata  map[string]string

	exprType     *exprpb.Type
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
			name:         name,
			Fields:       fieldMap,
			KeyType:      t.KeyType,
			ElemType:     t.ElemType,
			TypeParam:    t.TypeParam,
			Metadata:     t.Metadata,
			exprType:     decls.NewObjectType(name),
			traitMask:    t.traitMask,
			defaultValue: t.defaultValue,
		}
	}
	if t.IsMap() {
		elemTypeName := fmt.Sprintf("%s.@elem", name)
		updated := t.ElemType.MaybeAssignTypeName(elemTypeName)
		if updated == t.ElemType {
			return t
		}
		return NewMapType(t.KeyType, updated)
	}
	if t.IsList() {
		elemTypeName := fmt.Sprintf("%s.@idx", name)
		updated := t.ElemType.MaybeAssignTypeName(elemTypeName)
		if updated == t.ElemType {
			return t
		}
		return NewListType(updated)
	}
	return t
}

// ExprType returns the CEL expression type of this declaration.
func (t *DeclType) ExprType() *exprpb.Type {
	return t.exprType
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
	schema *schema.Structural,
	isResourceRoot bool,
	res Resolver) (*RuleTypes, error) {
	// Note, if the schema indicates that it's actually based on another proto
	// then prefer the proto definition. For expressions in the proto, a new field
	// annotation will be needed to indicate the expected environment and type of
	// the expression.
	schemaTypes, err := newSchemaTypeProvider(kind, schema, isResourceRoot)
	if err != nil {
		return nil, err
	}
	if schemaTypes == nil {
		return nil, nil
	}
	return &RuleTypes{
		Schema:              schema,
		ruleSchemaDeclTypes: schemaTypes,
		resolver:            res,
	}, nil
}

// RuleTypes extends the CEL ref.TypeProvider interface and provides an Open API Schema-based
// type-system.
type RuleTypes struct {
	ref.TypeProvider
	Schema              *schema.Structural
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
		Schema:              rt.Schema,
		ruleSchemaDeclTypes: rt.ruleSchemaDeclTypes,
		resolver:            rt.resolver,
	}
	for name, declType := range rt.ruleSchemaDeclTypes.types {
		tpType, found := tp.FindType(name)
		if found && !proto.Equal(tpType, declType.ExprType()) {
			return nil, fmt.Errorf(
				"type %s definition differs between CEL environment and rule", name)
		}
	}
	return []cel.EnvOption{
		cel.CustomTypeProvider(rtWithTypes),
		cel.CustomTypeAdapter(rtWithTypes),
		cel.Declarations(
			decls.NewVar("rule", rt.ruleSchemaDeclTypes.root.ExprType()),
		),
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
		return declType.ExprType(), found
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
		return &ref.FieldType{
			Type: ft.ExprType(),
		}, true
	}
	// This could be a dynamic map.
	if st.IsMap() {
		et := st.ElemType
		return &ref.FieldType{
			Type: et.ExprType(),
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

func (rt *RuleTypes) convertToCustomType(dyn *DynValue, declType *DeclType) *DynValue {
	switch v := dyn.Value().(type) {
	case *MapValue:
		if declType.IsObject() {
			obj := v.ConvertToObject(declType)
			for name, f := range obj.fieldMap {
				field := declType.Fields[name]
				f.Ref = rt.convertToCustomType(f.Ref, field.Type)
			}
			dyn.SetValue(obj)
			return dyn
		}
		fieldType := declType.ElemType
		for _, f := range v.fieldMap {
			f.Ref = rt.convertToCustomType(f.Ref, fieldType)
		}
		return dyn
	case *ListValue:
		for i := 0; i < len(v.Entries); i++ {
			elem := v.Entries[i]
			elem = rt.convertToCustomType(elem, declType.ElemType)
			v.Entries[i] = elem
		}
		return dyn
	default:
		return dyn
	}
}

func newSchemaTypeProvider(kind string, schema *schema.Structural, isResourceRoot bool) (*schemaTypeProvider, error) {
	delType := SchemaDeclType(schema, isResourceRoot)
	if delType == nil {
		return nil, nil
	}
	root := delType.MaybeAssignTypeName(kind)
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
	AnyType = newSimpleType("any", decls.Any, nil)

	// BoolType is equivalent to the CEL 'bool' type.
	BoolType = newSimpleType("bool", decls.Bool, types.False)

	// BytesType is equivalent to the CEL 'bytes' type.
	BytesType = newSimpleType("bytes", decls.Bytes, types.Bytes([]byte{}))

	// DoubleType is equivalent to the CEL 'double' type which is a 64-bit floating point value.
	DoubleType = newSimpleType("double", decls.Double, types.Double(0))

	// DurationType is equivalent to the CEL 'duration' type.
	DurationType = newSimpleType("duration", decls.Duration, types.Duration{Duration: time.Duration(0)})

	// DateType is equivalent to the CEL 'date' type.
	DateType = newSimpleType("date", decls.Timestamp, types.Timestamp{Time: time.Time{}})

	// DynType is the equivalent of the CEL 'dyn' concept which indicates that the type will be
	// determined at runtime rather than compile time.
	DynType = newSimpleType("dyn", decls.Dyn, nil)

	// IntType is equivalent to the CEL 'int' type which is a 64-bit signed int.
	IntType = newSimpleType("int", decls.Int, types.IntZero)

	// NullType is equivalent to the CEL 'null_type'.
	NullType = newSimpleType("null_type", decls.Null, types.NullValue)

	// StringType is equivalent to the CEL 'string' type which is expected to be a UTF-8 string.
	// StringType values may either be string literals or expression strings.
	StringType = newSimpleType("string", decls.String, types.String(""))

	// TimestampType corresponds to the well-known protobuf.Timestamp type supported within CEL.
	TimestampType = newSimpleType("timestamp", decls.Timestamp, types.Timestamp{Time: time.Time{}})

	// UintType is equivalent to the CEL 'uint' type.
	UintType = newSimpleType("uint", decls.Uint, types.Uint(0))

	// ListType is equivalent to the CEL 'list' type.
	ListType = NewListType(AnyType)

	// MapType is equivalent to the CEL 'map' type.
	MapType = NewMapType(AnyType, AnyType)
)
