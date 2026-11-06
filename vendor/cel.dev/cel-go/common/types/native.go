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
	"sync/atomic"
	"time"
	"unsafe"

	"google.golang.org/protobuf/reflect/protoreflect"

	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/common/types/traits"

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
	return newNativeType(rawType, tpOptions)
}

// NativeTypesFieldNameHandler is a handler for mapping a reflect.StructField to a CEL field name.
// This can be used to override the default Go struct field to CEL field name mapping.
type NativeTypesFieldNameHandler = func(field reflect.StructField) string

// NativeTypeOptions holds options for native types.
type NativeTypeOptions struct {
	fieldNameHandler NativeTypesFieldNameHandler
	typeName         string
	adapter          Adapter
}

// NativeTypeOption is a functional option for configuring handling of native types.
type NativeTypeOption func(*NativeTypeOptions) error

// NativeTypeAdapter sets the TypeAdapter to use when adapting nested dynamic fields for a NativeType.
func NativeTypeAdapter(adapter Adapter) NativeTypeOption {
	return func(opts *NativeTypeOptions) error {
		opts.adapter = adapter
		return nil
	}
}

// NativeTypeAlias configures a custom CEL type name (alias) for a native type.
func NativeTypeAlias(alias string) NativeTypeOption {
	return func(opts *NativeTypeOptions) error {
		opts.typeName = alias
		return nil
	}
}

// NativeTypeDesc describes a native Go struct type for registration with CEL.
type NativeTypeDesc struct {
	refType reflect.Type
	options []NativeTypeOption
}

// ReflectType returns the reflect.Type wrapped by this descriptor.
func (d *NativeTypeDesc) ReflectType() reflect.Type {
	return d.refType
}

// Options returns the NativeTypeOption slice for this descriptor.
func (d *NativeTypeDesc) Options() []NativeTypeOption {
	return d.options
}

// NativeTypeFor constructs a NativeTypeDesc for a Go struct type T with the given options.
func NativeTypeFor[T any](opts ...NativeTypeOption) *NativeTypeDesc {
	return &NativeTypeDesc{
		refType: reflect.TypeFor[T](),
		options: opts,
	}
}

// ParseStructTags configures if native types field names should be overridable by CEL struct tags.
// This is equivalent to ParseStructTag("cel").
//
// A tag starting with "-" (e.g. `cel:"-"` or `cel:"-,"`) marks the field as skipped.
// A literal "-" can be specified by single-quoting the name (e.g. `cel:"'-'"`).
//
// Skipping an embedded struct also skips the fields promoted from it, so they cannot be
// reached through the enclosing struct.
//
// The configured tag only governs what is visible to CEL expressions. Serialization, whether
// via `ConvertToNative` to a protobuf `Struct`/`Value` or via the `json.encode` and
// `yaml.encode` extension functions, is governed by the `json` tag. A field tagged
// `cel:"-"` is therefore still serialized unless it is also tagged `json:"-"`, and a field
// tagged `json:"-"` is still visible to CEL unless it is also tagged `cel:"-"`.
// Configuring ParseStructTag("json") makes CEL visibility and serialization agree.
func ParseStructTags(enabled bool) NativeTypeOption {
	if enabled {
		return ParseStructTag("cel")
	}
	return ParseStructField(nil)
}

// ParseStructTag configures the struct tag to parse. The 0th item in the tag is used as the name of the CEL field.
//
// A tag starting with "-" (e.g. `cel:"-"` or `cel:"-,"`) marks the field as skipped.
// A literal "-" can be specified by single-quoting the name (e.g. `cel:"'-'"`).
//
// Skipping an embedded struct also skips the fields promoted from it, so they cannot be
// reached through the enclosing struct.
//
// The configured tag only governs what is visible to CEL expressions; serialization is always
// governed by the `json` tag. See ParseStructTags for details.
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
		tagInfo := parseStructTag(field, structTagToParse, field.Name)
		if tagInfo.Skip {
			return "-"
		}
		return tagInfo.Name
	}
}

func isSkippedFieldName(name string) bool {
	return name == "" || name == "-"
}

type nativeJSONField struct {
	index     []int
	jsonName  string
	omitEmpty bool
	hasTag    bool
}

// NativeType represents a CEL struct type descriptor generated from a native Go struct.
type NativeType struct {
	typeName     string
	refType      reflect.Type
	fieldsByName map[string]reflect.StructField
	jsonFields   []nativeJSONField
	adapter      Adapter
}

// Clone creates a copy of the NativeType with optional configuration overrides.
func (t *NativeType) Clone(opts ...NativeTypeOption) (*NativeType, error) {
	if t == nil {
		return nil, nil
	}
	if len(opts) == 0 {
		cpy := *t
		return &cpy, nil
	}
	tpOptions := NativeTypeOptions{
		typeName: t.typeName,
		adapter:  t.adapter,
	}
	for _, opt := range opts {
		if err := opt(&tpOptions); err != nil {
			return nil, err
		}
	}
	if tpOptions.fieldNameHandler == nil {
		return &NativeType{
			typeName:     tpOptions.typeName,
			refType:      t.refType,
			fieldsByName: t.fieldsByName,
			jsonFields:   t.jsonFields,
			adapter:      tpOptions.adapter,
		}, nil
	}
	return newNativeType(t.refType, tpOptions)
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
	var structPtr unsafe.Pointer
	if refVal.Kind() == reflect.Ptr {
		if refVal.IsNil() {
			return NullValue
		}
		structPtr = unsafe.Pointer(refVal.Pointer())
		refVal = refVal.Elem()
	} else if refVal.Kind() == reflect.Struct {
		if !isDirectIface(t.refType) {
			structPtr = anyDataPtr(value)
		} else {
			ptr := reflect.New(t.refType)
			ptr.Elem().Set(refVal)
			structPtr = unsafe.Pointer(ptr.Pointer())
		}
	}
	return &nativeObj{
		adapter:   adapter,
		val:       value,
		valType:   t,
		refValue:  refVal,
		structPtr: structPtr,
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

// NativeToValue implements the ref.TypeAdapter interface method.
func (t *NativeType) NativeToValue(value any) ref.Val {
	return t.adapterFrom(nil).NativeToValue(value)
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
		Type:    celType,
		IsSet:   t.makeFieldTester(refField),
		GetFrom: t.makeFieldGetter(refField),
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
	adapter   Adapter
	val       any
	valType   *NativeType
	refValue  reflect.Value
	structPtr unsafe.Pointer
	aggSize   uint32
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
		fields := make(map[string]*structpb.Value, len(o.valType.jsonFields))
		for _, jf := range o.valType.jsonFields {
			fieldValue := safeGetFieldByIndex(refVal, jf.index)
			if !fieldValue.IsValid() {
				continue
			}
			if fieldValue.IsZero() {
				if !jf.hasTag || jf.omitEmpty || fieldValue.Kind() == reflect.Pointer || fieldValue.Kind() == reflect.Slice || fieldValue.Kind() == reflect.Map || fieldValue.Kind() == reflect.Interface {
					continue
				}
			}
			fieldCELVal := o.NativeToValue(fieldValue.Interface())
			fieldJSONVal, err := fieldCELVal.ConvertToNative(jsonValueType)
			if err != nil {
				return nil, err
			}
			fields[jf.jsonName] = fieldJSONVal.(*structpb.Value)
		}
		return &structpb.Struct{Fields: fields}, nil
	}
	return nil, fmt.Errorf("type conversion error from '%v' to '%v'", o.Type(), typeDesc)
}

// NativeToValue implements the ref.TypeAdapter interface method.
func (o *nativeObj) NativeToValue(value any) ref.Val {
	return o.adapterFrom().NativeToValue(value)
}

type structTagInfo struct {
	Name      string
	OmitEmpty bool
	Skip      bool
	HasTag    bool
}

// parseStructTag parses a struct field's tag for the given tagName (e.g. "cel" or "json").
//
// If the tag name or leading comma-separated segment is "-" (such as `json:"-"` or `json:"-,"`),
// the field is marked as skipped. To specify a literal "-" as the field name, single quotes
// can be used, such as `json:"'-'"`.
func parseStructTag(field reflect.StructField, tagName, defaultName string) structTagInfo {
	tag, found := field.Tag.Lookup(tagName)
	if !found {
		return structTagInfo{Name: defaultName}
	}
	parts := strings.Split(tag, ",")
	if parts[0] == "-" {
		return structTagInfo{Skip: true, HasTag: true}
	}
	name := strings.Trim(parts[0], "'")
	if name == "" {
		name = defaultName
	}
	omitEmpty := false
	for _, opt := range parts[1:] {
		if opt == "omitempty" {
			omitEmpty = true
		}
	}
	return structTagInfo{
		Name:      name,
		OmitEmpty: omitEmpty,
		HasTag:    true,
	}
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
	return Bool(refField.IsValid() && !refField.IsZero())
}

func (o *nativeObj) Get(field ref.Val) ref.Val {
	refField, refErr := o.getReflectedField(field)
	if refErr != nil {
		return refErr
	}
	return adaptFieldValue(o.adapterFrom(), refField)
}

func (o *nativeObj) adapterFrom() Adapter {
	if o.adapter != nil {
		return o.adapter
	}
	return o.valType.adapterFrom(nil)
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

// AggregateSize implements the AggregateSizeVisitor interface method.
func (o *nativeObj) AggregateSize(sizer AggregateSizer) uint32 {
	if sz := atomic.LoadUint32(&o.aggSize); sz != 0 {
		return sz
	}
	refVal := reflect.Indirect(o.refValue)
	if !refVal.IsValid() {
		return 1
	}
	total := uint32(1)
	for _, fieldType := range o.valType.fieldsByName {
		fieldValue := safeGetFieldByIndex(refVal, fieldType.Index)
		if !fieldValue.IsValid() || fieldValue.IsZero() {
			continue
		}
		total = safeAddUint32(total, sizer.AggregateSize(fieldValue))
	}
	if cacheableAggregateSize(sizer) {
		atomic.StoreUint32(&o.aggSize, total)
	}
	return total
}

func getNativeTypeName(refType reflect.Type, customName string) string {
	if customName != "" {
		return customName
	}
	return fmt.Sprintf("%s.%s", simplePkgAlias(refType.PkgPath()), refType.Name())
}

func newNativeTypes(rawType reflect.Type, options NativeTypeOptions) ([]*NativeType, error) {
	nt, err := newNativeType(rawType, options)
	if err != nil {
		return nil, err
	}
	result := []*NativeType{nt}

	unwrappedRaw := rawType
	if unwrappedRaw.Kind() == reflect.Pointer {
		unwrappedRaw = unwrappedRaw.Elem()
	}
	alreadySeen := map[string]struct{}{
		unwrappedRaw.String(): {},
	}
	var iterateStructMembers func(reflect.Type)
	iterateStructMembers = func(t reflect.Type) {
		if t.Implements(reflect.TypeFor[ref.Val]()) {
			return
		}
		k := t.Kind()
		if k == reflect.Pointer || k == reflect.Slice || k == reflect.Array {
			iterateStructMembers(t.Elem())
			return
		}
		if k == reflect.Map {
			iterateStructMembers(t.Key())
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
		// Nested member structs use fieldNameHandler only (custom typeName applies to root rawType)
		nestedOpts := NativeTypeOptions{
			fieldNameHandler: options.fieldNameHandler,
			adapter:          options.adapter,
		}
		nt, ntErr := newNativeType(t, nestedOpts)
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
	for _, field := range reflect.VisibleFields(unwrappedRaw) {
		if !field.IsExported() || !isSupportedType(field.Type) {
			continue
		}
		iterateStructMembers(field.Type)
	}

	return result, err
}

func toFieldName(f reflect.StructField, fieldNameHandler NativeTypesFieldNameHandler) string {
	if fieldNameHandler == nil {
		return f.Name
	}
	return fieldNameHandler(f)
}

func newNativeType(rawType reflect.Type, options NativeTypeOptions) (*NativeType, error) {
	refType := rawType
	if refType.Kind() == reflect.Pointer {
		refType = refType.Elem()
	}
	if !isValidObjectType(refType) {
		return nil, fmt.Errorf("unsupported reflect.Type %v, must be reflect.Struct", rawType)
	}
	fieldsByName, err := celFieldsByName(refType, options.fieldNameHandler)
	if err != nil {
		return nil, err
	}
	return &NativeType{
		typeName:     getNativeTypeName(refType, options.typeName),
		refType:      refType,
		fieldsByName: fieldsByName,
		jsonFields:   jsonFieldsOf(refType, options.fieldNameHandler),
		adapter:      options.adapter,
	}, nil
}

// celFieldsByName determines the fields of a struct which are visible to CEL authors, keyed by
// their CEL field name.
//
// A field is visible unless it is skipped according to the configured field name handler, which
// consults the configured struct tag (e.g. `cel:"-"`).
func celFieldsByName(refType reflect.Type, handler NativeTypesFieldNameHandler) (map[string]reflect.StructField, error) {
	skipped := func(f reflect.StructField) bool {
		return isSkippedFieldName(toFieldName(f, handler))
	}
	fieldsByName := make(map[string]reflect.StructField)
	for _, field := range visibleFields(refType, skipped) {
		fieldName := toFieldName(field, handler)
		if _, found := fieldsByName[fieldName]; found {
			return nil, fmt.Errorf("invalid field name `%s` in struct `%s`: %w", fieldName, refType.Name(), errDuplicatedFieldName)
		}
		fieldsByName[fieldName] = field
	}
	return fieldsByName, nil
}

// jsonFieldsOf determines the fields of a struct which are included when serializing to JSON
// (and by extension YAML), along with their serialized names.
//
// A field is serialized unless it is skipped according to the `json` struct tag (`json:"-"`).
// Embedded structs follow encoding/json conventions: those without a json name are serialized
// inline via their promoted fields, those with a json name are serialized as a nested object.
// The serialized name is the json tag name when present, otherwise the CEL field name.
func jsonFieldsOf(refType reflect.Type, handler NativeTypesFieldNameHandler) []nativeJSONField {
	skipped := func(f reflect.StructField) bool {
		return parseStructTag(f, "json", "").Skip
	}
	var jsonFields []nativeJSONField
	for _, field := range visibleFields(refType, skipped) {
		if field.Anonymous && !hasJSONName(field) {
			continue
		}
		if isPromotedThroughNamedJSONField(refType, field) {
			continue
		}
		defaultName := toFieldName(field, handler)
		if isSkippedFieldName(defaultName) {
			defaultName = field.Name
		}
		tagInfo := parseStructTag(field, "json", defaultName)
		jsonFields = append(jsonFields, nativeJSONField{
			index:     field.Index,
			jsonName:  tagInfo.Name,
			omitEmpty: tagInfo.OmitEmpty,
			hasTag:    tagInfo.HasTag,
		})
	}
	return jsonFields
}

// visibleFields returns the exported, supported fields of a struct which are not skipped.
//
// A field is skipped when the predicate holds for the field itself or for any embedded struct
// through which the field is promoted, so that a skipped embedded struct cannot be reached via
// its members.
func visibleFields(refType reflect.Type, skipped func(reflect.StructField) bool) []reflect.StructField {
	var fields []reflect.StructField
	for _, field := range reflect.VisibleFields(refType) {
		if !field.IsExported() || !isSupportedType(field.Type) {
			continue
		}
		if isSkippedField(refType, field, skipped) {
			continue
		}
		fields = append(fields, field)
	}
	return fields
}

// isSkippedField reports whether the field, or any embedded struct through which it is
// promoted, satisfies the skip predicate.
func isSkippedField(refType reflect.Type, field reflect.StructField, skipped func(reflect.StructField) bool) bool {
	if skipped(field) {
		return true
	}
	for _, parent := range embeddedAncestors(refType, field) {
		if skipped(parent) {
			return true
		}
	}
	return false
}

// embeddedAncestors returns the chain of embedded struct fields through which a promoted
// field is reached, from outermost to innermost. The result is empty for direct fields.
func embeddedAncestors(refType reflect.Type, field reflect.StructField) []reflect.StructField {
	if len(field.Index) <= 1 {
		return nil
	}
	ancestors := make([]reflect.StructField, 0, len(field.Index)-1)
	cur := refType
	for _, idx := range field.Index[:len(field.Index)-1] {
		if cur.Kind() == reflect.Pointer {
			cur = cur.Elem()
		}
		if cur.Kind() != reflect.Struct || idx >= cur.NumField() {
			return ancestors
		}
		parent := cur.Field(idx)
		ancestors = append(ancestors, parent)
		cur = parent.Type
	}
	return ancestors
}

// hasJSONName reports whether the field carries a json tag with an explicit name.
func hasJSONName(field reflect.StructField) bool {
	tagInfo := parseStructTag(field, "json", "")
	return tagInfo.HasTag && tagInfo.Name != ""
}

// isPromotedThroughNamedJSONField reports whether a promoted field is reached through an
// embedded struct with a json name, in which case the field is serialized within that
// struct's nested object rather than inline.
func isPromotedThroughNamedJSONField(refType reflect.Type, field reflect.StructField) bool {
	for _, parent := range embeddedAncestors(refType, field) {
		if hasJSONName(parent) {
			return true
		}
	}
	return false
}

func adaptFieldValue(adapter Adapter, refField reflect.Value) ref.Val {
	return adapter.NativeToValue(getFieldValue(adapter, refField))
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
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return reflect.Value{}
		}
		v = v.Elem()
	}
	if len(index) == 1 {
		if v.Kind() != reflect.Struct || index[0] >= v.NumField() {
			return reflect.Value{}
		}
		return v.Field(index[0])
	}
	for _, i := range index {
		if v.Kind() == reflect.Pointer {
			if v.IsNil() {
				v = reflect.Zero(v.Type().Elem())
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

func structFieldOffset(rootType reflect.Type, index []int) (uintptr, bool) {
	var offset uintptr
	cur := rootType
	for i, idx := range index {
		if cur.Kind() != reflect.Struct || idx >= cur.NumField() {
			return 0, false
		}
		sf := cur.Field(idx)
		offset += sf.Offset
		if i < len(index)-1 {
			cur = sf.Type
		}
	}
	return offset, true
}

func (t *NativeType) expectedTypePtrs() (ptrTyp, structTyp unsafe.Pointer) {
	ptrTyp = runtimeTypePtr(reflect.PointerTo(t.refType))
	if !isDirectIface(t.refType) {
		structTyp = runtimeTypePtr(t.refType)
	}
	return ptrTyp, structTyp
}

func (t *NativeType) structPtrFrom(obj any, expectedPtrTyp, expectedStructTyp unsafe.Pointer) (unsafe.Pointer, bool) {
	e := unpackAny(obj)
	if e.typ == expectedPtrTyp {
		return e.ptr, e.ptr != nil
	}
	if expectedStructTyp != nil && e.typ == expectedStructTyp {
		return e.ptr, e.ptr != nil
	}
	if no, ok := obj.(*nativeObj); ok && no.valType == t {
		return no.structPtr, no.structPtr != nil
	}
	return nil, false
}

func unwrapStruct(obj any) reflect.Value {
	if no, ok := obj.(*nativeObj); ok {
		return no.refValue
	}
	if rv, ok := obj.(ref.Val); ok {
		obj = rv.Value()
	}
	return reflect.ValueOf(obj)
}

func fieldTesterFor[T comparable](t *NativeType, refField reflect.StructField, offset uintptr, expectedPtrTyp, expectedStructTyp unsafe.Pointer) func(obj any) bool {
	var zero T
	return func(obj any) bool {
		if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
			return *(*T)(unsafe.Add(ptr, offset)) != zero
		}
		refVal := unwrapStruct(obj)
		refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
		return refFieldVal.IsValid() && !refFieldVal.IsZero()
	}
}

func intFieldGetter[T ~int | ~int32 | ~int64](t *NativeType, refField reflect.StructField, offset uintptr, expectedPtrTyp, expectedStructTyp unsafe.Pointer) func(obj any) (any, error) {
	return func(obj any) (any, error) {
		if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
			return Int(*(*T)(unsafe.Add(ptr, offset))), nil
		}
		refVal := unwrapStruct(obj)
		refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
		if !refFieldVal.IsValid() {
			return nil, nil
		}
		return Int(refFieldVal.Int()), nil
	}
}

func uintFieldGetter[T ~uint | ~uint32 | ~uint64](t *NativeType, refField reflect.StructField, offset uintptr, expectedPtrTyp, expectedStructTyp unsafe.Pointer) func(obj any) (any, error) {
	return func(obj any) (any, error) {
		if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
			return Uint(*(*T)(unsafe.Add(ptr, offset))), nil
		}
		refVal := unwrapStruct(obj)
		refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
		if !refFieldVal.IsValid() {
			return nil, nil
		}
		return Uint(refFieldVal.Uint()), nil
	}
}

func floatFieldGetter[T ~float32 | ~float64](t *NativeType, refField reflect.StructField, offset uintptr, expectedPtrTyp, expectedStructTyp unsafe.Pointer) func(obj any) (any, error) {
	return func(obj any) (any, error) {
		if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
			return Double(*(*T)(unsafe.Add(ptr, offset))), nil
		}
		refVal := unwrapStruct(obj)
		refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
		if !refFieldVal.IsValid() {
			return nil, nil
		}
		return Double(refFieldVal.Float()), nil
	}
}

func (t *NativeType) adapterFrom(obj any) Adapter {
	if no, ok := obj.(*nativeObj); ok && no.adapter != nil {
		return no.adapter
	}
	if t != nil && t.adapter != nil {
		return t.adapter
	}
	return DefaultTypeAdapter
}

func sliceFieldGetter[T any](t *NativeType, refField reflect.StructField, offset uintptr, expectedPtrTyp, expectedStructTyp unsafe.Pointer) func(obj any) (any, error) {
	return func(obj any) (any, error) {
		adapter := t.adapterFrom(obj)
		if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
			return NewList(adapter, *(*[]T)(unsafe.Add(ptr, offset))), nil
		}
		refVal := unwrapStruct(obj)
		refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
		if !refFieldVal.IsValid() {
			return nil, nil
		}
		return NewList(adapter, refFieldVal.Interface().([]T)), nil
	}
}

func mapFieldGetter[K comparable, V any](t *NativeType, refField reflect.StructField, offset uintptr, expectedPtrTyp, expectedStructTyp unsafe.Pointer) func(obj any) (any, error) {
	return func(obj any) (any, error) {
		adapter := t.adapterFrom(obj)
		if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
			return NewMap(adapter, *(*map[K]V)(unsafe.Add(ptr, offset))), nil
		}
		refVal := unwrapStruct(obj)
		refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
		if !refFieldVal.IsValid() {
			return nil, nil
		}
		return NewMap(adapter, refFieldVal.Interface().(map[K]V)), nil
	}
}

func (t *NativeType) makeFieldTester(refField reflect.StructField) func(obj any) bool {
	if offset, ok := structFieldOffset(t.refType, refField.Index); ok {
		expectedPtrTyp, expectedStructTyp := t.expectedTypePtrs()
		switch refField.Type.Kind() {
		case reflect.Bool:
			return fieldTesterFor[bool](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.Int:
			return fieldTesterFor[int](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.Int32:
			return fieldTesterFor[int32](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.Int64:
			return fieldTesterFor[int64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.Uint:
			return fieldTesterFor[uint](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.Uint32, reflect.Float32:
			return fieldTesterFor[uint32](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.Uint64, reflect.Float64:
			return fieldTesterFor[uint64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.String:
			return fieldTesterFor[string](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func, reflect.UnsafePointer, reflect.Interface:
			return fieldTesterFor[unsafe.Pointer](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.Struct:
			if refField.Type == timestampType {
				return func(obj any) bool {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						return !(*time.Time)(unsafe.Add(ptr, offset)).IsZero()
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					return refFieldVal.IsValid() && !refFieldVal.IsZero()
				}
			}
			return func(obj any) bool {
				if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
					return !reflect.NewAt(refField.Type, unsafe.Add(ptr, offset)).Elem().IsZero()
				}
				refVal := unwrapStruct(obj)
				refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
				return refFieldVal.IsValid() && !refFieldVal.IsZero()
			}
		}
	}
	return func(obj any) bool {
		refVal := unwrapStruct(obj)
		refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
		return refFieldVal.IsValid() && !refFieldVal.IsZero()
	}
}

func (t *NativeType) makeFieldGetter(refField reflect.StructField) func(obj any) (any, error) {
	if offset, ok := structFieldOffset(t.refType, refField.Index); ok {
		expectedPtrTyp, expectedStructTyp := t.expectedTypePtrs()
		switch refField.Type.Kind() {
		case reflect.Bool:
			return func(obj any) (any, error) {
				if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
					if *(*bool)(unsafe.Add(ptr, offset)) {
						return True, nil
					}
					return False, nil
				}
				refVal := unwrapStruct(obj)
				refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
				if !refFieldVal.IsValid() {
					return nil, nil
				}
				if refFieldVal.Bool() {
					return True, nil
				}
				return False, nil
			}
		case reflect.Int, reflect.Int32, reflect.Int64:
			if refField.Type == durationType {
				return func(obj any) (any, error) {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						return Duration{Duration: *(*time.Duration)(unsafe.Add(ptr, offset))}, nil
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					if !refFieldVal.IsValid() {
						return nil, nil
					}
					return Duration{Duration: time.Duration(refFieldVal.Int())}, nil
				}
			}
			switch refField.Type.Kind() {
			case reflect.Int:
				return intFieldGetter[int](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.Int32:
				return intFieldGetter[int32](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.Int64:
				return intFieldGetter[int64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			}
		case reflect.Uint, reflect.Uint32, reflect.Uint64:
			switch refField.Type.Kind() {
			case reflect.Uint:
				return uintFieldGetter[uint](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.Uint32:
				return uintFieldGetter[uint32](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.Uint64:
				return uintFieldGetter[uint64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			}
		case reflect.Float32:
			return floatFieldGetter[float32](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.Float64:
			return floatFieldGetter[float64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
		case reflect.String:
			return func(obj any) (any, error) {
				if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
					return String(*(*string)(unsafe.Add(ptr, offset))), nil
				}
				refVal := unwrapStruct(obj)
				refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
				if !refFieldVal.IsValid() {
					return nil, nil
				}
				return String(refFieldVal.String()), nil
			}
		case reflect.Slice:
			if refField.Type.Elem() == reflect.TypeOf(byte(0)) {
				return func(obj any) (any, error) {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						return Bytes(*(*[]byte)(unsafe.Add(ptr, offset))), nil
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					if !refFieldVal.IsValid() {
						return nil, nil
					}
					return Bytes(refFieldVal.Bytes()), nil
				}
			}
			switch refField.Type {
			case reflect.TypeFor[[]string]():
				return sliceFieldGetter[string](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]ref.Val]():
				return sliceFieldGetter[ref.Val](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]int]():
				return sliceFieldGetter[int](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]int64]():
				return sliceFieldGetter[int64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]float64]():
				return sliceFieldGetter[float64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]bool]():
				return sliceFieldGetter[bool](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]int32]():
				return sliceFieldGetter[int32](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]uint]():
				return sliceFieldGetter[uint](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]uint64]():
				return sliceFieldGetter[uint64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]uint32]():
				return sliceFieldGetter[uint32](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]float32]():
				return sliceFieldGetter[float32](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]time.Time]():
				return sliceFieldGetter[time.Time](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]time.Duration]():
				return sliceFieldGetter[time.Duration](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[[]any]():
				return sliceFieldGetter[any](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			default:
				if meta := getDynamicSliceMeta(refField.Type); meta.supported {
					elemType := meta.sliceType.Elem()
					if meta.isPtrElem {
						return func(obj any) (any, error) {
							adapter := t.adapterFrom(obj)
							if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
								elems := *(*[]unsafe.Pointer)(unsafe.Add(ptr, offset))
								return &sliceList[unsafe.Pointer]{
									Adapter:       adapter,
									elems:         elems,
									elemTypePtr:   meta.elemTypePtr,
									meta:          meta,
									qualifyRawVal: canQualifyRawVal(adapter, elemType, meta.qualifyRawVal),
									isNilSlice:    elems == nil,
								}, nil
							}
							refVal := unwrapStruct(obj)
							refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
							if !refFieldVal.IsValid() {
								return nil, nil
							}
							return NewDynamicList(adapter, refFieldVal.Interface()), nil
						}
					}
					return func(obj any) (any, error) {
						adapter := t.adapterFrom(obj)
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							hdr := *(*unsafeSlice)(unsafe.Add(ptr, offset))
							var bytes []byte
							if hdr.Data != nil && hdr.Len > 0 {
								bytes = unsafe.Slice((*byte)(hdr.Data), uintptr(hdr.Len)*meta.elemStride)
							}
							return &sliceList[byte]{
								Adapter:       adapter,
								elems:         bytes,
								elemTypePtr:   meta.elemTypePtr,
								meta:          meta,
								qualifyRawVal: canQualifyRawVal(adapter, elemType, meta.qualifyRawVal),
								isNilSlice:    hdr.Data == nil,
							}, nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						if !refFieldVal.IsValid() {
							return nil, nil
						}
						return NewDynamicList(adapter, refFieldVal.Interface()), nil
					}
				}
			}
		case reflect.Map:
			switch refField.Type {
			case reflect.TypeFor[map[string]string]():
				return mapFieldGetter[string, string](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[map[string]any]():
				return mapFieldGetter[string, any](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[map[ref.Val]ref.Val]():
				return mapFieldGetter[ref.Val, ref.Val](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[map[int64]bool]():
				return mapFieldGetter[int64, bool](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[map[string]int64]():
				return mapFieldGetter[string, int64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[map[string]int]():
				return mapFieldGetter[string, int](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[map[string]bool]():
				return mapFieldGetter[string, bool](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			case reflect.TypeFor[map[string]float64]():
				return mapFieldGetter[string, float64](t, refField, offset, expectedPtrTyp, expectedStructTyp)
			}
		case reflect.Struct:
			if refField.Type == timestampType {
				return func(obj any) (any, error) {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						tm := *(*time.Time)(unsafe.Add(ptr, offset))
						if tm.IsZero() {
							return Timestamp{Time: time.Unix(0, 0)}, nil
						}
						return Timestamp{Time: tm}, nil
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
				}
			}
			if !refField.Type.Implements(refValType) && !refField.Type.Implements(pbMsgInterfaceType) {
				expectedFieldPtrTyp := runtimeTypePtr(reflect.PointerTo(refField.Type))
				return func(obj any) (any, error) {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						return packAny(expectedFieldPtrTyp, unsafe.Add(ptr, offset)), nil
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
				}
			}
		case reflect.Pointer:
			elemType := refField.Type.Elem()
			switch elemType.Kind() {
			case reflect.Bool:
				return func(obj any) (any, error) {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
						if elemPtr == nil || !*(*bool)(elemPtr) {
							return False, nil
						}
						return True, nil
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
				}
			case reflect.Int, reflect.Int32, reflect.Int64:
				if elemType == durationType {
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return Duration{Duration: 0}, nil
							}
							return Duration{Duration: *(*time.Duration)(elemPtr)}, nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				}
				switch elemType.Kind() {
				case reflect.Int:
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return IntZero, nil
							}
							return Int(*(*int)(elemPtr)), nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				case reflect.Int32:
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return IntZero, nil
							}
							return Int(*(*int32)(elemPtr)), nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				case reflect.Int64:
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return IntZero, nil
							}
							return Int(*(*int64)(elemPtr)), nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				}
			case reflect.Uint, reflect.Uint32, reflect.Uint64:
				switch elemType.Kind() {
				case reflect.Uint:
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return Uint(0), nil
							}
							return Uint(*(*uint)(elemPtr)), nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				case reflect.Uint32:
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return Uint(0), nil
							}
							return Uint(*(*uint32)(elemPtr)), nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				case reflect.Uint64:
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return Uint(0), nil
							}
							return Uint(*(*uint64)(elemPtr)), nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				}
			case reflect.Float32, reflect.Float64:
				if elemType.Kind() == reflect.Float32 {
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return Double(0), nil
							}
							return Double(*(*float32)(elemPtr)), nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				}
				return func(obj any) (any, error) {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
						if elemPtr == nil {
							return Double(0), nil
						}
						return Double(*(*float64)(elemPtr)), nil
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
				}
			case reflect.String:
				return func(obj any) (any, error) {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
						if elemPtr == nil {
							return String(""), nil
						}
						return String(*(*string)(elemPtr)), nil
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
				}
			case reflect.Struct:
				if elemType == timestampType {
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return Timestamp{Time: time.Unix(0, 0)}, nil
							}
							tm := *(*time.Time)(elemPtr)
							if tm.IsZero() {
								return Timestamp{Time: time.Unix(0, 0)}, nil
							}
							return Timestamp{Time: tm}, nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				}
				if !refField.Type.Implements(refValType) {
					expectedFieldPtrTyp := runtimeTypePtr(refField.Type)
					var zeroFieldPtr any
					if isNullZeroPointerType(refField.Type) {
						zeroFieldPtr = NullValue
					} else {
						zeroFieldPtr = reflect.New(elemType).Interface()
					}
					return func(obj any) (any, error) {
						if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
							elemPtr := *(*unsafe.Pointer)(unsafe.Add(ptr, offset))
							if elemPtr == nil {
								return zeroFieldPtr, nil
							}
							return packAny(expectedFieldPtrTyp, elemPtr), nil
						}
						refVal := unwrapStruct(obj)
						refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
						return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
					}
				}
			}
		case reflect.Interface:
			if refField.Type == reflect.TypeFor[any]() {
				return func(obj any) (any, error) {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						val := *(*any)(unsafe.Add(ptr, offset))
						return t.adapterFrom(obj).NativeToValue(val), nil
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
				}
			}
			if refField.Type.Implements(refValType) {
				return func(obj any) (any, error) {
					if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
						val := *(*ref.Val)(unsafe.Add(ptr, offset))
						if val == nil {
							return NullValue, nil
						}
						return val, nil
					}
					refVal := unwrapStruct(obj)
					refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
					return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
				}
			}
			return func(obj any) (any, error) {
				if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
					val := reflect.NewAt(refField.Type, unsafe.Add(ptr, offset)).Elem().Interface()
					return t.adapterFrom(obj).NativeToValue(val), nil
				}
				refVal := unwrapStruct(obj)
				refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
				return getFieldValue(t.adapterFrom(obj), refFieldVal), nil
			}
		}
		return func(obj any) (any, error) {
			adapter := t.adapterFrom(obj)
			if ptr, ok := t.structPtrFrom(obj, expectedPtrTyp, expectedStructTyp); ok {
				return getFieldValue(adapter, reflect.NewAt(refField.Type, unsafe.Add(ptr, offset)).Elem()), nil
			}
			refVal := unwrapStruct(obj)
			refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
			return getFieldValue(adapter, refFieldVal), nil
		}
	}
	return func(obj any) (any, error) {
		adapter := t.adapterFrom(obj)
		refVal := unwrapStruct(obj)
		refFieldVal := safeGetFieldByIndex(refVal, refField.Index)
		return getFieldValue(adapter, refFieldVal), nil
	}
}

func isNullZeroPointerType(t reflect.Type) bool {
	switch t {
	case boolWrapperType, byteWrapperType, doubleWrapperType, floatWrapperType,
		int32WrapperType, int64WrapperType, stringWrapperType, uint32WrapperType,
		uint64WrapperType, jsonValueType, anyValueType:
		return true
	default:
		return false
	}
}

func getFieldValue(adapter Adapter, refField reflect.Value) any {
	if adapter == nil {
		adapter = DefaultTypeAdapter
	}
	if !refField.IsValid() {
		return nil
	}
	switch refField.Kind() {
	case reflect.Bool:
		if refField.Bool() {
			return True
		}
		return False
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if refField.Type() == durationType {
			return Duration{Duration: time.Duration(refField.Int())}
		}
		return Int(refField.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return Uint(refField.Uint())
	case reflect.Float32, reflect.Float64:
		return Double(refField.Float())
	case reflect.String:
		return String(refField.String())
	case reflect.Slice:
		if refField.Type().Elem() == reflect.TypeOf(byte(0)) {
			return Bytes(refField.Bytes())
		}
		if refField.Type() == reflect.TypeFor[[]string]() {
			return NewStringList(adapter, refField.Interface().([]string))
		}
		if refField.Type() == reflect.TypeFor[[]ref.Val]() {
			return NewRefValList(adapter, refField.Interface().([]ref.Val))
		}
	case reflect.Map:
		if refField.Type() == reflect.TypeFor[map[string]string]() {
			return NewStringStringMap(adapter, refField.Interface().(map[string]string))
		}
		if refField.Type() == reflect.TypeFor[map[string]any]() {
			return NewStringInterfaceMap(adapter, refField.Interface().(map[string]any))
		}
		if refField.Type() == reflect.TypeFor[map[ref.Val]ref.Val]() {
			return NewRefValMap(adapter, refField.Interface().(map[ref.Val]ref.Val))
		}
	case reflect.Struct:
		if refField.Type() == timestampType {
			if refField.IsZero() {
				return Timestamp{Time: time.Unix(0, 0)}
			}
			return Timestamp{Time: refField.Interface().(time.Time)}
		}
	case reflect.Pointer:
		if refField.IsZero() {
			if isNullZeroPointerType(refField.Type()) {
				return NullValue
			}
			elemType := refField.Type().Elem()
			switch elemType.Kind() {
			case reflect.Bool:
				return False
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				if elemType == durationType {
					return Duration{Duration: 0}
				}
				return IntZero
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				return Uint(0)
			case reflect.Float32, reflect.Float64:
				return Double(0)
			case reflect.String:
				return String("")
			case reflect.Struct:
				if elemType == timestampType {
					return Timestamp{Time: time.Unix(0, 0)}
				}
			}
			return reflect.New(elemType).Interface()
		}
		elemType := refField.Type().Elem()
		switch elemType.Kind() {
		case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
			reflect.Float32, reflect.Float64, reflect.String:
			return getFieldValue(adapter, refField.Elem())
		case reflect.Struct:
			if elemType == timestampType {
				return getFieldValue(adapter, refField.Elem())
			}
		}
	case reflect.Interface:
		if refField.IsNil() {
			return NullValue
		}
		return adapter.NativeToValue(refField.Interface())
	}
	return refField.Interface()
}

func simplePkgAlias(pkgPath string) string {
	paths := strings.Split(pkgPath, "/")
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
		if refElem.Kind() == reflect.Uint8 {
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
		return NewObjectType(getNativeTypeName(refType, "")), true
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
	case reflect.Interface:
		return DynType, true
	}
	return nil, false
}
