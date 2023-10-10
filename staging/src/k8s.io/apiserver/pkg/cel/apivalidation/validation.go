package apivalidation

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/cel"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/kube-openapi/pkg/validation/errors"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
	"k8s.io/kube-openapi/pkg/validation/validate"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

type ValidationOptions struct {
	PreserveUnknownFields bool
	Ratcheting            bool
	Allocator             value.Allocator
	Registry              strfmt.Registry
	CELCostBudget         *int64
}

type ChildKey struct {
	Key   string
	Index int
}

func (c ChildKey) String() string {
	if len(c.Key) > 0 {
		return c.Key
	}
	return strconv.Itoa(c.Index)
}

// ValidationContext represents the result of validating a value. The tree
// follows the structure of the value and stores the correlated old value, if
// possible.
type ValidationContext struct {
	Result
	ValidationOptions

	Value    value.Value
	OldValue value.Value

	// TODO: enum
	Children map[ChildKey]*ValidationContext

	hasParent        bool
	comparisonResult *bool
}

func (v *ValidationSchema) Validate(new interface{}, options ValidationOptions) field.ErrorList {
	value, err := toSMDValue(new)
	if err != nil {
		return field.ErrorList{field.InternalError(nil, err)}
	}

	if options.CELCostBudget == nil {
		budget := int64(cel.RuntimeCELCostBudget)
		options.CELCostBudget = &budget
	}

	if options.Registry == nil {
		options.Registry = strfmt.Default
	}

	result := &ValidationContext{
		Value:             value,
		ValidationOptions: options,
	}
	validateNode(result, v)
	res, _, _ := result.ErrorList()
	return res
}

func (v *ValidationSchema) ValidateUpdate(new, old interface{}, options ValidationOptions) field.ErrorList {
	value, err := toSMDValue(new)
	if err != nil {
		return field.ErrorList{field.InternalError(nil, err)}
	}
	oldValue, err := toSMDValue(old)
	if err != nil {
		return field.ErrorList{field.InternalError(nil, err)}
	}

	if options.CELCostBudget == nil {
		budget := int64(cel.RuntimeCELCostBudget)
		options.CELCostBudget = &budget
	}
	result := &ValidationContext{
		Value:             value,
		OldValue:          oldValue,
		ValidationOptions: options,
	}
	validateNode(result, v)
	res, updateErrors, _ := result.ErrorList()
	return append(res, updateErrors...)
}

func toSMDValue(v interface{}) (value.Value, error) {
	if val, ok := v.(value.Value); ok {
		return val, nil
	}

	vT := reflect.ValueOf(&v)
	for vT.Kind() == reflect.Interface || vT.Kind() == reflect.Pointer {
		vT = vT.Elem()
	}

	if vT.Kind() == reflect.Struct {
		if res, err := value.NewValueReflect(vT.Addr().Interface()); err == nil {
			return res, nil
		} else {
			return nil, err
		}
	}

	return value.NewValueInterface(v), nil
}

// Returns a validator that is capable of recursively validating the
// given quantifier for the provided schemas
func validateNode(result *ValidationContext, sch common.Schema) {
	// Convert to SMD value
	if result.Value.IsNull() {
		validateType(result, sch)
		validateEnum(result, sch)
		return
	}

	// Common Validations
	validateType(result, sch)
	validateEnum(result, sch)
	validateFormat(result, sch)

	// Type-specific validations
	validateString(result, sch)
	validateNumber(result, sch)
	validateSlice(result, sch)
	validateMap(result, sch)
	validateQuantifiers(result, sch)
	validateCEL(result, sch)
}

func validateType(result *ValidationContext, sch common.Schema) {
	typ := OpenAPISchemaType(sch.Type())
	if len(typ) == 0 {
		return
	}

	if expectedType := schemaTypeForValue(result.Value); expectedType != typ {
		if (typ == SchemaTypeNumber && expectedType == SchemaTypeInteger) || (typ == SchemaTypeInteger && expectedType == SchemaTypeNumber) {
			//!TODO: check if provided value is convertible to the expected type
			// without loss of precision
		} else if result.Value.IsNull() && sch.Nullable() {
			// Don't throw an error if the value was null and the schema is nullable
		} else {
			result.AddErrors(errors.InvalidType("", "", string(typ), expectedType))

		}
	}
}

func validateEnum(result *ValidationContext, sch common.Schema) {
	if enums := sch.Enum(); len(enums) > 0 {
		matched := false
		for _, enumValue := range enums {
			smdEnumValue := value.NewValueInterface(enumValue)
			if value.Equals(smdEnumValue, result.Value) {
				matched = true
				break
			}
		}

		if !matched {
			result.AddErrors(errors.EnumFail("", "", result.Value.Unstructured(), enums))
		}
	}
}

func validateFormat(result *ValidationContext, sch common.Schema) {
	switch {
	case result.Value.IsString():
		// Ignore unrecognized formats in schemas.
		// Formats that do not match the type of their schema should be checked
		// during validation of the schema itself. This validator assumes the
		// schema is valid.
		if result.Registry.ContainsName(sch.Format()) {
			if err := validate.FormatOf("", "", sch.Format(), result.Value.AsString(), result.Registry); err != nil {
				result.AddErrors(err)
			}
		}
	case result.Value.IsInt():
		//!TODO: ensure format isn't float
		//!TODO: int32, int64 format range validation
	case result.Value.IsFloat():
		//!TODO: ensure format isn't integer
		//!TODO: float32, float64 format range validation
	}
}

func validateNumber(result *ValidationContext, sch common.Schema) {
	if !result.Value.IsFloat() && !result.Value.IsInt() {
		return
	}

	var asFloat float64 = 0
	if result.Value.IsFloat() {
		asFloat = result.Value.AsFloat()
	} else {
		asFloat = float64(result.Value.AsInt())
	}

	if multiple := sch.MultipleOf(); multiple != nil {
		// Is the constraint specifier within the range of the specific numeric type and format?
		if err := validate.IsValueValidAgainstRange(*multiple, sch.Type(), sch.Format(), "MultipleOf", ""); err == nil {
			result.AddErrors(validate.MultipleOfNativeType("", "", result.Value.Unstructured(), *multiple))
		} else {
			// Constraint nevertheless validated, converted as general number
			result.AddErrors(err, validate.MultipleOf("", "", asFloat, *multiple))
		}
	}

	if maximum := sch.Maximum(); maximum != nil {
		// Is the constraint specifier within the range of the specific numeric type and format?
		if err := validate.IsValueValidAgainstRange(*maximum, sch.Type(), sch.Format(), "Maximum boundary", ""); err == nil {
			// Constraint validated with compatible types
			result.AddErrors(validate.MaximumNativeType("", "", result.Value.Unstructured(), *maximum, sch.IsExclusiveMaximum()))

		} else {
			// Constraint nevertheless validated, converted as general number
			result.AddErrors(err, validate.Maximum("", "", asFloat, *maximum, sch.IsExclusiveMaximum()))
		}
	}

	if minimum := sch.Minimum(); minimum != nil {
		// Is the constraint specifier within the range of the specific numeric type and format?
		if err := validate.IsValueValidAgainstRange(*minimum, sch.Type(), sch.Format(), "Minimum boundary", ""); err == nil {
			// Constraint validated with compatible types
			result.AddErrors(validate.MinimumNativeType("", "", result.Value.Unstructured(), *minimum, sch.IsExclusiveMinimum()))

		} else {
			// Constraint nevertheless validated, converted as general number
			result.AddErrors(err, validate.Minimum("", "", asFloat, *minimum, sch.IsExclusiveMinimum()))
		}
	}
}

func validateString(result *ValidationContext, sch common.Schema) {
	if !result.Value.IsString() {
		return
	}

	data := result.Value.AsString()

	if max := sch.MaxLength(); max != nil {
		result.AddErrors(validate.MaxLength("", "", data, *max))
	}

	if min := sch.MinLength(); min != nil {
		result.AddErrors(validate.MinLength("", "", data, *min))
	}

	if pattern := sch.Pattern(); len(pattern) > 0 {
		result.AddErrors(validate.Pattern("", "", data, pattern))
	}
}

func findDuplicates(fldPath *field.Path, list value.List) ([]int, *field.Error) {
	if list.Length() <= 1 {
		return nil, nil
	}

	seenScalars := make(map[interface{}]int, list.Length())
	seenCompounds := make(map[string]int, list.Length())
	var nonUniqueIndices []int
	for iter := list.Range(); iter.Next(); {
		i, item := iter.Item()
		x := item.Unstructured()
		switch x.(type) {
		case map[string]interface{}, []interface{}:
			bs, err := json.Marshal(x)
			if err != nil {
				return nil, field.Invalid(fldPath.Index(i), x, "internal error")
			}
			s := string(bs)
			if times, seen := seenCompounds[s]; !seen {
				seenCompounds[s] = 1
			} else {
				seenCompounds[s]++
				if times == 1 {
					nonUniqueIndices = append(nonUniqueIndices, i)
				}
			}
		default:
			if times, seen := seenScalars[x]; !seen {
				seenScalars[x] = 1
			} else {
				seenScalars[x]++
				if times == 1 {
					nonUniqueIndices = append(nonUniqueIndices, i)
				}
			}
		}
	}

	return nonUniqueIndices, nil
}

func validateXListTypeSet(result *ValidationContext, list value.List) {
	nonUnique, err := findDuplicates(result.Path, list)
	if err != nil {
		result.AddErrors(err)
	} else {
		for _, i := range nonUnique {
			result.Index(i, nil, nil).AddErrors(field.Duplicate(nil, list.At(i).Unstructured()))
		}
	}
}

// !TODO: This should not be validated on the non-structural portions of the
// schema
func validateXListTypeMapKeys(result *ValidationContext, list value.List, mapKeys []string) {
	// only allow nil and objects
	for iter := list.Range(); iter.Next(); {
		i, item := iter.Item()
		if !item.IsNull() && !item.IsMap() {
			result.Index(i, item, nil).AddErrors(field.Invalid(nil, item.Unstructured(), "must be an object for an array of list-type map"))
			return
		}
	}

	if list.Length() <= 1 {
		return
	}

	// optimize simple case of one key
	if len(mapKeys) == 1 {
		type unspecifiedKeyValue struct{}

		keyField := mapKeys[0]
		keys := make([]interface{}, 0, list.Length())
		for iter := list.Range(); iter.Next(); {
			_, x := iter.Item()
			if x.IsNull() {
				keys = append(keys, unspecifiedKeyValue{}) // nil object means unspecified key
				continue
			}

			xMap := x.AsMap()

			// undefined key?
			key, ok := xMap.Get(keyField)
			if !ok {
				keys = append(keys, unspecifiedKeyValue{})
				continue
			}

			keys = append(keys, key.Unstructured())
		}

		nonUnique, err := findDuplicates(result.Path, value.NewValueInterface(keys).AsList())
		if err != nil {
			result.AddErrors(err)
			return
		}

		for _, i := range nonUnique {
			switch keys[i] {
			case unspecifiedKeyValue{}:
				result.Index(i, nil, nil).AddErrors(field.Duplicate(nil, map[string]interface{}{}))
			default:
				result.Index(i, nil, nil).AddErrors(field.Duplicate(nil, map[string]interface{}{keyField: keys[i]}))
			}
		}
	}

	// multiple key fields
	keys := make([]interface{}, 0, list.Length())
	for iter := list.Range(); iter.Next(); {
		_, item := iter.Item()
		key := map[string]interface{}{}
		if item.IsNull() {
			keys = append(keys, key)
			continue
		}

		asMap := item.AsMap()
		for _, keyField := range mapKeys {
			if k, ok := asMap.Get(keyField); ok {
				key[keyField] = k.Unstructured()
			}
		}

		keys = append(keys, key)
	}

	nonUnique, err := findDuplicates(result.Path, value.NewValueInterface(keys).AsList())
	if err != nil {
		result.AddErrors(err)
		return
	}

	for _, i := range nonUnique {
		result.Index(i, nil, nil).AddErrors(field.Duplicate(nil, keys[i]))
	}
}

func validateSlice(result *ValidationContext, sch common.Schema) {
	if !result.Value.IsList() {
		return
	}

	asList := result.Value.AsList()
	asListLength := int64(asList.Length())

	// Validate Properties on the node before recursing
	if minimum := sch.MinItems(); minimum != nil && asListLength < *minimum {
		result.AddErrors(errors.TooFewItems("", "", *minimum, asListLength))
	}

	if maximum := sch.MaxItems(); maximum != nil && asListLength < *maximum {
		result.AddErrors(errors.TooManyItems("", "", *maximum, asListLength))
	}

	if sch.UniqueItems() || sch.XListType() == "set" {
		validateXListTypeSet(result, asList)
	}

	if sch.XListType() == "map" {
		validateXListTypeMapKeys(result, asList, sch.XListMapKeys())
	}

	var oldList common.MapList
	if result.OldValue != nil && result.OldValue.IsList() {
		oldAsList := result.OldValue.AsList()
		oldItems := make([]interface{}, oldAsList.Length())
		for i := 0; i < oldAsList.Length(); i++ {
			oldItems = append(oldItems, oldAsList.At(i))
		}
		oldList = common.MakeMapList(sch, oldItems)
	}

	if items := sch.Items(); items != nil {
		iter := asList.Range()
		for iter.Next() {
			idx, item := iter.Item()
			var oldItemValue value.Value
			if oldList != nil {
				if oldItemValueValue := oldList.Get(result.Value); oldItemValueValue != nil {
					oldItemValue, _ = oldItemValueValue.(value.Value)
				}
			}
			validateNode(result.Index(idx, item, oldItemValue), items)
		}
	} else if !result.PreserveUnknownFields && sch.Type() == string(SchemaTypeArray) {
		// Array with no items is an error without preserveunknownfields
		//!TODO: This error is not thrown by earlier k8s validators.
		// maybe do away with it?
		// CRDs will never hit this, since they are validated not to
		// native type schemas also wont hit it, since they are generated
		// such that arrays always have items.
		// leave here for sanity?
		result.AddErrors(errors.PropertyNotAllowed("", "", ""))
	} else {
		// No schema for non-array is fine. A type error is thrown elsewhere.
	}
}

func validateMap(result *ValidationContext, sch common.Schema) {
	if !result.Value.IsMap() {
		return
	}

	asMap := result.Value.AsMap()
	asMapLength := int64(asMap.Length())

	if minimum := sch.MinProperties(); minimum != nil && asMapLength < *minimum {
		result.AddErrors(errors.TooFewProperties("", "", *minimum, int64(asMapLength)))
	}

	if maximum := sch.MaxProperties(); maximum != nil && asMapLength > *maximum {
		result.AddErrors(errors.TooManyProperties("", "", *maximum, int64(asMapLength)))
	}

	var oldMap value.Map
	if result.OldValue != nil && result.OldValue.IsMap() {
		oldMap = result.OldValue.AsMap()
	}

	properties := sch.Properties()
	additionalProperties := sch.AdditionalProperties()

	asMap.Iterate(func(key string, fieldValue value.Value) bool {
		var oldValue value.Value
		if oldMap != nil {
			oldValue, _ = oldMap.Get(key)
		}

		var childResult *ValidationContext
		childSchema, hasChildSchema := properties[key]
		if hasChildSchema {
			childResult = result.Child(key, fieldValue, oldValue)
		} else if additionalProperties != nil {
			childResult = result.Key(key, fieldValue, oldValue)

			if additionalPropertiesSchema := additionalProperties.Schema(); additionalPropertiesSchema != nil {
				childSchema = additionalPropertiesSchema
			} else if additionalProperties.Allows() {
				childSchema = WildcardSchema
			}
		} else {
			childResult = result.Key(key, fieldValue, oldValue)
		}

		if childSchema == nil {
			childResult.AddErrors(errors.PropertyNotAllowed("", "", ""))
		} else {
			validateNode(childResult, childSchema)
		}

		return true
	})

	for _, key := range sch.Required() {
		if !asMap.Has(key) {
			var oldValue value.Value
			if oldMap != nil {
				oldValue, _ = oldMap.Get(key)
			}
			result.Child(key, value.NewValueInterface(nil), oldValue).AddErrors(errors.Required("", ""))
		}
	}
}

func validateQuantifiers(result *ValidationContext, sch common.Schema) {
	if allOf := sch.AllOf(); len(allOf) > 0 {

		numSuccess := 0
		for _, s := range sch.AllOf() {
			quantorResult := result.Clone()

			validateNode(quantorResult, s)
			if quantorResult.IsValid() {
				numSuccess += 1
			}
			result.Merge(quantorResult)
		}
		if numSuccess != len(allOf) {
			msg := ""
			if numSuccess == 0 {
				msg = ". None validated"
			}
			result.AddErrors(errors.New(errors.CompositeErrorCode, validate.MustValidateAllSchemasError, "", msg))
		}
	}

	if anyOf := sch.AnyOf(); len(anyOf) > 0 {
		var bestAnyOfFailure *ValidationContext
		bestAnyofDepth := 0
		var firstSuccess *ValidationContext
		for _, s := range sch.AnyOf() {
			quantorResult := result.Clone()
			validateNode(quantorResult, s)

			if quantorResult.IsValid() {
				bestAnyOfFailure = nil
				firstSuccess = quantorResult
				break
			}

			if bestAnyOfFailure == nil {
				bestAnyOfFailure = quantorResult
				bestAnyofDepth = quantorResult.Depth()
			} else if d := quantorResult.Depth(); d > bestAnyofDepth {
				bestAnyOfFailure = quantorResult
				bestAnyofDepth = d
			}
		}

		if firstSuccess == nil {
			result.AddErrors(errors.New(errors.CompositeErrorCode, validate.MustValidateAtLeastOneSchemaError, ""))
			result.Merge(bestAnyOfFailure)
		} else {
			result.Merge(firstSuccess)
		}
	}

	if oneOf := sch.OneOf(); len(oneOf) > 0 {
		numOneOfValid := 0
		var firstSuccess *ValidationContext
		var bestOneOfFailure *ValidationContext
		bestOneOfFailureDepth := 0

		for _, s := range oneOf {
			quantorResult := result.Clone()
			validateNode(quantorResult, s)
			if quantorResult.IsValid() {
				numOneOfValid += 1
				firstSuccess = quantorResult
			} else if bestOneOfFailure == nil {
				bestOneOfFailure = quantorResult
				bestOneOfFailureDepth = quantorResult.Depth()
			} else if d := quantorResult.Depth(); d > bestOneOfFailureDepth {
				bestOneOfFailure = quantorResult
				bestOneOfFailureDepth = d
			}
		}

		if numOneOfValid == 1 {
			result.Merge(firstSuccess)
		} else if numOneOfValid == 0 {
			result.Merge(bestOneOfFailure)
			result.AddErrors(errors.New(errors.CompositeErrorCode, validate.MustValidateOnlyOneSchemaError, "", "Found none valid"))
		} else {
			result.AddErrors(errors.New(errors.CompositeErrorCode, validate.MustValidateOnlyOneSchemaError, "", fmt.Sprintf("Found %d alternatives", numOneOfValid)))
		}
	}

	if not := sch.Not(); not != nil {
		quantorResult := result.Clone()
		validateNode(quantorResult, not)

		// We keep inner IMPORTANT! errors no matter what MatchCount tells us
		if quantorResult.IsValid() {
			result.AddErrors(errors.New(errors.CompositeErrorCode, validate.MustNotValidateSchemaError, ""))
		}
	}
}

func CompileSchema(s common.Schema) ([]CompilationResult, error) {
	// Todo: mative ypes should use new expressions env
	// crds should use storedexpressions env
	//!TODO: resource root argument is wrong
	//!TODO: i dont like that we even inject type meta at all.
	return compile(
		s,
		common.SchemaDeclType(s, s.IsXEmbeddedResource()),
		celconfig.PerCallLimit,
		environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()),
		StoredExpressionsEnvLoader(),
	)
}

func validateCEL(result *ValidationContext, sch common.Schema) {
	if result.CELCostBudget == nil {
		return
	}

	var compiledRules []CompilationResult
	var compilationError error

	if vSch, ok := sch.(*ValidationSchema); ok {
		compiledRules, compilationError = vSch.CELRules()
	} else {
		compiledRules, compilationError = CompileSchema(sch)
	}

	if compilationError != nil {
		result.AddErrors(field.Invalid(result.Path, sch.Type(), fmt.Sprintf("rule compiler initialization error: %v", compilationError)))
		return
	} else if len(compiledRules) == 0 {
		return
	}

	*result.CELCostBudget = validateExpressions(context.TODO(), result, sch, compiledRules, *result.CELCostBudget)
}

func schemaTypeForValue(val value.Value) OpenAPISchemaType {
	switch {
	case val.IsBool():
		return SchemaTypeBool
	case val.IsString():
		return SchemaTypeString
	case val.IsInt():
		return SchemaTypeInteger
	case val.IsFloat():
		return SchemaTypeNumber
	case val.IsMap():
		return SchemaTypeObject
	case val.IsList():
		return SchemaTypeArray
	case val.IsNull():
		return SchemaTypeNull
	default:
		return SchemaTypeObject
	}
}

func (r *ValidationContext) Merge(other *ValidationContext) {
	r.Result.Merge(&other.Result)

	for k, c := range other.Children {
		if existing, exists := r.Children[k]; exists {
			existing.Merge(c)
		} else {
			r.Children[k] = c
		}
	}
}

func (r *ValidationContext) Key(k string, value, oldValue value.Value) *ValidationContext {
	key := ChildKey{Key: k}
	if existing, exists := r.Children[key]; exists {
		return existing
	}
	ret := &ValidationContext{
		ValidationOptions: r.ValidationOptions,
		Value:             value,
		OldValue:          oldValue,
		Result: Result{
			Path: r.Path.Key(k),
		},
	}
	if r.Children == nil {
		r.Children = map[ChildKey]*ValidationContext{}
	}
	r.Children[key] = ret
	return ret
}

func (r *ValidationContext) Index(index int, value, oldValue value.Value) *ValidationContext {
	key := ChildKey{Index: index}
	if existing, exists := r.Children[key]; exists {
		if value != nil && existing.Value == nil {
			existing.Value = value
		}
		if oldValue != nil && existing.OldValue == nil {
			existing.Value = value
		}
		return existing
	}
	ret := &ValidationContext{
		ValidationOptions: r.ValidationOptions,
		Value:             value,
		OldValue:          oldValue,
		Result: Result{
			Path: r.Path.Index(index),
		},
	}
	if r.Children == nil {
		r.Children = map[ChildKey]*ValidationContext{}
	}
	r.Children[key] = ret
	return ret
}

func (r *ValidationContext) Child(k string, value, oldValue value.Value) *ValidationContext {
	key := ChildKey{Key: k}
	if existing, exists := r.Children[key]; exists {
		return existing
	}
	ret := &ValidationContext{
		ValidationOptions: r.ValidationOptions,
		Value:             value,
		OldValue:          oldValue,
		Result: Result{
			Path: r.Path.Child(k),
		},
	}
	if r.Children == nil {
		r.Children = map[ChildKey]*ValidationContext{}
	}
	r.Children[key] = ret
	return ret
}

func (r *ValidationContext) IsRoot() bool {
	//!TODO: would be cool if field.Path had more accessors and we could remove
	// this field
	return r.hasParent
}

func (r *ValidationContext) Clone() *ValidationContext {
	return &ValidationContext{
		Value:             r.Value,
		OldValue:          r.OldValue,
		ValidationOptions: r.ValidationOptions,
		hasParent:         r.hasParent,
		Result: Result{
			Path: r.Path,
		},
	}
}
func (r *ValidationContext) IsValid() bool {
	if !r.Result.IsValid() {
		return false
	}

	for _, c := range r.Children {
		if !c.IsValid() {
			return false
		}
	}
	return true
}

func (r *ValidationContext) Depth() int {
	maxChildDepth := 0
	for _, c := range r.Children {
		if d := c.Depth(); d > maxChildDepth {
			maxChildDepth = d
		}
	}
	return 1 + maxChildDepth
}

// CachedDeepEqual is equivalent to reflect.DeepEqual, but caches the
// results in the tree of ratchetInvocationScratch objects on the way:
//
// For objects and arrays, this function will make a best effort to make
// use of past DeepEqual checks performed by this Node's children, if available.
//
// If a lazy computation could not be found for all children possibly due
// to validation logic short circuiting and skipping the children, then
// this function simply defers to reflect.DeepEqual.
func (r *ValidationContext) CachedDeepEqual() (res bool) {
	if r.comparisonResult != nil {
		return *r.comparisonResult
	}

	defer func() {
		r.comparisonResult = &res
	}()

	if r.Value == nil && r.OldValue == nil {
		return true
	} else if r.Value == nil || r.OldValue == nil {
		return false
	} else if len(r.Children) == 0 {
		return value.Equals(r.Value, r.OldValue)
	}

	// Correctly considers map-type lists due to fact that index here
	// is only used for numbering. The correlation is stored in the
	// childInvocation itself
	//
	// NOTE: This does not consider sets, since we don't correlate them.
	for _, child := range r.Children {
		// Query for child
		if !child.CachedDeepEqual() {
			// If one child is not equal the entire object is not equal
			return false
		}
	}
	return true
}

func (r *ValidationContext) ErrorList() (errors field.ErrorList, updateErrors field.ErrorList, warnings field.ErrorList) {
	errors = append(errors, r.Errors...)
	errors = append(errors, r.UpdateErrors...)
	warnings = append(warnings, r.Warnings...)
	for _, c := range r.Children {
		childErrors, childUpdateErrors, childWarnings := c.ErrorList()
		errors = append(errors, childErrors...)
		updateErrors = append(updateErrors, childUpdateErrors...)
		warnings = append(warnings, childWarnings...)
	}

	if r.OldValue != nil && len(errors) > 0 && r.Ratcheting && r.CachedDeepEqual() {
		warnings = append(warnings, errors...)
		errors = nil
	}

	return errors, updateErrors, warnings
}
