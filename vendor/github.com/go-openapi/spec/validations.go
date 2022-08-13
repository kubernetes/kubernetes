package spec

// CommonValidations describe common JSON-schema validations
type CommonValidations struct {
	Maximum          *float64      `json:"maximum,omitempty"`
	ExclusiveMaximum bool          `json:"exclusiveMaximum,omitempty"`
	Minimum          *float64      `json:"minimum,omitempty"`
	ExclusiveMinimum bool          `json:"exclusiveMinimum,omitempty"`
	MaxLength        *int64        `json:"maxLength,omitempty"`
	MinLength        *int64        `json:"minLength,omitempty"`
	Pattern          string        `json:"pattern,omitempty"`
	MaxItems         *int64        `json:"maxItems,omitempty"`
	MinItems         *int64        `json:"minItems,omitempty"`
	UniqueItems      bool          `json:"uniqueItems,omitempty"`
	MultipleOf       *float64      `json:"multipleOf,omitempty"`
	Enum             []interface{} `json:"enum,omitempty"`
}

// SetValidations defines all validations for a simple schema.
//
// NOTE: the input is the larger set of validations available for schemas.
// For simple schemas, MinProperties and MaxProperties are ignored.
func (v *CommonValidations) SetValidations(val SchemaValidations) {
	v.Maximum = val.Maximum
	v.ExclusiveMaximum = val.ExclusiveMaximum
	v.Minimum = val.Minimum
	v.ExclusiveMinimum = val.ExclusiveMinimum
	v.MaxLength = val.MaxLength
	v.MinLength = val.MinLength
	v.Pattern = val.Pattern
	v.MaxItems = val.MaxItems
	v.MinItems = val.MinItems
	v.UniqueItems = val.UniqueItems
	v.MultipleOf = val.MultipleOf
	v.Enum = val.Enum
}

type clearedValidation struct {
	Validation string
	Value      interface{}
}

type clearedValidations []clearedValidation

func (c clearedValidations) apply(cbs []func(string, interface{})) {
	for _, cb := range cbs {
		for _, cleared := range c {
			cb(cleared.Validation, cleared.Value)
		}
	}
}

// ClearNumberValidations clears all number validations.
//
// Some callbacks may be set by the caller to capture changed values.
func (v *CommonValidations) ClearNumberValidations(cbs ...func(string, interface{})) {
	done := make(clearedValidations, 0, 5)
	defer func() {
		done.apply(cbs)
	}()

	if v.Minimum != nil {
		done = append(done, clearedValidation{Validation: "minimum", Value: v.Minimum})
		v.Minimum = nil
	}
	if v.Maximum != nil {
		done = append(done, clearedValidation{Validation: "maximum", Value: v.Maximum})
		v.Maximum = nil
	}
	if v.ExclusiveMaximum {
		done = append(done, clearedValidation{Validation: "exclusiveMaximum", Value: v.ExclusiveMaximum})
		v.ExclusiveMaximum = false
	}
	if v.ExclusiveMinimum {
		done = append(done, clearedValidation{Validation: "exclusiveMinimum", Value: v.ExclusiveMinimum})
		v.ExclusiveMinimum = false
	}
	if v.MultipleOf != nil {
		done = append(done, clearedValidation{Validation: "multipleOf", Value: v.MultipleOf})
		v.MultipleOf = nil
	}
}

// ClearStringValidations clears all string validations.
//
// Some callbacks may be set by the caller to capture changed values.
func (v *CommonValidations) ClearStringValidations(cbs ...func(string, interface{})) {
	done := make(clearedValidations, 0, 3)
	defer func() {
		done.apply(cbs)
	}()

	if v.Pattern != "" {
		done = append(done, clearedValidation{Validation: "pattern", Value: v.Pattern})
		v.Pattern = ""
	}
	if v.MinLength != nil {
		done = append(done, clearedValidation{Validation: "minLength", Value: v.MinLength})
		v.MinLength = nil
	}
	if v.MaxLength != nil {
		done = append(done, clearedValidation{Validation: "maxLength", Value: v.MaxLength})
		v.MaxLength = nil
	}
}

// ClearArrayValidations clears all array validations.
//
// Some callbacks may be set by the caller to capture changed values.
func (v *CommonValidations) ClearArrayValidations(cbs ...func(string, interface{})) {
	done := make(clearedValidations, 0, 3)
	defer func() {
		done.apply(cbs)
	}()

	if v.MaxItems != nil {
		done = append(done, clearedValidation{Validation: "maxItems", Value: v.MaxItems})
		v.MaxItems = nil
	}
	if v.MinItems != nil {
		done = append(done, clearedValidation{Validation: "minItems", Value: v.MinItems})
		v.MinItems = nil
	}
	if v.UniqueItems {
		done = append(done, clearedValidation{Validation: "uniqueItems", Value: v.UniqueItems})
		v.UniqueItems = false
	}
}

// Validations returns a clone of the validations for a simple schema.
//
// NOTE: in the context of simple schema objects, MinProperties, MaxProperties
// and PatternProperties remain unset.
func (v CommonValidations) Validations() SchemaValidations {
	return SchemaValidations{
		CommonValidations: v,
	}
}

// HasNumberValidations indicates if the validations are for numbers or integers
func (v CommonValidations) HasNumberValidations() bool {
	return v.Maximum != nil || v.Minimum != nil || v.MultipleOf != nil
}

// HasStringValidations indicates if the validations are for strings
func (v CommonValidations) HasStringValidations() bool {
	return v.MaxLength != nil || v.MinLength != nil || v.Pattern != ""
}

// HasArrayValidations indicates if the validations are for arrays
func (v CommonValidations) HasArrayValidations() bool {
	return v.MaxItems != nil || v.MinItems != nil || v.UniqueItems
}

// HasEnum indicates if the validation includes some enum constraint
func (v CommonValidations) HasEnum() bool {
	return len(v.Enum) > 0
}

// SchemaValidations describes the validation properties of a schema
//
// NOTE: at this moment, this is not embedded in SchemaProps because this would induce a breaking change
// in the exported members: all initializers using litterals would fail.
type SchemaValidations struct {
	CommonValidations

	PatternProperties SchemaProperties `json:"patternProperties,omitempty"`
	MaxProperties     *int64           `json:"maxProperties,omitempty"`
	MinProperties     *int64           `json:"minProperties,omitempty"`
}

// HasObjectValidations indicates if the validations are for objects
func (v SchemaValidations) HasObjectValidations() bool {
	return v.MaxProperties != nil || v.MinProperties != nil || v.PatternProperties != nil
}

// SetValidations for schema validations
func (v *SchemaValidations) SetValidations(val SchemaValidations) {
	v.CommonValidations.SetValidations(val)
	v.PatternProperties = val.PatternProperties
	v.MaxProperties = val.MaxProperties
	v.MinProperties = val.MinProperties
}

// Validations for a schema
func (v SchemaValidations) Validations() SchemaValidations {
	val := v.CommonValidations.Validations()
	val.PatternProperties = v.PatternProperties
	val.MinProperties = v.MinProperties
	val.MaxProperties = v.MaxProperties
	return val
}

// ClearObjectValidations returns a clone of the validations with all object validations cleared.
//
// Some callbacks may be set by the caller to capture changed values.
func (v *SchemaValidations) ClearObjectValidations(cbs ...func(string, interface{})) {
	done := make(clearedValidations, 0, 3)
	defer func() {
		done.apply(cbs)
	}()

	if v.MaxProperties != nil {
		done = append(done, clearedValidation{Validation: "maxProperties", Value: v.MaxProperties})
		v.MaxProperties = nil
	}
	if v.MinProperties != nil {
		done = append(done, clearedValidation{Validation: "minProperties", Value: v.MinProperties})
		v.MinProperties = nil
	}
	if v.PatternProperties != nil {
		done = append(done, clearedValidation{Validation: "patternProperties", Value: v.PatternProperties})
		v.PatternProperties = nil
	}
}
