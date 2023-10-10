package apivalidation

import (
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func NewValidationSchema(sch common.Schema) *ValidationSchema {
	return &ValidationSchema{
		Schema:  sch,
		visited: map[*spec.Schema]*ValidationSchema{},
	}
}

// ValidationSchema wraps a common.Schema to precompute & cache the results of
// its subschema accessors and associated expensive-to-compute data associated
// with a node in the schema such as CEL expressions.
// !TODO: Lazy[T] everything
type ValidationSchema struct {
	common.Schema

	celRules []CompilationResult
	celError error

	properties           map[string]common.Schema
	additionalProperties common.SchemaOrBool
	items                common.Schema

	allOf []common.Schema
	anyOf []common.Schema
	oneOf []common.Schema
	not   common.Schema

	// OpenAPI schemas have the possibility of being recursive via shared
	// references. We account for this by storing a pointer to the previously
	// result into the map. That way we don't cache the same CEL expressions
	// multiple times.
	visited map[*spec.Schema]*ValidationSchema
}

func (v *ValidationSchema) Properties() map[string]common.Schema {
	if v.properties != nil {
		return v.properties
	}

	props := v.Schema.Properties()
	newProps := make(map[string]common.Schema, len(props))
	for k, value := range props {
		newProps[k] = v.newValidationSchemaNode(value)
	}
	v.properties = newProps
	return newProps
}

// AdditionalProperties returns the OpenAPI additional properties field,
// or nil if this field does not exist.
func (v *ValidationSchema) AdditionalProperties() common.SchemaOrBool {
	if v.additionalProperties != nil {
		return v.additionalProperties
	}

	var result common.SchemaOrBool
	if additionalProperties := v.Schema.AdditionalProperties(); additionalProperties != nil {
		if additionalPropertiesSchema := additionalProperties.Schema(); additionalPropertiesSchema != nil {
			result = &SchemaOrBool{Value: v.newValidationSchemaNode(additionalPropertiesSchema)}
		} else if additionalProperties.Allows() {
			result = &SchemaOrBool{v.newValidationSchemaNode(WildcardSchema)}
		}
	}
	v.additionalProperties = result
	return result
}

func (v *ValidationSchema) Items() common.Schema {
	if v.items != nil {
		return v.items
	}

	nested := v.Schema.Items()
	if nested == nil {
		return nil
	}

	result := v.newValidationSchemaNode(nested)
	v.items = result
	return result
}

func (v *ValidationSchema) AllOf() []common.Schema {
	if v.allOf != nil {
		return v.allOf
	}
	nested := v.Schema.AllOf()
	if nested == nil {
		return nil
	}

	result := make([]common.Schema, 0, len(nested))
	for _, value := range nested {
		result = append(result, v.newValidationSchemaNode(value))
	}
	v.allOf = result
	return result
}

func (v *ValidationSchema) OneOf() []common.Schema {
	if v.oneOf != nil {
		return v.oneOf
	}

	nested := v.Schema.OneOf()
	if nested == nil {
		return nil
	}

	result := make([]common.Schema, 0, len(nested))
	for _, value := range nested {
		result = append(result, v.newValidationSchemaNode(value))
	}
	v.oneOf = result
	return result
}

func (v *ValidationSchema) AnyOf() []common.Schema {
	if v.anyOf != nil {
		return v.anyOf
	}

	nested := v.Schema.AnyOf()
	if nested == nil {
		return nil
	}

	result := make([]common.Schema, 0, len(nested))
	for _, value := range nested {
		result = append(result, v.newValidationSchemaNode(value))
	}
	v.anyOf = result
	return result
}
func (v *ValidationSchema) Not() common.Schema {
	if v.not != nil {
		return v.not
	}

	nested := v.Schema.Not()
	if nested == nil {
		return nil
	}
	result := v.newValidationSchemaNode(nested)
	v.not = result
	return result
}

func (v *ValidationSchema) CELRules() ([]CompilationResult, error) {
	if v.celRules != nil {
		return v.celRules, v.celError
	}
	// Compile cel rules
	compiledRules, compilationError := CompileSchema(v.Schema)
	v.celRules = compiledRules
	v.celError = compilationError
	return compiledRules, compilationError
}

func (v *ValidationSchema) newValidationSchemaNode(sch common.Schema) *ValidationSchema {
	specSchema, isKubeOpenAPI := sch.(*openapi.Schema)
	if isKubeOpenAPI {
		if previous, hasVisited := v.visited[specSchema.Schema]; hasVisited {
			return previous
		}
	}

	// Build schema validator
	res := &ValidationSchema{
		Schema:  sch,
		visited: v.visited,
	}

	if isKubeOpenAPI {
		v.visited[specSchema.Schema] = res
	}

	return res
}

type SchemaOrBool struct {
	Value common.Schema
}

func (sb *SchemaOrBool) Schema() common.Schema {
	return sb.Value
}

func (sb *SchemaOrBool) Allows() bool {
	return sb.Value != nil
}
