/*
Copyright 2017 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"reflect"
	"strings"

	genericvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	validationutil "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiservervalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
)

var (
	printerColumnDatatypes                = sets.NewString("integer", "number", "string", "boolean", "date")
	customResourceColumnDefinitionFormats = sets.NewString("int32", "int64", "float", "double", "byte", "date", "date-time", "password")
)

// ValidateCustomResourceDefinition statically validates
func ValidateCustomResourceDefinition(obj *apiextensions.CustomResourceDefinition) field.ErrorList {
	nameValidationFn := func(name string, prefix bool) []string {
		ret := genericvalidation.NameIsDNSSubdomain(name, prefix)
		requiredName := obj.Spec.Names.Plural + "." + obj.Spec.Group
		if name != requiredName {
			ret = append(ret, fmt.Sprintf(`must be spec.names.plural+"."+spec.group`))
		}
		return ret
	}

	allErrs := genericvalidation.ValidateObjectMeta(&obj.ObjectMeta, false, nameValidationFn, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateCustomResourceDefinitionSpec(&obj.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateCustomResourceDefinitionStatus(&obj.Status, field.NewPath("status"))...)
	allErrs = append(allErrs, ValidateCustomResourceDefinitionStoredVersions(obj.Status.StoredVersions, obj.Spec.Versions, field.NewPath("status").Child("storedVersions"))...)
	return allErrs
}

// ValidateCustomResourceDefinitionUpdate statically validates
func ValidateCustomResourceDefinitionUpdate(obj, oldObj *apiextensions.CustomResourceDefinition) field.ErrorList {
	allErrs := genericvalidation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateCustomResourceDefinitionSpecUpdate(&obj.Spec, &oldObj.Spec, apiextensions.IsCRDConditionTrue(oldObj, apiextensions.Established), field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateCustomResourceDefinitionStatus(&obj.Status, field.NewPath("status"))...)
	allErrs = append(allErrs, ValidateCustomResourceDefinitionStoredVersions(obj.Status.StoredVersions, obj.Spec.Versions, field.NewPath("status").Child("storedVersions"))...)
	return allErrs
}

// ValidateCustomResourceDefinitionStoredVersions statically validates
func ValidateCustomResourceDefinitionStoredVersions(storedVersions []string, versions []apiextensions.CustomResourceDefinitionVersion, fldPath *field.Path) field.ErrorList {
	if len(storedVersions) == 0 {
		return field.ErrorList{field.Invalid(fldPath, storedVersions, "must have at least one stored version")}
	}
	allErrs := field.ErrorList{}
	storedVersionsMap := map[string]int{}
	for i, v := range storedVersions {
		storedVersionsMap[v] = i
	}
	for _, v := range versions {
		_, ok := storedVersionsMap[v.Name]
		if v.Storage && !ok {
			allErrs = append(allErrs, field.Invalid(fldPath, v, "must have the storage version "+v.Name))
		}
		if ok {
			delete(storedVersionsMap, v.Name)
		}
	}

	for v, i := range storedVersionsMap {
		allErrs = append(allErrs, field.Invalid(fldPath.Index(i), v, "must appear in spec.versions"))
	}

	return allErrs
}

// ValidateUpdateCustomResourceDefinitionStatus statically validates
func ValidateUpdateCustomResourceDefinitionStatus(obj, oldObj *apiextensions.CustomResourceDefinition) field.ErrorList {
	allErrs := genericvalidation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateCustomResourceDefinitionStatus(&obj.Status, field.NewPath("status"))...)
	return allErrs
}

// ValidateCustomResourceDefinitionSpec statically validates
func ValidateCustomResourceDefinitionSpec(spec *apiextensions.CustomResourceDefinitionSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(spec.Group) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("group"), ""))
	} else if errs := validationutil.IsDNS1123Subdomain(spec.Group); len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("group"), spec.Group, strings.Join(errs, ",")))
	} else if len(strings.Split(spec.Group, ".")) < 2 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("group"), spec.Group, "should be a domain with at least one dot"))
	}

	switch spec.Scope {
	case "":
		allErrs = append(allErrs, field.Required(fldPath.Child("scope"), ""))
	case apiextensions.ClusterScoped, apiextensions.NamespaceScoped:
	default:
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("scope"), spec.Scope, []string{string(apiextensions.ClusterScoped), string(apiextensions.NamespaceScoped)}))
	}

	storageFlagCount := 0
	versionsMap := map[string]bool{}
	uniqueNames := true
	for i, version := range spec.Versions {
		if version.Storage {
			storageFlagCount++
		}
		if versionsMap[version.Name] {
			uniqueNames = false
		} else {
			versionsMap[version.Name] = true
		}
		if errs := validationutil.IsDNS1035Label(version.Name); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("versions").Index(i).Child("name"), spec.Versions[i].Name, strings.Join(errs, ",")))
		}
	}
	if !uniqueNames {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("versions"), spec.Versions, "must contain unique version names"))
	}
	if storageFlagCount != 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("versions"), spec.Versions, "must have exactly one version marked as storage version"))
	}
	if len(spec.Version) != 0 {
		if errs := validationutil.IsDNS1035Label(spec.Version); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("version"), spec.Version, strings.Join(errs, ",")))
		}
		if len(spec.Versions) >= 1 && spec.Versions[0].Name != spec.Version {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("version"), spec.Version, "must match the first version in spec.versions"))
		}
	}

	// in addition to the basic name restrictions, some names are required for spec, but not for status
	if len(spec.Names.Plural) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("names", "plural"), ""))
	}
	if len(spec.Names.Singular) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("names", "singular"), ""))
	}
	if len(spec.Names.Kind) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("names", "kind"), ""))
	}
	if len(spec.Names.ListKind) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("names", "listKind"), ""))
	}

	allErrs = append(allErrs, ValidateCustomResourceDefinitionNames(&spec.Names, fldPath.Child("names"))...)

	if utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceValidation) {
		statusEnabled := false
		if spec.Subresources != nil && spec.Subresources.Status != nil {
			statusEnabled = true
		}
		allErrs = append(allErrs, ValidateCustomResourceDefinitionValidation(spec.Validation, statusEnabled, fldPath.Child("validation"))...)
	} else if spec.Validation != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("validation"), "disabled by feature-gate CustomResourceValidation"))
	}

	if utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceSubresources) {
		allErrs = append(allErrs, ValidateCustomResourceDefinitionSubresources(spec.Subresources, fldPath.Child("subresources"))...)
	} else if spec.Subresources != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("subresources"), "disabled by feature-gate CustomResourceSubresources"))
	}

	for i := range spec.AdditionalPrinterColumns {
		if errs := ValidateCustomResourceColumnDefinition(&spec.AdditionalPrinterColumns[i], fldPath.Child("columns").Index(i)); len(errs) > 0 {
			allErrs = append(allErrs, errs...)
		}
	}

	return allErrs
}

// ValidateCustomResourceDefinitionSpecUpdate statically validates
func ValidateCustomResourceDefinitionSpecUpdate(spec, oldSpec *apiextensions.CustomResourceDefinitionSpec, established bool, fldPath *field.Path) field.ErrorList {
	allErrs := ValidateCustomResourceDefinitionSpec(spec, fldPath)

	if established {
		// these effect the storage and cannot be changed therefore
		allErrs = append(allErrs, genericvalidation.ValidateImmutableField(spec.Scope, oldSpec.Scope, fldPath.Child("scope"))...)
		allErrs = append(allErrs, genericvalidation.ValidateImmutableField(spec.Names.Kind, oldSpec.Names.Kind, fldPath.Child("names", "kind"))...)
	}

	// these affects the resource name, which is always immutable, so this can't be updated.
	allErrs = append(allErrs, genericvalidation.ValidateImmutableField(spec.Group, oldSpec.Group, fldPath.Child("group"))...)
	allErrs = append(allErrs, genericvalidation.ValidateImmutableField(spec.Names.Plural, oldSpec.Names.Plural, fldPath.Child("names", "plural"))...)

	return allErrs
}

// ValidateCustomResourceDefinitionStatus statically validates
func ValidateCustomResourceDefinitionStatus(status *apiextensions.CustomResourceDefinitionStatus, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateCustomResourceDefinitionNames(&status.AcceptedNames, fldPath.Child("acceptedNames"))...)
	return allErrs
}

// ValidateCustomResourceDefinitionNames statically validates
func ValidateCustomResourceDefinitionNames(names *apiextensions.CustomResourceDefinitionNames, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if errs := validationutil.IsDNS1035Label(names.Plural); len(names.Plural) > 0 && len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("plural"), names.Plural, strings.Join(errs, ",")))
	}
	if errs := validationutil.IsDNS1035Label(names.Singular); len(names.Singular) > 0 && len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("singular"), names.Singular, strings.Join(errs, ",")))
	}
	if errs := validationutil.IsDNS1035Label(strings.ToLower(names.Kind)); len(names.Kind) > 0 && len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("kind"), names.Kind, "may have mixed case, but should otherwise match: "+strings.Join(errs, ",")))
	}
	if errs := validationutil.IsDNS1035Label(strings.ToLower(names.ListKind)); len(names.ListKind) > 0 && len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("listKind"), names.ListKind, "may have mixed case, but should otherwise match: "+strings.Join(errs, ",")))
	}

	for i, shortName := range names.ShortNames {
		if errs := validationutil.IsDNS1035Label(shortName); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("shortNames").Index(i), shortName, strings.Join(errs, ",")))
		}
	}

	// kind and listKind may not be the same or parsing become ambiguous
	if len(names.Kind) > 0 && names.Kind == names.ListKind {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("listKind"), names.ListKind, "kind and listKind may not be the same"))
	}

	for i, category := range names.Categories {
		if errs := validationutil.IsDNS1035Label(category); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("categories").Index(i), category, strings.Join(errs, ",")))
		}
	}

	return allErrs
}

// ValidateCustomResourceColumnDefinition statically validates a printer column.
func ValidateCustomResourceColumnDefinition(col *apiextensions.CustomResourceColumnDefinition, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(col.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	}

	if len(col.Type) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("type"), fmt.Sprintf("must be one of %s", strings.Join(printerColumnDatatypes.List(), ","))))
	} else if !printerColumnDatatypes.Has(col.Type) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("type"), col.Type, fmt.Sprintf("must be one of %s", strings.Join(printerColumnDatatypes.List(), ","))))
	}

	if len(col.Format) > 0 && !customResourceColumnDefinitionFormats.Has(col.Format) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("format"), col.Format, fmt.Sprintf("must be one of %s", strings.Join(customResourceColumnDefinitionFormats.List(), ","))))
	}

	if len(col.JSONPath) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("JSONPath"), ""))
	} else if errs := validateSimpleJSONPath(col.JSONPath, fldPath.Child("JSONPath")); len(errs) > 0 {
		allErrs = append(allErrs, errs...)
	}

	return allErrs
}

// specStandardValidator applies validations for different OpenAPI specification versions.
type specStandardValidator interface {
	validate(spec *apiextensions.JSONSchemaProps, fldPath *field.Path) field.ErrorList
}

// ValidateCustomResourceDefinitionValidation statically validates
func ValidateCustomResourceDefinitionValidation(customResourceValidation *apiextensions.CustomResourceValidation, statusSubresourceEnabled bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if customResourceValidation == nil {
		return allErrs
	}

	if schema := customResourceValidation.OpenAPIV3Schema; schema != nil {
		// if the status subresource is enabled, only certain fields are allowed inside the root schema.
		// these fields are chosen such that, if status is extracted as properties["status"], it's validation is not lost.
		if utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceSubresources) && statusSubresourceEnabled {
			v := reflect.ValueOf(schema).Elem()
			for i := 0; i < v.NumField(); i++ {
				// skip zero values
				if value := v.Field(i).Interface(); reflect.DeepEqual(value, reflect.Zero(reflect.TypeOf(value)).Interface()) {
					continue
				}

				fieldName := v.Type().Field(i).Name

				// only "object" type is valid at root of the schema since validation schema for status is extracted as properties["status"]
				if fieldName == "Type" {
					if schema.Type != "object" {
						allErrs = append(allErrs, field.Invalid(fldPath.Child("openAPIV3Schema.type"), schema.Type, fmt.Sprintf(`only "object" is allowed as the type at the root of the schema if the status subresource is enabled`)))
						break
					}
					continue
				}

				if !allowedAtRootSchema(fieldName) {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("openAPIV3Schema"), *schema, fmt.Sprintf(`only %v fields are allowed at the root of the schema if the status subresource is enabled`, allowedFieldsAtRootSchema)))
					break
				}
			}
		}

		openAPIV3Schema := &specStandardValidatorV3{}
		allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(schema, fldPath.Child("openAPIV3Schema"), openAPIV3Schema)...)
	}

	// if validation passed otherwise, make sure we can actually construct a schema validator from this custom resource validation.
	if len(allErrs) == 0 {
		if _, _, err := apiservervalidation.NewSchemaValidator(customResourceValidation); err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, "", fmt.Sprintf("error building validator: %v", err)))
		}
	}
	return allErrs
}

// ValidateCustomResourceDefinitionOpenAPISchema statically validates
func ValidateCustomResourceDefinitionOpenAPISchema(schema *apiextensions.JSONSchemaProps, fldPath *field.Path, ssv specStandardValidator) field.ErrorList {
	allErrs := field.ErrorList{}

	if schema == nil {
		return allErrs
	}

	allErrs = append(allErrs, ssv.validate(schema, fldPath)...)

	if schema.UniqueItems == true {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("uniqueItems"), "uniqueItems cannot be set to true since the runtime complexity becomes quadratic"))
	}

	// additionalProperties and properties are mutual exclusive because otherwise they
	// contradict Kubernetes' API convention to ignore unknown fields.
	//
	// In other words:
	// - properties are for structs,
	// - additionalProperties are for map[string]interface{}
	//
	// Note: when patternProperties is added to OpenAPI some day, this will have to be
	//       restricted like additionalProperties.
	if schema.AdditionalProperties != nil {
		if len(schema.Properties) != 0 {
			if schema.AdditionalProperties.Allows == false || schema.AdditionalProperties.Schema != nil {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("additionalProperties"), "additionalProperties and properties are mutual exclusive"))
			}
		}
		allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(schema.AdditionalProperties.Schema, fldPath.Child("additionalProperties"), ssv)...)
	}

	if len(schema.Properties) != 0 {
		for property, jsonSchema := range schema.Properties {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("properties").Key(property), ssv)...)
		}
	}

	if len(schema.PatternProperties) != 0 {
		for property, jsonSchema := range schema.PatternProperties {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("patternProperties").Key(property), ssv)...)
		}
	}

	if schema.AdditionalItems != nil {
		allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(schema.AdditionalItems.Schema, fldPath.Child("additionalItems"), ssv)...)
	}

	allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(schema.Not, fldPath.Child("not"), ssv)...)

	if len(schema.AllOf) != 0 {
		for i, jsonSchema := range schema.AllOf {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("allOf").Index(i), ssv)...)
		}
	}

	if len(schema.OneOf) != 0 {
		for i, jsonSchema := range schema.OneOf {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("oneOf").Index(i), ssv)...)
		}
	}

	if len(schema.AnyOf) != 0 {
		for i, jsonSchema := range schema.AnyOf {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("anyOf").Index(i), ssv)...)
		}
	}

	if len(schema.Definitions) != 0 {
		for definition, jsonSchema := range schema.Definitions {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("definitions").Key(definition), ssv)...)
		}
	}

	if schema.Items != nil {
		allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(schema.Items.Schema, fldPath.Child("items"), ssv)...)
		if len(schema.Items.JSONSchemas) != 0 {
			for i, jsonSchema := range schema.Items.JSONSchemas {
				allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("items").Index(i), ssv)...)
			}
		}
	}

	if schema.Dependencies != nil {
		for dependency, jsonSchemaPropsOrStringArray := range schema.Dependencies {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(jsonSchemaPropsOrStringArray.Schema, fldPath.Child("dependencies").Key(dependency), ssv)...)
		}
	}

	return allErrs
}

type specStandardValidatorV3 struct{}

// validate validates against OpenAPI Schema v3.
func (v *specStandardValidatorV3) validate(schema *apiextensions.JSONSchemaProps, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if schema == nil {
		return allErrs
	}

	if schema.Default != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("default"), "default is not supported"))
	}

	if schema.ID != "" {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("id"), "id is not supported"))
	}

	if schema.AdditionalItems != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("additionalItems"), "additionalItems is not supported"))
	}

	if len(schema.PatternProperties) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("patternProperties"), "patternProperties is not supported"))
	}

	if len(schema.Definitions) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("definitions"), "definitions is not supported"))
	}

	if schema.Dependencies != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("dependencies"), "dependencies is not supported"))
	}

	if schema.Ref != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("$ref"), "$ref is not supported"))
	}

	if schema.Type == "null" {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("type"), "type cannot be set to null"))
	}

	if schema.Items != nil && len(schema.Items.JSONSchemas) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("items"), "items must be a schema object and not an array"))
	}

	return allErrs
}

// ValidateCustomResourceDefinitionSubresources statically validates
func ValidateCustomResourceDefinitionSubresources(subresources *apiextensions.CustomResourceSubresources, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if subresources == nil {
		return allErrs
	}

	if subresources.Scale != nil {
		if len(subresources.Scale.SpecReplicasPath) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("scale.specReplicasPath"), ""))
		} else {
			// should be constrained json path under .spec
			if errs := validateSimpleJSONPath(subresources.Scale.SpecReplicasPath, fldPath.Child("scale.specReplicasPath")); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			} else if !strings.HasPrefix(subresources.Scale.SpecReplicasPath, ".spec.") {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("scale.specReplicasPath"), subresources.Scale.SpecReplicasPath, "should be a json path under .spec"))
			}
		}

		if len(subresources.Scale.StatusReplicasPath) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("scale.statusReplicasPath"), ""))
		} else {
			// should be constrained json path under .status
			if errs := validateSimpleJSONPath(subresources.Scale.StatusReplicasPath, fldPath.Child("scale.statusReplicasPath")); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			} else if !strings.HasPrefix(subresources.Scale.StatusReplicasPath, ".status.") {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("scale.statusReplicasPath"), subresources.Scale.StatusReplicasPath, "should be a json path under .status"))
			}
		}

		// if labelSelectorPath is present, it should be a constrained json path under .status
		if subresources.Scale.LabelSelectorPath != nil && len(*subresources.Scale.LabelSelectorPath) > 0 {
			if errs := validateSimpleJSONPath(*subresources.Scale.LabelSelectorPath, fldPath.Child("scale.labelSelectorPath")); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			} else if !strings.HasPrefix(*subresources.Scale.LabelSelectorPath, ".status.") {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("scale.labelSelectorPath"), subresources.Scale.LabelSelectorPath, "should be a json path under .status"))
			}
		}
	}

	return allErrs
}

func validateSimpleJSONPath(s string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	switch {
	case len(s) == 0:
		allErrs = append(allErrs, field.Invalid(fldPath, s, "must not be empty"))
	case s[0] != '.':
		allErrs = append(allErrs, field.Invalid(fldPath, s, "must be a simple json path starting with ."))
	case s != ".":
		if cs := strings.Split(s[1:], "."); len(cs) < 1 {
			allErrs = append(allErrs, field.Invalid(fldPath, s, "must be a json path in the dot notation"))
		}
	}

	return allErrs
}

var allowedFieldsAtRootSchema = []string{"Description", "Type", "Format", "Title", "Maximum", "ExclusiveMaximum", "Minimum", "ExclusiveMinimum", "MaxLength", "MinLength", "Pattern", "MaxItems", "MinItems", "UniqueItems", "MultipleOf", "Required", "Items", "Properties", "ExternalDocs", "Example"}

func allowedAtRootSchema(field string) bool {
	for _, v := range allowedFieldsAtRootSchema {
		if field == v {
			return true
		}
	}
	return false
}
