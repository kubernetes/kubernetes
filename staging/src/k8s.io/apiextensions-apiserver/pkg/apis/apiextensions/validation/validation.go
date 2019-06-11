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

	"github.com/go-openapi/strfmt"
	govalidate "github.com/go-openapi/validate"
	schemaobjectmeta "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/objectmeta"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	genericvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/webhook"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
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

// ValidateCustomResourceDefinitionVersion statically validates.
func ValidateCustomResourceDefinitionVersion(version *apiextensions.CustomResourceDefinitionVersion, fldPath *field.Path, mustBeStructural, statusEnabled, allowDefaults bool) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateCustomResourceDefinitionValidation(version.Schema, mustBeStructural, statusEnabled, allowDefaults, fldPath.Child("schema"))...)
	allErrs = append(allErrs, ValidateCustomResourceDefinitionSubresources(version.Subresources, fldPath.Child("subresources"))...)
	for i := range version.AdditionalPrinterColumns {
		allErrs = append(allErrs, ValidateCustomResourceColumnDefinition(&version.AdditionalPrinterColumns[i], fldPath.Child("additionalPrinterColumns").Index(i))...)
	}
	return allErrs
}

// ValidateCustomResourceDefinitionSpec statically validates
func ValidateCustomResourceDefinitionSpec(spec *apiextensions.CustomResourceDefinitionSpec, fldPath *field.Path) field.ErrorList {
	allowDefaults := utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceDefaulting)
	return validateCustomResourceDefinitionSpec(spec, true, allowDefaults, fldPath)
}

func validateCustomResourceDefinitionSpec(spec *apiextensions.CustomResourceDefinitionSpec, requireRecognizedVersion, allowDefaults bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(spec.Group) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("group"), ""))
	} else if errs := utilvalidation.IsDNS1123Subdomain(spec.Group); len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("group"), spec.Group, strings.Join(errs, ",")))
	} else if len(strings.Split(spec.Group, ".")) < 2 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("group"), spec.Group, "should be a domain with at least one dot"))
	}

	allErrs = append(allErrs, validateEnumStrings(fldPath.Child("scope"), string(spec.Scope), []string{string(apiextensions.ClusterScoped), string(apiextensions.NamespaceScoped)}, true)...)

	mustBeStructural := false
	if spec.PreserveUnknownFields == nil || *spec.PreserveUnknownFields == false {
		mustBeStructural = true
		// check that either a global schema or versioned schemas are set
		if spec.Validation == nil || spec.Validation.OpenAPIV3Schema == nil {
			for i, v := range spec.Versions {
				schemaPath := fldPath.Child("versions").Index(i).Child("schema", "openAPIV3Schema")
				if v.Served && (v.Schema == nil || v.Schema.OpenAPIV3Schema == nil) {
					allErrs = append(allErrs, field.Required(schemaPath, "because otherwise all fields are pruned"))
				}
			}
		}
	}
	if allowDefaults && specHasDefaults(spec) {
		mustBeStructural = true
		if spec.PreserveUnknownFields == nil || *spec.PreserveUnknownFields == true {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("preserveUnknownFields"), true, "must be false in order to use defaults in the schema"))
		}
	}
	if specHasKubernetesExtensions(spec) {
		mustBeStructural = true
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
		if errs := utilvalidation.IsDNS1035Label(version.Name); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("versions").Index(i).Child("name"), spec.Versions[i].Name, strings.Join(errs, ",")))
		}
		subresources := getSubresourcesForVersion(spec, version.Name)
		allErrs = append(allErrs, ValidateCustomResourceDefinitionVersion(&version, fldPath.Child("versions").Index(i), mustBeStructural, hasStatusEnabled(subresources), allowDefaults)...)
	}

	// The top-level and per-version fields are mutual exclusive
	if spec.Validation != nil && hasPerVersionSchema(spec.Versions) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("validation"), "top-level and per-version schemas are mutually exclusive"))
	}
	if spec.Subresources != nil && hasPerVersionSubresources(spec.Versions) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("subresources"), "top-level and per-version subresources are mutually exclusive"))
	}
	if len(spec.AdditionalPrinterColumns) > 0 && hasPerVersionColumns(spec.Versions) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("additionalPrinterColumns"), "top-level and per-version additionalPrinterColumns are mutually exclusive"))
	}

	// Per-version fields may not all be set to identical values (top-level field should be used instead)
	if hasIdenticalPerVersionSchema(spec.Versions) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("versions"), spec.Versions, "per-version schemas may not all be set to identical values (top-level validation should be used instead)"))
	}
	if hasIdenticalPerVersionSubresources(spec.Versions) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("versions"), spec.Versions, "per-version subresources may not all be set to identical values (top-level subresources should be used instead)"))
	}
	if hasIdenticalPerVersionColumns(spec.Versions) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("versions"), spec.Versions, "per-version additionalPrinterColumns may not all be set to identical values (top-level additionalPrinterColumns should be used instead)"))
	}

	if !uniqueNames {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("versions"), spec.Versions, "must contain unique version names"))
	}
	if storageFlagCount != 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("versions"), spec.Versions, "must have exactly one version marked as storage version"))
	}
	if len(spec.Version) != 0 {
		if errs := utilvalidation.IsDNS1035Label(spec.Version); len(errs) > 0 {
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
	allErrs = append(allErrs, ValidateCustomResourceDefinitionValidation(spec.Validation, mustBeStructural, hasAnyStatusEnabled(spec), allowDefaults, fldPath.Child("validation"))...)
	allErrs = append(allErrs, ValidateCustomResourceDefinitionSubresources(spec.Subresources, fldPath.Child("subresources"))...)

	for i := range spec.AdditionalPrinterColumns {
		if errs := ValidateCustomResourceColumnDefinition(&spec.AdditionalPrinterColumns[i], fldPath.Child("additionalPrinterColumns").Index(i)); len(errs) > 0 {
			allErrs = append(allErrs, errs...)
		}
	}

	if (spec.Conversion != nil && spec.Conversion.Strategy != apiextensions.NoneConverter) && (spec.PreserveUnknownFields == nil || *spec.PreserveUnknownFields) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conversion").Child("strategy"), spec.Conversion.Strategy, "must be None if spec.preserveUnknownFields is true"))
	}
	allErrs = append(allErrs, validateCustomResourceConversion(spec.Conversion, requireRecognizedVersion, fldPath.Child("conversion"))...)

	return allErrs
}

func validateEnumStrings(fldPath *field.Path, value string, accepted []string, required bool) field.ErrorList {
	if value == "" {
		if required {
			return field.ErrorList{field.Required(fldPath, "")}
		}
		return field.ErrorList{}
	}
	for _, a := range accepted {
		if a == value {
			return field.ErrorList{}
		}
	}
	return field.ErrorList{field.NotSupported(fldPath, value, accepted)}
}

var acceptedConversionReviewVersion = []string{v1beta1.SchemeGroupVersion.Version}

func isAcceptedConversionReviewVersion(v string) bool {
	for _, version := range acceptedConversionReviewVersion {
		if v == version {
			return true
		}
	}
	return false
}

func validateConversionReviewVersions(versions []string, requireRecognizedVersion bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(versions) < 1 {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	} else {
		seen := map[string]bool{}
		hasAcceptedVersion := false
		for i, v := range versions {
			if seen[v] {
				allErrs = append(allErrs, field.Invalid(fldPath.Index(i), v, "duplicate version"))
				continue
			}
			seen[v] = true
			for _, errString := range utilvalidation.IsDNS1035Label(v) {
				allErrs = append(allErrs, field.Invalid(fldPath.Index(i), v, errString))
			}
			if isAcceptedConversionReviewVersion(v) {
				hasAcceptedVersion = true
			}
		}
		if requireRecognizedVersion && !hasAcceptedVersion {
			allErrs = append(allErrs, field.Invalid(
				fldPath, versions,
				fmt.Sprintf("none of the versions accepted by this server. accepted version(s) are %v",
					strings.Join(acceptedConversionReviewVersion, ", "))))
		}
	}
	return allErrs
}

// hasValidConversionReviewVersion return true if there is a valid version or if the list is empty.
func hasValidConversionReviewVersionOrEmpty(versions []string) bool {
	if len(versions) < 1 {
		return true
	}
	for _, v := range versions {
		if isAcceptedConversionReviewVersion(v) {
			return true
		}
	}
	return false
}

// ValidateCustomResourceConversion statically validates
func ValidateCustomResourceConversion(conversion *apiextensions.CustomResourceConversion, fldPath *field.Path) field.ErrorList {
	return validateCustomResourceConversion(conversion, true, fldPath)
}

func validateCustomResourceConversion(conversion *apiextensions.CustomResourceConversion, requireRecognizedVersion bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if conversion == nil {
		return allErrs
	}
	allErrs = append(allErrs, validateEnumStrings(fldPath.Child("strategy"), string(conversion.Strategy), []string{string(apiextensions.NoneConverter), string(apiextensions.WebhookConverter)}, true)...)
	if conversion.Strategy == apiextensions.WebhookConverter {
		if conversion.WebhookClientConfig == nil {
			if utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceWebhookConversion) {
				allErrs = append(allErrs, field.Required(fldPath.Child("webhookClientConfig"), "required when strategy is set to Webhook"))
			} else {
				allErrs = append(allErrs, field.Required(fldPath.Child("webhookClientConfig"), "required when strategy is set to Webhook, but not allowed because the CustomResourceWebhookConversion feature is disabled"))
			}
		} else {
			cc := conversion.WebhookClientConfig
			switch {
			case (cc.URL == nil) == (cc.Service == nil):
				allErrs = append(allErrs, field.Required(fldPath.Child("webhookClientConfig"), "exactly one of url or service is required"))
			case cc.URL != nil:
				allErrs = append(allErrs, webhook.ValidateWebhookURL(fldPath.Child("webhookClientConfig").Child("url"), *cc.URL, true)...)
			case cc.Service != nil:
				allErrs = append(allErrs, webhook.ValidateWebhookService(fldPath.Child("webhookClientConfig").Child("service"), cc.Service.Name, cc.Service.Namespace, cc.Service.Path, cc.Service.Port)...)
			}
		}
		allErrs = append(allErrs, validateConversionReviewVersions(conversion.ConversionReviewVersions, requireRecognizedVersion, fldPath.Child("conversionReviewVersions"))...)
	} else {
		if conversion.WebhookClientConfig != nil {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("webhookClientConfig"), "should not be set when strategy is not set to Webhook"))
		}
		if len(conversion.ConversionReviewVersions) > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("conversionReviewVersions"), "should not be set when strategy is not set to Webhook"))
		}
	}
	return allErrs
}

// ValidateCustomResourceDefinitionSpecUpdate statically validates
func ValidateCustomResourceDefinitionSpecUpdate(spec, oldSpec *apiextensions.CustomResourceDefinitionSpec, established bool, fldPath *field.Path) field.ErrorList {
	requireRecognizedVersion := oldSpec.Conversion == nil || hasValidConversionReviewVersionOrEmpty(oldSpec.Conversion.ConversionReviewVersions)

	// find out whether any schema had default before. Then we keep allowing it.
	allowDefaults := utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceDefaulting) || specHasDefaults(oldSpec)

	allErrs := validateCustomResourceDefinitionSpec(spec, requireRecognizedVersion, allowDefaults, fldPath)

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

// getSubresourcesForVersion returns the subresources for given version in given CRD spec.
// NOTE That this function assumes version always exist since it's used by the validation process
// that iterates through the existing versions.
func getSubresourcesForVersion(crd *apiextensions.CustomResourceDefinitionSpec, version string) *apiextensions.CustomResourceSubresources {
	if !hasPerVersionSubresources(crd.Versions) {
		return crd.Subresources
	}
	for _, v := range crd.Versions {
		if version == v.Name {
			return v.Subresources
		}
	}
	return nil
}

// hasAnyStatusEnabled returns true if given CRD spec has at least one Status Subresource set
// among the top-level and per-version Subresources.
func hasAnyStatusEnabled(crd *apiextensions.CustomResourceDefinitionSpec) bool {
	if hasStatusEnabled(crd.Subresources) {
		return true
	}
	for _, v := range crd.Versions {
		if hasStatusEnabled(v.Subresources) {
			return true
		}
	}
	return false
}

// hasStatusEnabled returns true if given CRD Subresources has non-nil Status set.
func hasStatusEnabled(subresources *apiextensions.CustomResourceSubresources) bool {
	if subresources != nil && subresources.Status != nil {
		return true
	}
	return false
}

// hasPerVersionSchema returns true if a CRD uses per-version schema.
func hasPerVersionSchema(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if v.Schema != nil {
			return true
		}
	}
	return false
}

// hasPerVersionSubresources returns true if a CRD uses per-version subresources.
func hasPerVersionSubresources(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if v.Subresources != nil {
			return true
		}
	}
	return false
}

// hasPerVersionColumns returns true if a CRD uses per-version columns.
func hasPerVersionColumns(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	for _, v := range versions {
		if len(v.AdditionalPrinterColumns) > 0 {
			return true
		}
	}
	return false
}

// hasIdenticalPerVersionSchema returns true if a CRD sets identical non-nil values
// to all per-version schemas
func hasIdenticalPerVersionSchema(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	if len(versions) == 0 {
		return false
	}
	value := versions[0].Schema
	for _, v := range versions {
		if v.Schema == nil || !apiequality.Semantic.DeepEqual(v.Schema, value) {
			return false
		}
	}
	return true
}

// hasIdenticalPerVersionSubresources returns true if a CRD sets identical non-nil values
// to all per-version subresources
func hasIdenticalPerVersionSubresources(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	if len(versions) == 0 {
		return false
	}
	value := versions[0].Subresources
	for _, v := range versions {
		if v.Subresources == nil || !apiequality.Semantic.DeepEqual(v.Subresources, value) {
			return false
		}
	}
	return true
}

// hasIdenticalPerVersionColumns returns true if a CRD sets identical non-nil values
// to all per-version columns
func hasIdenticalPerVersionColumns(versions []apiextensions.CustomResourceDefinitionVersion) bool {
	if len(versions) == 0 {
		return false
	}
	value := versions[0].AdditionalPrinterColumns
	for _, v := range versions {
		if len(v.AdditionalPrinterColumns) == 0 || !apiequality.Semantic.DeepEqual(v.AdditionalPrinterColumns, value) {
			return false
		}
	}
	return true
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
	if errs := utilvalidation.IsDNS1035Label(names.Plural); len(names.Plural) > 0 && len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("plural"), names.Plural, strings.Join(errs, ",")))
	}
	if errs := utilvalidation.IsDNS1035Label(names.Singular); len(names.Singular) > 0 && len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("singular"), names.Singular, strings.Join(errs, ",")))
	}
	if errs := utilvalidation.IsDNS1035Label(strings.ToLower(names.Kind)); len(names.Kind) > 0 && len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("kind"), names.Kind, "may have mixed case, but should otherwise match: "+strings.Join(errs, ",")))
	}
	if errs := utilvalidation.IsDNS1035Label(strings.ToLower(names.ListKind)); len(names.ListKind) > 0 && len(errs) > 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("listKind"), names.ListKind, "may have mixed case, but should otherwise match: "+strings.Join(errs, ",")))
	}

	for i, shortName := range names.ShortNames {
		if errs := utilvalidation.IsDNS1035Label(shortName); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("shortNames").Index(i), shortName, strings.Join(errs, ",")))
		}
	}

	// kind and listKind may not be the same or parsing become ambiguous
	if len(names.Kind) > 0 && names.Kind == names.ListKind {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("listKind"), names.ListKind, "kind and listKind may not be the same"))
	}

	for i, category := range names.Categories {
		if errs := utilvalidation.IsDNS1035Label(category); len(errs) > 0 {
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
	withForbiddenDefaults(reason string) specStandardValidator

	// insideResourceMeta returns true when validating either TypeMeta or ObjectMeta, from an embedded resource or on the top-level.
	insideResourceMeta() bool
	withInsideResourceMeta() specStandardValidator
}

// ValidateCustomResourceDefinitionValidation statically validates
func ValidateCustomResourceDefinitionValidation(customResourceValidation *apiextensions.CustomResourceValidation, mustBeStructural, statusSubresourceEnabled, allowDefaults bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if customResourceValidation == nil {
		return allErrs
	}

	if schema := customResourceValidation.OpenAPIV3Schema; schema != nil {
		// if the status subresource is enabled, only certain fields are allowed inside the root schema.
		// these fields are chosen such that, if status is extracted as properties["status"], it's validation is not lost.
		if statusSubresourceEnabled {
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

		if schema.Nullable {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("openAPIV3Schema.nullable"), fmt.Sprintf(`nullable cannot be true at the root`)))
		}

		openAPIV3Schema := &specStandardValidatorV3{
			allowDefaults: allowDefaults,
		}
		allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(schema, fldPath.Child("openAPIV3Schema"), openAPIV3Schema, true)...)

		if mustBeStructural {
			if ss, err := structuralschema.NewStructural(schema); err != nil {
				// if the generic schema validation did its job, we should never get an error here. Hence, we hide it if there are validation errors already.
				if len(allErrs) == 0 {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("openAPIV3Schema"), "", err.Error()))
				}
			} else {
				allErrs = append(allErrs, structuralschema.ValidateStructural(ss, fldPath.Child("openAPIV3Schema"))...)
			}
		}
	}

	// if validation passed otherwise, make sure we can actually construct a schema validator from this custom resource validation.
	if len(allErrs) == 0 {
		if _, _, err := apiservervalidation.NewSchemaValidator(customResourceValidation); err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, "", fmt.Sprintf("error building validator: %v", err)))
		}
	}
	return allErrs
}

var metaFields = sets.NewString("metadata", "apiVersion", "kind")

// ValidateCustomResourceDefinitionOpenAPISchema statically validates
func ValidateCustomResourceDefinitionOpenAPISchema(schema *apiextensions.JSONSchemaProps, fldPath *field.Path, ssv specStandardValidator, isRoot bool) field.ErrorList {
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
		// Note: we forbid additionalProperties at resource root, both embedded and top-level.
		//       But further inside, additionalProperites is possible, e.g. for labels or annotations.
		subSsv := ssv
		if ssv.insideResourceMeta() {
			// we have to forbid defaults inside additionalProperties because pruning without actual value is ambiguous
			subSsv = ssv.withForbiddenDefaults("inside additionalProperties applying to object metadata")
		}
		allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(schema.AdditionalProperties.Schema, fldPath.Child("additionalProperties"), subSsv, false)...)
	}

	if len(schema.Properties) != 0 {
		for property, jsonSchema := range schema.Properties {
			subSsv := ssv
			if (isRoot || schema.XEmbeddedResource) && metaFields.Has(property) {
				// we recurse into the schema that applies to ObjectMeta.
				subSsv = ssv.withInsideResourceMeta()
				if isRoot {
					subSsv = subSsv.withForbiddenDefaults(fmt.Sprintf("in top-level %s", property))
				}
			}
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("properties").Key(property), subSsv, false)...)
		}
	}

	allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(schema.Not, fldPath.Child("not"), ssv, false)...)

	if len(schema.AllOf) != 0 {
		for i, jsonSchema := range schema.AllOf {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("allOf").Index(i), ssv, false)...)
		}
	}

	if len(schema.OneOf) != 0 {
		for i, jsonSchema := range schema.OneOf {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("oneOf").Index(i), ssv, false)...)
		}
	}

	if len(schema.AnyOf) != 0 {
		for i, jsonSchema := range schema.AnyOf {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("anyOf").Index(i), ssv, false)...)
		}
	}

	if len(schema.Definitions) != 0 {
		for definition, jsonSchema := range schema.Definitions {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("definitions").Key(definition), ssv, false)...)
		}
	}

	if schema.Items != nil {
		allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(schema.Items.Schema, fldPath.Child("items"), ssv, false)...)
		if len(schema.Items.JSONSchemas) != 0 {
			for i, jsonSchema := range schema.Items.JSONSchemas {
				allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(&jsonSchema, fldPath.Child("items").Index(i), ssv, false)...)
			}
		}
	}

	if schema.Dependencies != nil {
		for dependency, jsonSchemaPropsOrStringArray := range schema.Dependencies {
			allErrs = append(allErrs, ValidateCustomResourceDefinitionOpenAPISchema(jsonSchemaPropsOrStringArray.Schema, fldPath.Child("dependencies").Key(dependency), ssv, false)...)
		}
	}

	if schema.XPreserveUnknownFields != nil && *schema.XPreserveUnknownFields == false {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("x-kubernetes-preserve-unknown-fields"), *schema.XPreserveUnknownFields, "must be true or undefined"))
	}

	return allErrs
}

type specStandardValidatorV3 struct {
	allowDefaults          bool
	disallowDefaultsReason string
	isInsideResourceMeta   bool
}

func (v *specStandardValidatorV3) withForbiddenDefaults(reason string) specStandardValidator {
	clone := *v
	clone.disallowDefaultsReason = reason
	clone.allowDefaults = false
	return &clone
}

func (v *specStandardValidatorV3) withInsideResourceMeta() specStandardValidator {
	clone := *v
	clone.isInsideResourceMeta = true
	return &clone
}

func (v *specStandardValidatorV3) insideResourceMeta() bool {
	return v.isInsideResourceMeta
}

// validate validates against OpenAPI Schema v3.
func (v *specStandardValidatorV3) validate(schema *apiextensions.JSONSchemaProps, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if schema == nil {
		return allErrs
	}

	//
	// WARNING: if anything new is allowed below, NewStructural must be adapted to support it.
	//

	if schema.Default != nil {
		if v.allowDefaults {
			if s, err := structuralschema.NewStructural(schema); err == nil {
				// ignore errors here locally. They will show up for the root of the schema.

				clone := runtime.DeepCopyJSONValue(interface{}(*schema.Default))
				if !v.isInsideResourceMeta || s.XEmbeddedResource {
					pruning.Prune(clone, s, s.XEmbeddedResource)
					// If we are under metadata, there are implicitly specified fields like kind, apiVersion, metadata, labels.
					// We cannot prune as they are pruned as well. This allows more defaults than we would like to.
					// TODO: be precise about pruning under metadata
				}
				// TODO: coerce correctly if we are not at the object root, but somewhere below.
				if err := schemaobjectmeta.Coerce(fldPath, clone, s, s.XEmbeddedResource, false); err != nil {
					allErrs = append(allErrs, err)
				}
				if !reflect.DeepEqual(clone, interface{}(*schema.Default)) {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("default"), schema.Default, "must not have unknown fields"))
				} else if s.XEmbeddedResource {
					// validate an embedded resource
					schemaobjectmeta.Validate(fldPath, interface{}(*schema.Default), nil, true)
				}

				// validate the default value with user the provided schema.
				validator := govalidate.NewSchemaValidator(s.ToGoOpenAPI(), nil, "", strfmt.Default)
				if err := apiservervalidation.ValidateCustomResource(interface{}(*schema.Default), validator); err != nil {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("default"), schema.Default, fmt.Sprintf("must validate: %v", err)))
				}
			}
		} else {
			detail := "must not be set"
			if len(v.disallowDefaultsReason) > 0 {
				detail += " " + v.disallowDefaultsReason
			}
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("default"), detail))
		}
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
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("type"), "type cannot be set to null, use nullable as an alternative"))
	}

	if schema.Items != nil && len(schema.Items.JSONSchemas) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("items"), "items must be a schema object and not an array"))
	}

	if v.isInsideResourceMeta && schema.XEmbeddedResource {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("x-kubernetes-embedded-resource"), "must not be used inside of resource meta"))
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
			} else if !strings.HasPrefix(*subresources.Scale.LabelSelectorPath, ".spec.") && !strings.HasPrefix(*subresources.Scale.LabelSelectorPath, ".status.") {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("scale.labelSelectorPath"), subresources.Scale.LabelSelectorPath, "should be a json path under either .spec or .status"))
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

var allowedFieldsAtRootSchema = []string{"Description", "Type", "Format", "Title", "Maximum", "ExclusiveMaximum", "Minimum", "ExclusiveMinimum", "MaxLength", "MinLength", "Pattern", "MaxItems", "MinItems", "UniqueItems", "MultipleOf", "Required", "Items", "Properties", "ExternalDocs", "Example", "XPreserveUnknownFields"}

func allowedAtRootSchema(field string) bool {
	for _, v := range allowedFieldsAtRootSchema {
		if field == v {
			return true
		}
	}
	return false
}

func specHasDefaults(spec *apiextensions.CustomResourceDefinitionSpec) bool {
	if spec.Validation != nil && schemaHasDefaults(spec.Validation.OpenAPIV3Schema) {
		return true
	}
	for _, v := range spec.Versions {
		if v.Schema != nil && schemaHasDefaults(v.Schema.OpenAPIV3Schema) {
			return true
		}
	}
	return false
}

func schemaHasDefaults(s *apiextensions.JSONSchemaProps) bool {
	if s == nil {
		return false
	}

	if s.Default != nil {
		return true
	}

	if s.Items != nil {
		if s.Items != nil && schemaHasDefaults(s.Items.Schema) {
			return true
		}
		for _, s := range s.Items.JSONSchemas {
			if schemaHasDefaults(&s) {
				return true
			}
		}
	}
	for _, s := range s.AllOf {
		if schemaHasDefaults(&s) {
			return true
		}
	}
	for _, s := range s.AnyOf {
		if schemaHasDefaults(&s) {
			return true
		}
	}
	for _, s := range s.OneOf {
		if schemaHasDefaults(&s) {
			return true
		}
	}
	if schemaHasDefaults(s.Not) {
		return true
	}
	for _, s := range s.Properties {
		if schemaHasDefaults(&s) {
			return true
		}
	}
	if s.AdditionalProperties != nil {
		if schemaHasDefaults(s.AdditionalProperties.Schema) {
			return true
		}
	}
	for _, s := range s.PatternProperties {
		if schemaHasDefaults(&s) {
			return true
		}
	}
	if s.AdditionalItems != nil {
		if schemaHasDefaults(s.AdditionalItems.Schema) {
			return true
		}
	}
	for _, s := range s.Definitions {
		if schemaHasDefaults(&s) {
			return true
		}
	}
	for _, d := range s.Dependencies {
		if schemaHasDefaults(d.Schema) {
			return true
		}
	}

	return false
}

func specHasKubernetesExtensions(spec *apiextensions.CustomResourceDefinitionSpec) bool {
	if spec.Validation != nil && schemaHasKubernetesExtensions(spec.Validation.OpenAPIV3Schema) {
		return true
	}
	for _, v := range spec.Versions {
		if v.Schema != nil && schemaHasKubernetesExtensions(v.Schema.OpenAPIV3Schema) {
			return true
		}
	}
	return false
}

func schemaHasKubernetesExtensions(s *apiextensions.JSONSchemaProps) bool {
	if s == nil {
		return false
	}

	if s.XEmbeddedResource || s.XPreserveUnknownFields != nil || s.XIntOrString {
		return true
	}

	if s.Items != nil {
		if s.Items != nil && schemaHasKubernetesExtensions(s.Items.Schema) {
			return true
		}
		for _, s := range s.Items.JSONSchemas {
			if schemaHasKubernetesExtensions(&s) {
				return true
			}
		}
	}
	for _, s := range s.AllOf {
		if schemaHasKubernetesExtensions(&s) {
			return true
		}
	}
	for _, s := range s.AnyOf {
		if schemaHasKubernetesExtensions(&s) {
			return true
		}
	}
	for _, s := range s.OneOf {
		if schemaHasKubernetesExtensions(&s) {
			return true
		}
	}
	if schemaHasKubernetesExtensions(s.Not) {
		return true
	}
	for _, s := range s.Properties {
		if schemaHasKubernetesExtensions(&s) {
			return true
		}
	}
	if s.AdditionalProperties != nil {
		if schemaHasKubernetesExtensions(s.AdditionalProperties.Schema) {
			return true
		}
	}
	for _, s := range s.PatternProperties {
		if schemaHasKubernetesExtensions(&s) {
			return true
		}
	}
	if s.AdditionalItems != nil {
		if schemaHasKubernetesExtensions(s.AdditionalItems.Schema) {
			return true
		}
	}
	for _, s := range s.Definitions {
		if schemaHasKubernetesExtensions(&s) {
			return true
		}
	}
	for _, d := range s.Dependencies {
		if schemaHasKubernetesExtensions(d.Schema) {
			return true
		}
	}

	return false
}
