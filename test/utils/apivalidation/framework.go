/*
Copyright 2024 The Kubernetes Authors.

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

package apivalidation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting"
	structurallisttype "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/listtype"
	schemaobjectmeta "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/objectmeta"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	pkgapivalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/api/validation/path"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/apiserver/pkg/endpoints/openapi"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type SkipReason string

const (
	// Rule was skipped because it is testing for a required error thrown for
	// a value that is defaulted in the schema. Our unversioned validation does
	// apply defaults, but schema validation is always versioned so defaults are
	// applied.
	SkipReasonDefaulted SkipReason = "defaulted"

	// Rule was skipped because it can only be applied once we add a CEL variables
	// feature to CEL validation.
	SkipReasonMissingCELVariables SkipReason = "missing CEL variables"
)

type ExpectedFieldError struct {
	Field string
	Type  field.ErrorType

	// Detail is checked that the actual error contains a substring of this value.
	// Errors are sorted by detail as a last resort so in some cases it may be
	// necessary to use a prefix of the actual detail for this value
	Detail string
	// Only checked for native errors, may be checked for schema errors in the future
	// (blocked by fact that some CRD/CEL errors place name of type for badvalue)
	BadValue interface{}

	// If this error should be skipped under schema validation for some reason
	SchemaSkipReason string
	// If this error should be skipped under native validation for some reason
	NativeSkipReason string

	// If it is not yet possible to reproduce this error exactly for schema
	// validation, then provides an alternative matching error
	SchemaType   field.ErrorType
	SchemaField  string
	SchemaDetail string
}

type ExpectedErrorList []ExpectedFieldError

func (e ExpectedErrorList) NativeErrors() field.ErrorList {
	var res field.ErrorList
	for _, err := range e {
		if len(err.NativeSkipReason) > 0 {
			continue
		}

		// field.Required and field.Forbidden unconditionally place "" in BadValue
		if err.BadValue == nil && (err.Type == field.ErrorTypeRequired || err.Type == field.ErrorTypeForbidden) {
			err.BadValue = ""
		}

		res = append(res, &field.Error{
			Type:     err.Type,
			Field:    err.Field,
			BadValue: err.BadValue,
			Detail:   err.Detail,
		})
	}
	return res
}

func (e ExpectedErrorList) SchemaErrors() field.ErrorList {
	var res field.ErrorList
	for _, err := range e {
		if len(err.SchemaSkipReason) > 0 {
			continue
		}
		typ := err.Type
		if len(err.SchemaType) > 0 {
			typ = err.SchemaType
		}

		fld := err.Field
		if len(err.SchemaField) > 0 {
			fld = err.SchemaField
		}

		dt := err.Detail
		if len(err.SchemaDetail) > 0 {
			dt = err.SchemaDetail
		}

		res = append(res, &field.Error{
			Type:   typ,
			Field:  fld,
			Detail: dt,
		})
	}
	return res
}

type TestCase[T any, O any] struct {
	// Regex patterns of errors expected in any order
	Name           string
	ExpectedErrors ExpectedErrorList
	Object         T
	OldObject      T
	Options        O
}

type comparableObject[T any] interface {
	runtime.Object
	*T
}

// Validates a part of a resource's schema against provided test cases.
// Useful for validating a type's schema which may be shared among several
// resources in isolation from any parent schema validations.
func TestValidateComponentWIP[T comparable, O any](
	t *testing.T,
	scheme *runtime.Scheme,
	defs *resolver.DefinitionsSchemaResolver,
	validator func(T, O) field.ErrorList,
	versionedTypes map[string]func() any,
	cases ...TestCase[T, O],
) {
	validators := map[string]schemaValidators{}
	namer := openapi.NewDefinitionNamer(scheme)

	for version, newFunc := range versionedTypes {
		example := newFunc()
		rtype := reflect.TypeOf(example)
		for rtype.Kind() == reflect.Ptr {
			rtype = rtype.Elem()
		}
		name, _ := namer.GetDefinitionName(rtype.PkgPath() + "." + rtype.Name())
		sch, err := defs.LookupSchema(name)
		if err != nil {
			t.Fatal(err)
		}

		openAPIValidator := validation.NewSchemaValidatorFromOpenAPI(sch)
		sts, err := kubeOpenAPIToStructuralSlow(sch)
		if err != nil {
			t.Fatal(err)
		}

		validators[version] = schemaValidators{
			newFunc:          newFunc,
			celValidator:     cel.NewValidator(sts, true, celconfig.PerCallLimit),
			openAPIValidator: openAPIValidator,
			sts:              sts,
		}
	}
	testValidate[T, O](t, scheme, validators, validator, cases...)
}

// Validates a top-level resource against various test cases for all known
// versions of the resource.
func TestValidate[T any, O any, TPtr comparableObject[T]](
	t *testing.T,
	scheme *runtime.Scheme,
	defs *resolver.DefinitionsSchemaResolver,
	validator func(TPtr, O) field.ErrorList,
	cases ...TestCase[TPtr, O],
) {

	// Convert via the scheme the type to its versioned type
	var emptyT T
	gvks, _, err := scheme.ObjectKinds(TPtr(&emptyT))
	if err != nil {
		t.Fatal(fmt.Errorf("failed to get object kinds for %T: %w", emptyT, err))
	} else if len(gvks) == 0 {
		t.Fatal("no kinds found")
	}

	// Find the internal type (this test framework is intended to be
	// used with internal types only)
	internalGVK, err := func() (schema.GroupVersionKind, error) {
		for _, gvk := range gvks {
			if gvk.Version == runtime.APIVersionInternal {
				return gvk, nil
			}
		}

		return schema.GroupVersionKind{}, fmt.Errorf("no internal type found")
	}()

	if err != nil {
		t.Fatal(fmt.Errorf("no internal type found: %w", err))
	}

	validators := map[string]schemaValidators{}

	groupVersions := scheme.VersionsForGroupKind(internalGVK.GroupKind())
	for _, gv := range groupVersions {
		if gv.Version == runtime.APIVersionInternal {
			continue
		}

		gvk := gv.WithKind(internalGVK.Kind)
		sch, err := defs.ResolveSchema(gvk)
		if err != nil {
			t.Fatal(err)
		}

		openAPIValidator := validation.NewSchemaValidatorFromOpenAPI(sch)
		sts, err := kubeOpenAPIToStructuralSlow(sch)
		if err != nil {
			t.Fatal(err)
		}

		// We currently have validations on metadata for deletioncost
		// annotations
		// These could be migrated to CEL which doesnt have such
		// a strict check to pass the test.
		//
		// if errs := structuralschema.ValidateStructuralWithOptions(nil, sts, structuralschema.ValidationOptions{
		// 	AllowNestedAdditionalProperties:                   true,
		// 	AllowNestedXValidations:                           true,
		// 	AllowValidationPropertiesWithAdditionalProperties: true,
		// }); len(errs) != 0 {
		// 	t.Fatal(errs)
		// }
		example, err := scheme.New(gvk)
		if err != nil {
			t.Fatal(err)
		}

		validators[gvk.Version] = schemaValidators{
			newFunc: func() any {
				return example.DeepCopyObject()
			},
			celValidator:     cel.NewValidator(sts, true, celconfig.PerCallLimit),
			openAPIValidator: openAPIValidator,
			sts:              sts,
		}
	}

	testValidate[TPtr, O](t, scheme, validators, validator, cases...)
}

type schemaValidators struct {
	newFunc          func() any
	celValidator     *cel.Validator
	openAPIValidator validation.SchemaValidator
	sts              *structuralschema.Structural
}

func testValidate[T comparable, O any](
	t *testing.T,
	scheme *runtime.Scheme,
	validators map[string]schemaValidators,
	nativeValidator func(T, O) field.ErrorList,
	cases ...TestCase[T, O],
) {
	t.Parallel()

	// Run standard validation test
	for _, c := range cases {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			// t.Parallel()

			t.Run("__internal__", func(t *testing.T) {
				t.Parallel()

				nativeErrors := nativeValidator(c.Object, c.Options)
				t.Log("native errors", nativeErrors)

				if err := CompareErrorLists(c.ExpectedErrors.NativeErrors(), nativeErrors); err != nil {
					t.Fatal(fmt.Errorf("native object validation failed: %w", err))
				}
			})

			for version, vals := range validators {
				sts := vals.sts
				openAPIValidator := vals.openAPIValidator
				celValidator := vals.celValidator
				vals := vals

				t.Run(version, func(t *testing.T) {
					t.Parallel()

					converted := vals.newFunc()
					err := scheme.Convert(c.Object, converted, nil)
					if err != nil {
						t.Fatal(err)
					}

					// Convert to unstructured
					unstructuredMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(converted)
					if err != nil {
						t.Fatal(err)
					}

					unstructuredVersion := &unstructured.Unstructured{Object: unstructuredMap}
					defaulting.PruneNonNullableNullsWithoutDefaults(unstructuredVersion.Object, sts)
					defaulting.Default(unstructuredVersion, sts)

					var fieldErrors field.ErrorList

					// Validate ObjectMeta if T is a runtime.Object
					var q any = c.Object
					if _, ok := q.(runtime.Object); ok {
						// Skip namespace scope validation for now.
						// We should decide whether we want to add that into schema (makes it non-structural due to namespace validation)
						// or add information into an x-extension (more beurocratic, but keeps schema structural and using native validation)
						isNamespaceScoped := len(unstructuredVersion.GetNamespace()) > 0

						rest.FillObjectMetaSystemFields(unstructuredVersion)
						if len(unstructuredVersion.GetGenerateName()) > 0 && len(unstructuredVersion.GetName()) == 0 {
							unstructuredVersion.SetName(names.SimpleNameGenerator.GenerateName(unstructuredVersion.GetGenerateName()))
						}

						fieldErrors = append(fieldErrors, pkgapivalidation.ValidateObjectMetaAccessor(unstructuredVersion, isNamespaceScoped, path.ValidatePathSegmentName, field.NewPath("metadata"))...)
					}

					fieldErrors = append(fieldErrors, schemaobjectmeta.Validate(nil, unstructuredVersion, sts, false)...)
					fieldErrors = append(fieldErrors, structurallisttype.ValidateListSetsAndMaps(nil, sts, unstructuredVersion.Object)...)

					var empty T
					if c.OldObject == empty {
						// ValidateCreate
						fieldErrors = append(fieldErrors, validation.ValidateCustomResource(nil, unstructuredVersion, openAPIValidator)...)
						celErrors, _ := celValidator.Validate(context.TODO(), nil, sts, unstructuredVersion.Object, nil, celconfig.RuntimeCELCostBudget)
						fieldErrors = append(fieldErrors, celErrors...)

					} else {
						// ValidateUpdate
						convertedOld := vals.newFunc()
						err := scheme.Convert(c.OldObject, convertedOld, nil)
						if err != nil {
							t.Fatal(err)
						}
						unstructuredOldMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(convertedOld)
						if err != nil {
							t.Fatal(err)
						}
						unstructuredOldVersion := &unstructured.Unstructured{Object: unstructuredOldMap}
						fieldErrors = append(fieldErrors, validation.ValidateCustomResourceUpdate(nil, unstructuredOldVersion, unstructuredVersion, openAPIValidator)...)
						celErrors, _ := celValidator.Validate(context.TODO(), nil, sts, unstructuredVersion.Object, unstructuredOldVersion.Object, celconfig.RuntimeCELCostBudget)
						fieldErrors = append(fieldErrors, celErrors...)
					}

					// Schema BadValue is not checked since CRD/CEL validation
					// often/always puts name of type for this value.
					for _, err := range fieldErrors {
						err.BadValue = nil
					}

					var filteredErrors field.ErrorList
					for _, err := range fieldErrors {
						// Ignore redundant schema allOf error until we can remove it
						if strings.Contains(err.Detail, "must validate all the schemas (allOf)") {
							continue
						}
						filteredErrors = append(filteredErrors, err)
					}
					fieldErrors = filteredErrors

					t.Log("unstructured errors", fieldErrors)
					if err := CompareErrorLists(c.ExpectedErrors.SchemaErrors(), fieldErrors); err != nil {
						t.Fatal(fmt.Errorf("versioned object validation failed: %w", err))
					}
				})
			}
		})
	}
}

// Convert the openapi schema to a structural schema
// This is a slow path that generates the structural schema through JSON marshalling
// and unmarshalling. It is used for testing purposes only.
func kubeOpenAPIToStructuralSlow(sch *spec.Schema) (*structuralschema.Structural, error) {
	bs, err := json.Marshal(sch)
	if err != nil {
		return nil, err
	}

	v1SchemaProps := &apiextensionsv1.JSONSchemaProps{}
	err = json.Unmarshal(bs, v1SchemaProps)
	if err != nil {
		return nil, err
	}
	internalSchema := &apiextensions.JSONSchemaProps{}
	err = apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(v1SchemaProps, internalSchema, nil)
	if err != nil {
		return nil, err
	}
	s, err := structuralschema.NewStructural(internalSchema)
	if err != nil {
		return nil, err
	}
	return s, nil
}

// Compares a declarative validation error list with a legit one and
// returns the diff
func CompareErrorLists(lhs field.ErrorList, rhs field.ErrorList) error {
	// Categorize each error by field path
	// Make sure each field path list has matching errors
	fieldsToErrorsLHS := map[string]field.ErrorList{}
	fieldsToErrorsRHS := map[string]field.ErrorList{}
	for _, v := range lhs {
		if existing, exists := fieldsToErrorsLHS[v.Field]; exists {
			fieldsToErrorsLHS[v.Field] = append(existing, v)
		} else {
			fieldsToErrorsLHS[v.Field] = field.ErrorList{v}
		}
	}

	for _, v := range rhs {
		if existing, exists := fieldsToErrorsRHS[v.Field]; exists {
			fieldsToErrorsRHS[v.Field] = append(existing, v)
		} else {
			fieldsToErrorsRHS[v.Field] = field.ErrorList{v}
		}
	}

	// Sort
	for _, v := range fieldsToErrorsLHS {
		sort.SliceStable(v, func(i, j int) bool {
			iV := v[i]
			jV := v[j]

			if iV.Type < jV.Type {
				return true
			} else if iV.Type > jV.Type {
				return false
			} else if iV.Detail < jV.Detail {
				return true
			} else if iV.Detail > jV.Detail {
				return false
			}
			return false
		})
	}

	for _, v := range fieldsToErrorsRHS {
		sort.SliceStable(v, func(i, j int) bool {
			iV := v[i]
			jV := v[j]

			if iV.Type < jV.Type {
				return true
			} else if iV.Type > jV.Type {
				return false
			} else if iV.Detail < jV.Detail {
				return true
			} else if iV.Detail > jV.Detail {
				return false
			}
			return false
		})
	}

	// The expected error detail is supposed to be a substring of the actual
	// detail. But cmp.Diff doesn't support that. So, assuming our error lists
	// are exhaustive matches, we wipe out detail fields for pairwise errors
	// after sorting
	for k, lhsErrors := range fieldsToErrorsLHS {
		rhsErrors, ok := fieldsToErrorsRHS[k]
		if !ok {
			continue
		} else if len(lhsErrors) != len(rhsErrors) {
			continue
		}

		for i, lhsErr := range lhsErrors {
			rhsErr := rhsErrors[i]

			if strings.Contains(rhsErr.Detail, lhsErr.Detail) {
				lhsErr.Detail = rhsErr.Detail

				lhsErrors[i] = lhsErr
				rhsErrors[i] = rhsErr
			}
		}

	}

	// Diff
	if diff := cmp.Diff(fieldsToErrorsLHS, fieldsToErrorsRHS); len(diff) != 0 {
		return errors.New(diff)
	}
	return nil
}
