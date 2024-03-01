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
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type ExpectedFieldError struct {
	Type   field.ErrorType
	Field  string
	Detail string
	// Ignored, but left in case we may want to enforce this in the future
	BadValue interface{}

	// If this error should be skipped
	SchemaSkipReason string

	// If it is not yet possible to reproduce this error exactly for schema
	// validation, then provides an alternative matching error
	SchemaType field.ErrorType
}

type ExpectedErrorList []ExpectedFieldError

func (e ExpectedErrorList) NativeErrors() field.ErrorList {
	var res field.ErrorList
	for _, err := range e {
		res = append(res, &field.Error{
			Type:  err.Type,
			Field: err.Field,
		})
	}
	return res
}

func (e ExpectedErrorList) SchemaErrors() field.ErrorList {
	var res field.ErrorList
	for _, err := range e {
		typ := err.Type
		if len(err.SchemaType) > 0 {
			typ = err.SchemaType
		}
		res = append(res, &field.Error{
			Type:  typ,
			Field: err.Field,
		})
	}
	return res
}

type TestCase[T runtime.Object, O any] struct {
	// Regex patterns of errors expected in any order
	Name           string
	ExpectedErrors ExpectedErrorList
	Object         T
	OldObject      T
	Options        O
}

type comparableObject interface {
	runtime.Object
	comparable
}

// invalid-accessmode: fails due to bug in kube-openapi gen not respecting +enum in lists
// unexpected-namespace: fails due to schemas not encoding scope of resource (also needed for kubectl-validate)
// bad-name: fails due to missing name formats in kube-openapi and cel
// bad-volume-zero-cpacity: fails due to missing ability to parameterize shared schemas/override neested schemas
// missing-accessmodes: fails due to not encoding resource scope
// too-many-sources: can be encoded but needs giant N^2 rule to use same field path
// _with_recycle_reclaim_policy 2: missing path.Clean CEL library function in rule checking hostPath.path
// invalidate-storage-class-name: fails due to missing name formats in CEL
// volume-bad-node-affinity: fails due to missing ability to paramerize shared template schema validation rules, alternatively could supply way to override shared schemas at point-of-use

func TestValidate[T comparableObject, O any](t *testing.T, scheme *runtime.Scheme, defs *resolver.DefinitionsSchemaResolver, validator func(T, O) field.ErrorList, cases ...TestCase[T, O]) {
	// Run standard validation test
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			t.Run("__internal__", func(t *testing.T) {
				nativeErrors := validator(c.Object, c.Options)
				t.Log("native errors", nativeErrors)

				if err := CompareErrorLists(c.ExpectedErrors.NativeErrors(), nativeErrors); err != nil {
					t.Fatal(fmt.Errorf("native object validation failed: %w", err))
				}
			})

			// Convert via the scheme the type to its versioned type
			gvks, _, err := scheme.ObjectKinds(c.Object)
			if err != nil {
				t.Fatal(err)
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
				t.Fatal(err)
			}

			groupVersions := scheme.VersionsForGroupKind(internalGVK.GroupKind())
			for _, gv := range groupVersions {
				if gv.Version == runtime.APIVersionInternal {
					continue
				}

				t.Run(gv.Version, func(t *testing.T) {
					gvk := gv.WithKind(internalGVK.Kind)

					// Look up its versioned type in the schema using the definition namer
					converted, err := scheme.ConvertToVersion(c.Object, gv)
					if err != nil {
						t.Fatal(err)
					}

					k := reflect.TypeOf(converted).Elem().PkgPath() + "." + gvk.Kind
					sch := defs.LookupSchema(k)
					if sch == nil {
						t.Fatal("definition not found")
					}

					// Convert to unstructured
					unstructuredMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(converted)
					if err != nil {
						t.Fatal(err)
					}
					unstructuredVersion := &unstructured.Unstructured{Object: unstructuredMap}
					unstructuredVersion.SetGroupVersionKind(gvk)

					openAPIValidator := validation.NewSchemaValidatorFromOpenAPI(sch)
					sts, err := kubeOpenAPIToStructuralSlow(sch)
					if err != nil {
						t.Fatal(err)
					}
					celValidator := cel.NewValidator(sts, true, celconfig.PerCallLimit)

					var fieldErrors field.ErrorList

					//!TODO: How to know if the object is namespace scoped?
					// we do not store that information in the schema.
					// Should we? i.e. mandate maxLength: 0 on namespace field (or cel error)?
					isNamespaceScoped := len(unstructuredVersion.GetNamespace()) > 0

					rest.FillObjectMetaSystemFields(unstructuredVersion)
					if len(unstructuredVersion.GetGenerateName()) > 0 && len(unstructuredVersion.GetName()) == 0 {
						unstructuredVersion.SetName(names.SimpleNameGenerator.GenerateName(unstructuredVersion.GetGenerateName()))
					}

					fieldErrors = append(fieldErrors, pkgapivalidation.ValidateObjectMetaAccessor(unstructuredVersion, isNamespaceScoped, path.ValidatePathSegmentName, field.NewPath("metadata"))...)
					fieldErrors = append(fieldErrors, schemaobjectmeta.Validate(nil, unstructuredVersion, sts, false)...)
					fieldErrors = append(fieldErrors, structurallisttype.ValidateListSetsAndMaps(nil, sts, unstructuredVersion.Object)...)

					if isNil(c.OldObject) {
						// ValidateCreate
						fieldErrors = append(fieldErrors, validation.ValidateCustomResource(nil, unstructuredVersion, openAPIValidator)...)
						celErrors, _ := celValidator.Validate(context.TODO(), nil, sts, unstructuredVersion.Object, nil, celconfig.RuntimeCELCostBudget)
						fieldErrors = append(fieldErrors, celErrors...)

					} else {
						// ValidateUpdate
						convertedOld, err := scheme.ConvertToVersion(c.OldObject, gv)
						if err != nil {
							t.Fatal(err)
						}
						unstructuredOldMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(convertedOld)
						if err != nil {
							t.Fatal(err)
						}
						unstructuredOldVersion := &unstructured.Unstructured{Object: unstructuredOldMap}
						unstructuredOldVersion.SetGroupVersionKind(gvk)

						fieldErrors = append(fieldErrors, validation.ValidateCustomResourceUpdate(nil, unstructuredOldVersion, unstructuredVersion, openAPIValidator)...)
						celErrors, _ := celValidator.Validate(context.TODO(), nil, sts, unstructuredVersion.Object, unstructuredOldVersion.Object, celconfig.RuntimeCELCostBudget)
						fieldErrors = append(fieldErrors, celErrors...)
					}

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

func isNil[T comparable](arg T) bool {
	var t T
	return arg == t
}

// Compares a declarative validation error list with a legit one and
// returns the diff
func CompareErrorLists(lhs field.ErrorList, rhs field.ErrorList) error {
	// Categorize each error by field path
	// Make sure each field path list has matching errors
	fieldsToErrorsLHS := map[string]field.ErrorList{}
	fieldsToErrorsRHS := map[string]field.ErrorList{}
	for _, v := range lhs {
		// We don't yet support validating BadValue, since CEL errors put name of
		// type instead of value.
		v.BadValue = nil

		if existing, exists := fieldsToErrorsLHS[v.Field]; exists {
			fieldsToErrorsLHS[v.Field] = append(existing, v)
		} else {
			fieldsToErrorsLHS[v.Field] = field.ErrorList{v}
		}
	}

	for _, v := range rhs {
		// We don't yet support validating BadValue, since CEL errors put name of
		// type instead of value.
		v.BadValue = nil

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
				rhsErr.Detail = ""
				lhsErr.Detail = ""

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
