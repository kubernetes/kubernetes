package testing

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel/apivalidation"
	"k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
)

type TestCase[T runtime.Object, O any] struct {
	// Regex patterns of errors expected in any order
	Name           string
	ExpectedErrors []string
	Object         T
	OldObject      T
	Options        O
}

// Matches the list of expected errors against the provided error list
func (t TestCase[T, O]) Verify(errors field.ErrorList) error {
	if len(t.ExpectedErrors) != len(errors) {
		return fmt.Errorf("expected %d errors, got %d", len(t.ExpectedErrors), len(errors))
	}
	return nil
}

func TestValidate[T runtime.Object, O any](t *testing.T, scheme *runtime.Scheme, defs *resolver.DefinitionsSchemaResolver, validator func(T, O) field.ErrorList, cases ...TestCase[T, O]) {
	// Run standard validation test
	// namer := openapinamer.NewDefinitionNamer(scheme)

	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			nativeErrors := validator(c.Object, c.Options)
			if err := c.Verify(nativeErrors); err != nil {
				t.Fatal(fmt.Errorf("native object validation failed: %w", err))
			}

			// Convert via the scheme the type to its versioned type
			gvks, _, err := scheme.ObjectKinds(c.Object)
			if err != nil {
				t.Fatal(err)

			}

			//!TODO: multiple gvs?
			for _, gvk := range gvks {
				groupVersions := scheme.VersionsForGroupKind(gvk.GroupKind())
				for _, gv := range groupVersions {
					if gv.Version == runtime.APIVersionInternal {
						continue
					}

					t.Run(gv.WithKind(gvk.Kind).String(), func(t *testing.T) {
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

						vSch := apivalidation.NewValidationSchema(&openapi.Schema{Schema: sch})
						schemaErrors := vSch.Validate(converted, apivalidation.ValidationOptions{})
						if err := c.Verify(schemaErrors); err != nil {
							t.Error(err)
						}

						if err := CompareErrorLists(nativeErrors, schemaErrors); err != nil {
							t.Error(err)
						}
					})
				}
			}

		})
	}
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

	// Diff
	if diff := cmp.Diff(fieldsToErrorsLHS, fieldsToErrorsRHS); len(diff) != 0 {
		return errors.New(diff)
	}
	return nil
}
