/*
Copyright 2023 The Kubernetes Authors.

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

package cel

import (
	"reflect"
	"strings"
	"testing"

	authenticationv1 "k8s.io/api/authentication/v1"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

func TestCompileClaimsExpression(t *testing.T) {
	testCases := []struct {
		name                string
		expressionAccessors []ExpressionAccessor
	}{
		{
			name: "valid ClaimMappingCondition",
			expressionAccessors: []ExpressionAccessor{
				&ClaimMappingExpression{
					Expression: "claims.foo",
				},
			},
		},
		{
			name: "valid ClaimValidationCondition",
			expressionAccessors: []ExpressionAccessor{
				&ClaimValidationCondition{
					Expression: "claims.foo == 'bar'",
				},
			},
		},
		{
			name: "valid ExtraMapppingCondition",
			expressionAccessors: []ExpressionAccessor{
				&ExtraMappingExpression{
					Expression: "claims.foo",
				},
			},
		},
	}

	compiler := NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true))

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			for _, expressionAccessor := range tc.expressionAccessors {
				_, err := compiler.CompileClaimsExpression(expressionAccessor)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestCompileUserExpression(t *testing.T) {
	testCases := []struct {
		name                string
		expressionAccessors []ExpressionAccessor
	}{
		{
			name: "valid UserValidationCondition",
			expressionAccessors: []ExpressionAccessor{
				&ExtraMappingExpression{
					Expression: "user.username == 'bar'",
				},
			},
		},
	}

	compiler := NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true))

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			for _, expressionAccessor := range tc.expressionAccessors {
				_, err := compiler.CompileUserExpression(expressionAccessor)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestCompileClaimsExpressionError(t *testing.T) {
	testCases := []struct {
		name                string
		expressionAccessors []ExpressionAccessor
		wantErr             string
	}{
		{
			name: "invalid ClaimValidationCondition",
			expressionAccessors: []ExpressionAccessor{
				&ClaimValidationCondition{
					Expression: "claims.foo",
				},
			},
			wantErr: "must evaluate to bool",
		},
		{
			name: "UserValidationCondition with wrong env",
			expressionAccessors: []ExpressionAccessor{
				&UserValidationCondition{
					Expression: "user.username == 'foo'",
				},
			},
			wantErr: `compilation failed: ERROR: <input>:1:1: undeclared reference to 'user' (in container '')`,
		},
		{
			name: "invalid ClaimMappingCondition",
			expressionAccessors: []ExpressionAccessor{
				&ClaimMappingExpression{
					Expression: "claims + 1",
				},
			},
			wantErr: `compilation failed: ERROR: <input>:1:8: found no matching overload for '_+_' applied to '(map(string, any), int)'`,
		},
	}

	compiler := NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true))

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			for _, expressionAccessor := range tc.expressionAccessors {
				_, err := compiler.CompileClaimsExpression(expressionAccessor)
				if err == nil {
					t.Errorf("expected error but got nil")
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("expected error to contain %q but got %q", tc.wantErr, err.Error())
				}
			}
		})
	}
}

func TestCompileUserExpressionError(t *testing.T) {
	testCases := []struct {
		name                string
		expressionAccessors []ExpressionAccessor
		wantErr             string
	}{
		{
			name: "invalid UserValidationCondition",
			expressionAccessors: []ExpressionAccessor{
				&UserValidationCondition{
					Expression: "user.username",
				},
			},
			wantErr: "must evaluate to bool",
		},
		{
			name: "ClamMappingCondition with wrong env",
			expressionAccessors: []ExpressionAccessor{
				&ClaimMappingExpression{
					Expression: "claims.foo",
				},
			},
			wantErr: `compilation failed: ERROR: <input>:1:1: undeclared reference to 'claims' (in container '')`,
		},
		{
			name: "ExtraMappingCondition with wrong env",
			expressionAccessors: []ExpressionAccessor{
				&ExtraMappingExpression{
					Expression: "claims.foo",
				},
			},
			wantErr: `compilation failed: ERROR: <input>:1:1: undeclared reference to 'claims' (in container '')`,
		},
		{
			name: "ClaimValidationCondition with wrong env",
			expressionAccessors: []ExpressionAccessor{
				&ClaimValidationCondition{
					Expression: "claims.foo == 'bar'",
				},
			},
			wantErr: `compilation failed: ERROR: <input>:1:1: undeclared reference to 'claims' (in container '')`,
		},
		{
			name: "UserValidationCondition expression with unknown field",
			expressionAccessors: []ExpressionAccessor{
				&UserValidationCondition{
					Expression: "user.unknown == 'foo'",
				},
			},
			wantErr: `compilation failed: ERROR: <input>:1:5: undefined field 'unknown'`,
		},
	}

	compiler := NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true))

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			for _, expressionAccessor := range tc.expressionAccessors {
				_, err := compiler.CompileUserExpression(expressionAccessor)
				if err == nil {
					t.Errorf("expected error but got nil")
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("expected error to contain %q but got %q", tc.wantErr, err.Error())
				}
			}
		})
	}
}

func TestBuildUserType(t *testing.T) {
	userDeclType := buildUserType()
	userType := reflect.TypeOf(authenticationv1.UserInfo{})

	if len(userDeclType.Fields) != userType.NumField() {
		t.Errorf("expected %d fields, got %d", userType.NumField(), len(userDeclType.Fields))
	}

	for i := 0; i < userType.NumField(); i++ {
		field := userType.Field(i)
		jsonTagParts := strings.Split(field.Tag.Get("json"), ",")
		if len(jsonTagParts) < 1 {
			t.Fatal("expected json tag to be present")
		}
		fieldName := jsonTagParts[0]

		declField, ok := userDeclType.Fields[fieldName]
		if !ok {
			t.Errorf("expected field %q to be present", field.Name)
		}
		if nativeTypeToCELType(t, field.Type).CelType().Equal(declField.Type.CelType()).Value() != true {
			t.Errorf("expected field %q to have type %v, got %v", field.Name, field.Type, declField.Type)
		}
	}
}

func nativeTypeToCELType(t *testing.T, nativeType reflect.Type) *apiservercel.DeclType {
	switch nativeType {
	case reflect.TypeOf(""):
		return apiservercel.StringType
	case reflect.TypeOf([]string{}):
		return apiservercel.NewListType(apiservercel.StringType, -1)
	case reflect.TypeOf(map[string]authenticationv1.ExtraValue{}):
		return apiservercel.NewMapType(apiservercel.StringType, apiservercel.AnyType, -1)
	default:
		t.Fatalf("unsupported type %v", nativeType)
	}
	return nil
}
