/*
Copyright 2020 The Kubernetes Authors.

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

package protobuf

import (
	"go/ast"
	"testing"
)

/*
	struct fields in go AST:

     type Struct struct {
		// fields with a direct field Name as <Ident>
        A X  // regular fields
        B *X // pointer fields
        C    // embedded type field

        // qualified embedded type fields use an <SelExpr> in the AST
        v1.TypeMeta // X=v1, Sel=TypeMeta

        // fields without a direct name, but
        // a <StarExpr> in the go-AST
        *D   // type field embedded as pointer
        *v1.ListMeta   // qualified type field embedded as pointer
                       // with <StarExpr> pointing to <SelExpr>
     }
*/

func TestProtoParser(t *testing.T) {
	ident := ast.NewIdent("FieldName")
	tests := []struct {
		expr ast.Expr
		err  bool
	}{
		// valid struct field expressions
		{
			expr: ident,
			err:  false,
		},
		{
			expr: &ast.SelectorExpr{
				Sel: ident,
			},
			err: false,
		},
		{
			expr: &ast.StarExpr{
				X: ident,
			},
			err: false,
		},
		{
			expr: &ast.StarExpr{
				X: &ast.StarExpr{
					X: ident,
				},
			},
			err: false,
		},
		{
			expr: &ast.StarExpr{
				X: &ast.SelectorExpr{
					Sel: ident,
				},
			},
			err: false,
		},

		// something else should provide an error
		{
			expr: &ast.KeyValueExpr{
				Key:   ident,
				Colon: 0,
				Value: ident,
			},
			err: true,
		},
		{
			expr: &ast.StarExpr{
				X: &ast.KeyValueExpr{
					Key:   ident,
					Colon: 0,
					Value: ident,
				},
			},
			err: true,
		},
	}

	for _, test := range tests {
		actual, err := getFieldName(test.expr, "Struct")
		if !test.err {
			if err != nil {
				t.Errorf("%s: unexpected error %s", test.expr, err)
			} else {
				if actual != ident.Name {
					t.Errorf("%s: expected %s, got %s", test.expr, ident.Name, actual)
				}
			}
		} else {
			if err == nil {
				t.Errorf("%s: expected error did not occur, got %s instead", test.expr, actual)
			}
		}
	}
}
