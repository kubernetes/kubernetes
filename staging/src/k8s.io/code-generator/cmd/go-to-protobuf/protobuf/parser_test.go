/*
Copyright 2016 The Kubernetes Authors.

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

func TestProtoParser(t *testing.T) {
	ident := ast.NewIdent("FieldName")
	tests := []struct {
		expr ast.Expr
		err  bool
	}{
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
