// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cel

import (
	"fmt"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// CheckedExprToAst converts a checked expression proto message to an Ast.
func CheckedExprToAst(checkedExpr *exprpb.CheckedExpr) *Ast {
	return CheckedExprToAstWithSource(checkedExpr, nil)
}

// CheckedExprToAstWithSource converts a checked expression proto message to an Ast,
// using the provided Source as the textual contents.
//
// In general the source is not necessary unless the AST has been modified between the
// `Parse` and `Check` calls as an `Ast` created from the `Parse` step will carry the source
// through future calls.
//
// Prefer CheckedExprToAst if loading expressions from storage.
func CheckedExprToAstWithSource(checkedExpr *exprpb.CheckedExpr, src common.Source) *Ast {
	refMap := checkedExpr.GetReferenceMap()
	if refMap == nil {
		refMap = map[int64]*exprpb.Reference{}
	}
	typeMap := checkedExpr.GetTypeMap()
	if typeMap == nil {
		typeMap = map[int64]*exprpb.Type{}
	}
	si := checkedExpr.GetSourceInfo()
	if si == nil {
		si = &exprpb.SourceInfo{}
	}
	if src == nil {
		src = common.NewInfoSource(si)
	}
	return &Ast{
		expr:    checkedExpr.GetExpr(),
		info:    si,
		source:  src,
		refMap:  refMap,
		typeMap: typeMap,
	}
}

// AstToCheckedExpr converts an Ast to an protobuf CheckedExpr value.
//
// If the Ast.IsChecked() returns false, this conversion method will return an error.
func AstToCheckedExpr(a *Ast) (*exprpb.CheckedExpr, error) {
	if !a.IsChecked() {
		return nil, fmt.Errorf("cannot convert unchecked ast")
	}
	return &exprpb.CheckedExpr{
		Expr:         a.Expr(),
		SourceInfo:   a.SourceInfo(),
		ReferenceMap: a.refMap,
		TypeMap:      a.typeMap,
	}, nil
}

// ParsedExprToAst converts a parsed expression proto message to an Ast.
func ParsedExprToAst(parsedExpr *exprpb.ParsedExpr) *Ast {
	return ParsedExprToAstWithSource(parsedExpr, nil)
}

// ParsedExprToAstWithSource converts a parsed expression proto message to an Ast,
// using the provided Source as the textual contents.
//
// In general you only need this if you need to recheck a previously checked
// expression, or if you need to separately check a subset of an expression.
//
// Prefer ParsedExprToAst if loading expressions from storage.
func ParsedExprToAstWithSource(parsedExpr *exprpb.ParsedExpr, src common.Source) *Ast {
	si := parsedExpr.GetSourceInfo()
	if si == nil {
		si = &exprpb.SourceInfo{}
	}
	if src == nil {
		src = common.NewInfoSource(si)
	}
	return &Ast{
		expr:   parsedExpr.GetExpr(),
		info:   si,
		source: src,
	}
}

// AstToParsedExpr converts an Ast to an protobuf ParsedExpr value.
func AstToParsedExpr(a *Ast) (*exprpb.ParsedExpr, error) {
	return &exprpb.ParsedExpr{
		Expr:       a.Expr(),
		SourceInfo: a.SourceInfo(),
	}, nil
}

// AstToString converts an Ast back to a string if possible.
//
// Note, the conversion may not be an exact replica of the original expression, but will produce
// a string that is semantically equivalent and whose textual representation is stable.
func AstToString(a *Ast) (string, error) {
	expr := a.Expr()
	info := a.SourceInfo()
	return parser.Unparse(expr, info)
}
