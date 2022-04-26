// (c) Copyright 2016 Hewlett Packard Enterprise Development LP
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rules

import (
	"fmt"
	"go/ast"

	"github.com/securego/gosec/v2"
)

type decompressionBombCheck struct {
	gosec.MetaData
	readerCalls gosec.CallList
	copyCalls   gosec.CallList
}

func (d *decompressionBombCheck) ID() string {
	return d.MetaData.ID
}

func containsReaderCall(node ast.Node, ctx *gosec.Context, list gosec.CallList) bool {
	if list.ContainsPkgCallExpr(node, ctx, false) != nil {
		return true
	}
	// Resolve type info of ident (for *archive/zip.File.Open)
	s, idt, _ := gosec.GetCallInfo(node, ctx)
	return list.Contains(s, idt)
}

func (d *decompressionBombCheck) Match(node ast.Node, ctx *gosec.Context) (*gosec.Issue, error) {
	var readerVarObj map[*ast.Object]struct{}

	// To check multiple lines, ctx.PassedValues is used to store temporary data.
	if _, ok := ctx.PassedValues[d.ID()]; !ok {
		readerVarObj = make(map[*ast.Object]struct{})
		ctx.PassedValues[d.ID()] = readerVarObj
	} else if pv, ok := ctx.PassedValues[d.ID()].(map[*ast.Object]struct{}); ok {
		readerVarObj = pv
	} else {
		return nil, fmt.Errorf("PassedValues[%s] of Context is not map[*ast.Object]struct{}, but %T", d.ID(), ctx.PassedValues[d.ID()])
	}

	// io.Copy is a common function.
	// To reduce false positives, This rule detects code which is used for compressed data only.
	switch n := node.(type) {
	case *ast.AssignStmt:
		for _, expr := range n.Rhs {
			if callExpr, ok := expr.(*ast.CallExpr); ok && containsReaderCall(callExpr, ctx, d.readerCalls) {
				if idt, ok := n.Lhs[0].(*ast.Ident); ok && idt.Name != "_" {
					// Example:
					//  r, _ := zlib.NewReader(buf)
					//  Add r's Obj to readerVarObj map
					readerVarObj[idt.Obj] = struct{}{}
				}
			}
		}
	case *ast.CallExpr:
		if d.copyCalls.ContainsPkgCallExpr(n, ctx, false) != nil {
			if idt, ok := n.Args[1].(*ast.Ident); ok {
				if _, ok := readerVarObj[idt.Obj]; ok {
					// Detect io.Copy(x, r)
					return gosec.NewIssue(ctx, n, d.ID(), d.What, d.Severity, d.Confidence), nil
				}
			}
		}
	}

	return nil, nil
}

// NewDecompressionBombCheck detects if there is potential DoS vulnerability via decompression bomb
func NewDecompressionBombCheck(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	readerCalls := gosec.NewCallList()
	readerCalls.Add("compress/gzip", "NewReader")
	readerCalls.AddAll("compress/zlib", "NewReader", "NewReaderDict")
	readerCalls.Add("compress/bzip2", "NewReader")
	readerCalls.AddAll("compress/flate", "NewReader", "NewReaderDict")
	readerCalls.Add("compress/lzw", "NewReader")
	readerCalls.Add("archive/tar", "NewReader")
	readerCalls.Add("archive/zip", "NewReader")
	readerCalls.Add("*archive/zip.File", "Open")

	copyCalls := gosec.NewCallList()
	copyCalls.Add("io", "Copy")
	copyCalls.Add("io", "CopyBuffer")

	return &decompressionBombCheck{
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.Medium,
			Confidence: gosec.Medium,
			What:       "Potential DoS vulnerability via decompression bomb",
		},
		readerCalls: readerCalls,
		copyCalls:   copyCalls,
	}, []ast.Node{(*ast.FuncDecl)(nil), (*ast.AssignStmt)(nil), (*ast.CallExpr)(nil)}
}
