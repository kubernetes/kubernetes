// Copyright 2018 Google LLC
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

// Package debug provides tools to print a parsed expression graph and
// adorn each expression element with additional metadata.
package debug

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"

	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// Adorner returns debug metadata that will be tacked on to the string
// representation of an expression.
type Adorner interface {
	// GetMetadata for the input context.
	GetMetadata(ctx any) string
}

// Writer manages writing expressions to an internal string.
type Writer interface {
	fmt.Stringer

	// Buffer pushes an expression into an internal queue of expressions to
	// write to a string.
	Buffer(e ast.Expr)
}

type emptyDebugAdorner struct {
}

var emptyAdorner Adorner = &emptyDebugAdorner{}

func (a *emptyDebugAdorner) GetMetadata(e any) string {
	return ""
}

// ToDebugString gives the unadorned string representation of the Expr.
func ToDebugString(e ast.Expr) string {
	return ToAdornedDebugString(e, emptyAdorner)
}

// ToAdornedDebugString gives the adorned string representation of the Expr.
func ToAdornedDebugString(e ast.Expr, adorner Adorner) string {
	w := newDebugWriter(adorner)
	w.Buffer(e)
	return w.String()
}

// debugWriter is used to print out pretty-printed debug strings.
type debugWriter struct {
	adorner   Adorner
	buffer    bytes.Buffer
	indent    int
	lineStart bool
}

func newDebugWriter(a Adorner) *debugWriter {
	return &debugWriter{
		adorner:   a,
		indent:    0,
		lineStart: true,
	}
}

func (w *debugWriter) Buffer(e ast.Expr) {
	if e == nil {
		return
	}
	switch e.Kind() {
	case ast.LiteralKind:
		w.append(formatLiteral(e.AsLiteral()))
	case ast.IdentKind:
		w.append(e.AsIdent())
	case ast.SelectKind:
		w.appendSelect(e.AsSelect())
	case ast.CallKind:
		w.appendCall(e.AsCall())
	case ast.ListKind:
		w.appendList(e.AsList())
	case ast.MapKind:
		w.appendMap(e.AsMap())
	case ast.StructKind:
		w.appendStruct(e.AsStruct())
	case ast.ComprehensionKind:
		w.appendComprehension(e.AsComprehension())
	}
	w.adorn(e)
}

func (w *debugWriter) appendSelect(sel ast.SelectExpr) {
	w.Buffer(sel.Operand())
	w.append(".")
	w.append(sel.FieldName())
	if sel.IsTestOnly() {
		w.append("~test-only~")
	}
}

func (w *debugWriter) appendCall(call ast.CallExpr) {
	if call.IsMemberFunction() {
		w.Buffer(call.Target())
		w.append(".")
	}
	w.append(call.FunctionName())
	w.append("(")
	if len(call.Args()) > 0 {
		w.addIndent()
		w.appendLine()
		for i, arg := range call.Args() {
			if i > 0 {
				w.append(",")
				w.appendLine()
			}
			w.Buffer(arg)
		}
		w.removeIndent()
		w.appendLine()
	}
	w.append(")")
}

func (w *debugWriter) appendList(list ast.ListExpr) {
	w.append("[")
	if len(list.Elements()) > 0 {
		w.appendLine()
		w.addIndent()
		for i, elem := range list.Elements() {
			if i > 0 {
				w.append(",")
				w.appendLine()
			}
			w.Buffer(elem)
		}
		w.removeIndent()
		w.appendLine()
	}
	w.append("]")
}

func (w *debugWriter) appendStruct(obj ast.StructExpr) {
	w.append(obj.TypeName())
	w.append("{")
	if len(obj.Fields()) > 0 {
		w.appendLine()
		w.addIndent()
		for i, f := range obj.Fields() {
			field := f.AsStructField()
			if i > 0 {
				w.append(",")
				w.appendLine()
			}
			if field.IsOptional() {
				w.append("?")
			}
			w.append(field.Name())
			w.append(":")
			w.Buffer(field.Value())
			w.adorn(f)
		}
		w.removeIndent()
		w.appendLine()
	}
	w.append("}")
}

func (w *debugWriter) appendMap(m ast.MapExpr) {
	w.append("{")
	if m.Size() > 0 {
		w.appendLine()
		w.addIndent()
		for i, e := range m.Entries() {
			entry := e.AsMapEntry()
			if i > 0 {
				w.append(",")
				w.appendLine()
			}
			if entry.IsOptional() {
				w.append("?")
			}
			w.Buffer(entry.Key())
			w.append(":")
			w.Buffer(entry.Value())
			w.adorn(e)
		}
		w.removeIndent()
		w.appendLine()
	}
	w.append("}")
}

func (w *debugWriter) appendComprehension(comprehension ast.ComprehensionExpr) {
	w.append("__comprehension__(")
	w.addIndent()
	w.appendLine()
	w.append("// Variable")
	w.appendLine()
	w.append(comprehension.IterVar())
	w.append(",")
	w.appendLine()
	if comprehension.HasIterVar2() {
		w.append(comprehension.IterVar2())
		w.append(",")
		w.appendLine()
	}
	w.append("// Target")
	w.appendLine()
	w.Buffer(comprehension.IterRange())
	w.append(",")
	w.appendLine()
	w.append("// Accumulator")
	w.appendLine()
	w.append(comprehension.AccuVar())
	w.append(",")
	w.appendLine()
	w.append("// Init")
	w.appendLine()
	w.Buffer(comprehension.AccuInit())
	w.append(",")
	w.appendLine()
	w.append("// LoopCondition")
	w.appendLine()
	w.Buffer(comprehension.LoopCondition())
	w.append(",")
	w.appendLine()
	w.append("// LoopStep")
	w.appendLine()
	w.Buffer(comprehension.LoopStep())
	w.append(",")
	w.appendLine()
	w.append("// Result")
	w.appendLine()
	w.Buffer(comprehension.Result())
	w.append(")")
	w.removeIndent()
}

func formatLiteral(c ref.Val) string {
	switch v := c.(type) {
	case types.Bool:
		return fmt.Sprintf("%t", v)
	case types.Bytes:
		return fmt.Sprintf("b%s", strconv.Quote(string(v)))
	case types.Double:
		return fmt.Sprintf("%v", float64(v))
	case types.Int:
		return fmt.Sprintf("%d", int64(v))
	case types.String:
		return strconv.Quote(string(v))
	case types.Uint:
		return fmt.Sprintf("%du", uint64(v))
	case types.Null:
		return "null"
	default:
		panic("Unknown constant type")
	}
}

func (w *debugWriter) append(s string) {
	w.doIndent()
	w.buffer.WriteString(s)
}

func (w *debugWriter) appendFormat(f string, args ...any) {
	w.append(fmt.Sprintf(f, args...))
}

func (w *debugWriter) doIndent() {
	if w.lineStart {
		w.lineStart = false
		w.buffer.WriteString(strings.Repeat("  ", w.indent))
	}
}

func (w *debugWriter) adorn(e any) {
	w.append(w.adorner.GetMetadata(e))
}

func (w *debugWriter) appendLine() {
	w.buffer.WriteString("\n")
	w.lineStart = true
}

func (w *debugWriter) addIndent() {
	w.indent++
}

func (w *debugWriter) removeIndent() {
	w.indent--
	if w.indent < 0 {
		panic("negative indent")
	}
}

func (w *debugWriter) String() string {
	return w.buffer.String()
}
