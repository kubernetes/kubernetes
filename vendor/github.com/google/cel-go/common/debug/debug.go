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

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Adorner returns debug metadata that will be tacked on to the string
// representation of an expression.
type Adorner interface {
	// GetMetadata for the input context.
	GetMetadata(ctx interface{}) string
}

// Writer manages writing expressions to an internal string.
type Writer interface {
	fmt.Stringer

	// Buffer pushes an expression into an internal queue of expressions to
	// write to a string.
	Buffer(e *exprpb.Expr)
}

type emptyDebugAdorner struct {
}

var emptyAdorner Adorner = &emptyDebugAdorner{}

func (a *emptyDebugAdorner) GetMetadata(e interface{}) string {
	return ""
}

// ToDebugString gives the unadorned string representation of the Expr.
func ToDebugString(e *exprpb.Expr) string {
	return ToAdornedDebugString(e, emptyAdorner)
}

// ToAdornedDebugString gives the adorned string representation of the Expr.
func ToAdornedDebugString(e *exprpb.Expr, adorner Adorner) string {
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

func (w *debugWriter) Buffer(e *exprpb.Expr) {
	if e == nil {
		return
	}
	switch e.ExprKind.(type) {
	case *exprpb.Expr_ConstExpr:
		w.append(formatLiteral(e.GetConstExpr()))
	case *exprpb.Expr_IdentExpr:
		w.append(e.GetIdentExpr().Name)
	case *exprpb.Expr_SelectExpr:
		w.appendSelect(e.GetSelectExpr())
	case *exprpb.Expr_CallExpr:
		w.appendCall(e.GetCallExpr())
	case *exprpb.Expr_ListExpr:
		w.appendList(e.GetListExpr())
	case *exprpb.Expr_StructExpr:
		w.appendStruct(e.GetStructExpr())
	case *exprpb.Expr_ComprehensionExpr:
		w.appendComprehension(e.GetComprehensionExpr())
	}
	w.adorn(e)
}

func (w *debugWriter) appendSelect(sel *exprpb.Expr_Select) {
	w.Buffer(sel.GetOperand())
	w.append(".")
	w.append(sel.GetField())
	if sel.TestOnly {
		w.append("~test-only~")
	}
}

func (w *debugWriter) appendCall(call *exprpb.Expr_Call) {
	if call.Target != nil {
		w.Buffer(call.GetTarget())
		w.append(".")
	}
	w.append(call.GetFunction())
	w.append("(")
	if len(call.GetArgs()) > 0 {
		w.addIndent()
		w.appendLine()
		for i, arg := range call.GetArgs() {
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

func (w *debugWriter) appendList(list *exprpb.Expr_CreateList) {
	w.append("[")
	if len(list.GetElements()) > 0 {
		w.appendLine()
		w.addIndent()
		for i, elem := range list.GetElements() {
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

func (w *debugWriter) appendStruct(obj *exprpb.Expr_CreateStruct) {
	if obj.MessageName != "" {
		w.appendObject(obj)
	} else {
		w.appendMap(obj)
	}
}

func (w *debugWriter) appendObject(obj *exprpb.Expr_CreateStruct) {
	w.append(obj.GetMessageName())
	w.append("{")
	if len(obj.GetEntries()) > 0 {
		w.appendLine()
		w.addIndent()
		for i, entry := range obj.GetEntries() {
			if i > 0 {
				w.append(",")
				w.appendLine()
			}
			w.append(entry.GetFieldKey())
			w.append(":")
			w.Buffer(entry.GetValue())
			w.adorn(entry)
		}
		w.removeIndent()
		w.appendLine()
	}
	w.append("}")
}

func (w *debugWriter) appendMap(obj *exprpb.Expr_CreateStruct) {
	w.append("{")
	if len(obj.GetEntries()) > 0 {
		w.appendLine()
		w.addIndent()
		for i, entry := range obj.GetEntries() {
			if i > 0 {
				w.append(",")
				w.appendLine()
			}
			w.Buffer(entry.GetMapKey())
			w.append(":")
			w.Buffer(entry.GetValue())
			w.adorn(entry)
		}
		w.removeIndent()
		w.appendLine()
	}
	w.append("}")
}

func (w *debugWriter) appendComprehension(comprehension *exprpb.Expr_Comprehension) {
	w.append("__comprehension__(")
	w.addIndent()
	w.appendLine()
	w.append("// Variable")
	w.appendLine()
	w.append(comprehension.GetIterVar())
	w.append(",")
	w.appendLine()
	w.append("// Target")
	w.appendLine()
	w.Buffer(comprehension.GetIterRange())
	w.append(",")
	w.appendLine()
	w.append("// Accumulator")
	w.appendLine()
	w.append(comprehension.GetAccuVar())
	w.append(",")
	w.appendLine()
	w.append("// Init")
	w.appendLine()
	w.Buffer(comprehension.GetAccuInit())
	w.append(",")
	w.appendLine()
	w.append("// LoopCondition")
	w.appendLine()
	w.Buffer(comprehension.GetLoopCondition())
	w.append(",")
	w.appendLine()
	w.append("// LoopStep")
	w.appendLine()
	w.Buffer(comprehension.GetLoopStep())
	w.append(",")
	w.appendLine()
	w.append("// Result")
	w.appendLine()
	w.Buffer(comprehension.GetResult())
	w.append(")")
	w.removeIndent()
}

func formatLiteral(c *exprpb.Constant) string {
	switch c.GetConstantKind().(type) {
	case *exprpb.Constant_BoolValue:
		return fmt.Sprintf("%t", c.GetBoolValue())
	case *exprpb.Constant_BytesValue:
		return fmt.Sprintf("b\"%s\"", string(c.GetBytesValue()))
	case *exprpb.Constant_DoubleValue:
		return fmt.Sprintf("%v", c.GetDoubleValue())
	case *exprpb.Constant_Int64Value:
		return fmt.Sprintf("%d", c.GetInt64Value())
	case *exprpb.Constant_StringValue:
		return strconv.Quote(c.GetStringValue())
	case *exprpb.Constant_Uint64Value:
		return fmt.Sprintf("%du", c.GetUint64Value())
	case *exprpb.Constant_NullValue:
		return "null"
	default:
		panic("Unknown constant type")
	}
}

func (w *debugWriter) append(s string) {
	w.doIndent()
	w.buffer.WriteString(s)
}

func (w *debugWriter) appendFormat(f string, args ...interface{}) {
	w.append(fmt.Sprintf(f, args...))
}

func (w *debugWriter) doIndent() {
	if w.lineStart {
		w.lineStart = false
		w.buffer.WriteString(strings.Repeat("  ", w.indent))
	}
}

func (w *debugWriter) adorn(e interface{}) {
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
