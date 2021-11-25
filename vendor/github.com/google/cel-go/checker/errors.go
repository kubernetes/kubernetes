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

package checker

import (
	"github.com/google/cel-go/common"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// typeErrors is a specialization of Errors.
type typeErrors struct {
	*common.Errors
}

func (e *typeErrors) undeclaredReference(l common.Location, container string, name string) {
	e.ReportError(l, "undeclared reference to '%s' (in container '%s')", name, container)
}

func (e *typeErrors) expressionDoesNotSelectField(l common.Location) {
	e.ReportError(l, "expression does not select a field")
}

func (e *typeErrors) typeDoesNotSupportFieldSelection(l common.Location, t *exprpb.Type) {
	e.ReportError(l, "type '%s' does not support field selection", t)
}

func (e *typeErrors) undefinedField(l common.Location, field string) {
	e.ReportError(l, "undefined field '%s'", field)
}

func (e *typeErrors) noMatchingOverload(l common.Location, name string, args []*exprpb.Type, isInstance bool) {
	signature := formatFunction(nil, args, isInstance)
	e.ReportError(l, "found no matching overload for '%s' applied to '%s'", name, signature)
}

func (e *typeErrors) notAType(l common.Location, t *exprpb.Type) {
	e.ReportError(l, "'%s(%v)' is not a type", FormatCheckedType(t), t)
}

func (e *typeErrors) notAMessageType(l common.Location, t *exprpb.Type) {
	e.ReportError(l, "'%s' is not a message type", FormatCheckedType(t))
}

func (e *typeErrors) fieldTypeMismatch(l common.Location, name string, field *exprpb.Type, value *exprpb.Type) {
	e.ReportError(l, "expected type of field '%s' is '%s' but provided type is '%s'",
		name, FormatCheckedType(field), FormatCheckedType(value))
}

func (e *typeErrors) unexpectedFailedResolution(l common.Location, typeName string) {
	e.ReportError(l, "[internal] unexpected failed resolution of '%s'", typeName)
}

func (e *typeErrors) notAComprehensionRange(l common.Location, t *exprpb.Type) {
	e.ReportError(l, "expression of type '%s' cannot be range of a comprehension (must be list, map, or dynamic)",
		FormatCheckedType(t))
}

func (e *typeErrors) typeMismatch(l common.Location, expected *exprpb.Type, actual *exprpb.Type) {
	e.ReportError(l, "expected type '%s' but found '%s'",
		FormatCheckedType(expected), FormatCheckedType(actual))
}

func formatFunction(resultType *exprpb.Type, argTypes []*exprpb.Type, isInstance bool) string {
	result := ""
	if isInstance {
		target := argTypes[0]
		argTypes = argTypes[1:]

		result += FormatCheckedType(target)
		result += "."
	}

	result += "("
	for i, arg := range argTypes {
		if i > 0 {
			result += ", "
		}
		result += FormatCheckedType(arg)
	}
	result += ")"
	if resultType != nil {
		result += " -> "
		result += FormatCheckedType(resultType)
	}

	return result
}
