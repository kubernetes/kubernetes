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
	"fmt"
	"strings"

	"cel.dev/cel-go/common"
	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/env"
	"cel.dev/cel-go/common/types"
)

// typeErrors is a specialization of Errors.
type typeErrors struct {
	errs *common.Errors
	env  *Env
}

func (e *typeErrors) getCatalog() *env.Catalog {
	if e == nil || e.env == nil {
		return nil
	}
	return e.env.Catalog()
}

func (e *typeErrors) fieldTypeMismatch(id int64, l common.Location, name string, field, value *types.Type) {
	e.errs.ReportErrorAtID(id, l, "expected type of field '%s' is '%s' but provided type is '%s'",
		name, FormatCELType(field), FormatCELType(value))
}

func (e *typeErrors) incompatibleType(id int64, l common.Location, ex ast.Expr, prev, next *types.Type) {
	e.errs.ReportErrorAtID(id, l,
		"incompatible type already exists for expression: %v(%d) old:%v, new:%v", ex, ex.ID(), prev, next)
}

func (e *typeErrors) noMatchingOverload(id int64, l common.Location, name string, args []*types.Type, isInstance bool) {
	signature := formatFunctionDeclType(nil, args, isInstance)
	e.errs.ReportErrorAtID(id, l, "found no matching overload for '%s' applied to '%s'", name, signature)
}

func (e *typeErrors) notAComprehensionRange(id int64, l common.Location, t *types.Type) {
	e.errs.ReportErrorAtID(id, l, "expression of type '%s' cannot be range of a comprehension (must be list, map, or dynamic)",
		FormatCELType(t))
}

func (e *typeErrors) notAnOptionalFieldSelectionCall(id int64, l common.Location, err string) {
	e.errs.ReportErrorAtID(id, l, "unsupported optional field selection: %s", err)
}

func (e *typeErrors) notAnOptionalFieldSelection(id int64, l common.Location, field ast.Expr) {
	e.errs.ReportErrorAtID(id, l, "unsupported optional field selection: %v", field)
}

func (e *typeErrors) notAType(id int64, l common.Location, typeName string) {
	suggestion := ""
	if cat := e.getCatalog(); cat != nil {
		suggestion = formatSuggestionSuffix(typeName, cat.Find(typeName))
	}
	e.errs.ReportErrorAtID(id, l, "'%s' is not a type%s", typeName, suggestion)
}

func (e *typeErrors) notAMessageType(id int64, l common.Location, typeName string) {
	e.errs.ReportErrorAtID(id, l, "'%s' is not a message type", typeName)
}

func (e *typeErrors) referenceRedefinition(id int64, l common.Location, ex ast.Expr, prev, next *ast.ReferenceInfo) {
	e.errs.ReportErrorAtID(id, l,
		"reference already exists for expression: %v(%d) old:%v, new:%v", ex, ex.ID(), prev, next)
}

func (e *typeErrors) typeDoesNotSupportFieldSelection(id int64, l common.Location, t *types.Type) {
	e.errs.ReportErrorAtID(id, l, "type '%s' does not support field selection", FormatCELType(t))
}

func (e *typeErrors) typeMismatch(id int64, l common.Location, expected, actual *types.Type) {
	e.errs.ReportErrorAtID(id, l, "expected type '%s' but found '%s'",
		FormatCELType(expected), FormatCELType(actual))
}

func (e *typeErrors) undefinedField(id int64, l common.Location, field string) {
	e.errs.ReportErrorAtID(id, l, "undefined field '%s'", field)
}

func (e *typeErrors) undeclaredReference(id int64, l common.Location, container string, name string) {
	var syms []*env.CatalogSymbol
	if cat := e.getCatalog(); cat != nil {
		syms = cat.Find(name)
	}
	e.undeclaredReferenceWithSymbols(id, l, container, name, syms)
}

func (e *typeErrors) undeclaredReferenceWithSymbols(id int64, l common.Location, container string, name string, syms []*env.CatalogSymbol) {
	containerSuffix := ""
	if container != "" {
		containerSuffix = fmt.Sprintf(" (in container '%s')", container)
	}
	suggestion := formatSuggestionSuffix(name, syms)
	e.errs.ReportErrorAtID(id, l, "undeclared reference to '%s'%s%s", name, containerSuffix, suggestion)
}

func (e *typeErrors) checkUndeclaredIdent(env *Env, id int64, l common.Location, qualifiers ...string) bool {
	if len(qualifiers) <= 1 || e.hasDeclaredPrefix(env, qualifiers[:len(qualifiers)-1]...) {
		return false
	}
	return e.checkUndeclaredReference(env, id, l, strings.Join(qualifiers, "."))
}

func (e *typeErrors) checkUndeclaredFunction(env *Env, id int64, l common.Location, qualifiedPrefix, fnName string) bool {
	prefixParts := strings.Split(qualifiedPrefix, ".")
	if e.hasDeclaredPrefix(env, prefixParts...) {
		return false
	}
	return e.checkUndeclaredReference(env, id, l, qualifiedPrefix+"."+fnName)
}

func (e *typeErrors) checkUndeclaredReference(env *Env, id int64, l common.Location, qualifiedName string) bool {
	if env == nil {
		return false
	}
	cat := env.Catalog()
	if cat == nil {
		return false
	}
	syms := cat.Find(qualifiedName)
	if len(syms) == 0 {
		return false
	}
	container := ""
	if env.container != nil {
		container = env.container.Name()
	}
	e.undeclaredReferenceWithSymbols(id, l, container, qualifiedName, syms)
	return true
}

func (e *typeErrors) hasDeclaredPrefix(env *Env, qualifierPrefixes ...string) bool {
	if env == nil {
		return false
	}
	for i := 1; i <= len(qualifierPrefixes); i++ {
		if env.resolveQualifiedIdent(qualifierPrefixes[:i]...) != nil {
			return true
		}
	}
	return false
}

func formatSuggestionSuffix(name string, syms []*env.CatalogSymbol) string {
	if len(syms) == 0 {
		return ""
	}
	if len(syms) == 1 && syms[0].Name == name {
		if syms[0].Option != "" {
			return fmt.Sprintf(" (enable with `%s`)", syms[0].Option)
		}
		return ""
	}
	if len(syms) == 1 {
		if syms[0].Option != "" {
			return fmt.Sprintf(" (did you mean '%s'?, enable with `%s`)", syms[0].Name, syms[0].Option)
		}
		return fmt.Sprintf(" (did you mean '%s'?)", syms[0].Name)
	}
	return fmt.Sprintf(" (did you mean '%s' or '%s'?)", syms[0].Name, syms[1].Name)
}

func (e *typeErrors) unexpectedFailedResolution(id int64, l common.Location, typeName string) {
	e.errs.ReportErrorAtID(id, l, "unexpected failed resolution of '%s'", typeName)
}

func (e *typeErrors) unexpectedASTType(id int64, l common.Location, kind, typeName string) {
	e.errs.ReportErrorAtID(id, l, "unexpected %s type: %v", kind, typeName)
}
