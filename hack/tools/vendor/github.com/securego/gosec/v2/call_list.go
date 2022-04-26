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

package gosec

import (
	"go/ast"
	"strings"
)

const vendorPath = "vendor/"

type set map[string]bool

// CallList is used to check for usage of specific packages
// and functions.
type CallList map[string]set

// NewCallList creates a new empty CallList
func NewCallList() CallList {
	return make(CallList)
}

// AddAll will add several calls to the call list at once
func (c CallList) AddAll(selector string, idents ...string) {
	for _, ident := range idents {
		c.Add(selector, ident)
	}
}

// Add a selector and call to the call list
func (c CallList) Add(selector, ident string) {
	if _, ok := c[selector]; !ok {
		c[selector] = make(set)
	}
	c[selector][ident] = true
}

// Contains returns true if the package and function are
/// members of this call list.
func (c CallList) Contains(selector, ident string) bool {
	if idents, ok := c[selector]; ok {
		_, found := idents[ident]
		return found
	}
	return false
}

// ContainsPointer returns true if a pointer to the selector type or the type
// itself is a members of this call list.
func (c CallList) ContainsPointer(selector, indent string) bool {
	if strings.HasPrefix(selector, "*") {
		if c.Contains(selector, indent) {
			return true
		}
		s := strings.TrimPrefix(selector, "*")
		return c.Contains(s, indent)
	}
	return false
}

// ContainsPkgCallExpr resolves the call expression name and type, and then further looks
// up the package path for that type. Finally, it determines if the call exists within the call list
func (c CallList) ContainsPkgCallExpr(n ast.Node, ctx *Context, stripVendor bool) *ast.CallExpr {
	selector, ident, err := GetCallInfo(n, ctx)
	if err != nil {
		return nil
	}

	// Use only explicit path (optionally strip vendor path prefix) to reduce conflicts
	path, ok := GetImportPath(selector, ctx)
	if !ok {
		return nil
	}
	if stripVendor {
		if vendorIdx := strings.Index(path, vendorPath); vendorIdx >= 0 {
			path = path[vendorIdx+len(vendorPath):]
		}
	}
	if !c.Contains(path, ident) {
		return nil
	}

	return n.(*ast.CallExpr)
}

// ContainsCallExpr resolves the call expression name and type, and then determines
// if the call exists with the call list
func (c CallList) ContainsCallExpr(n ast.Node, ctx *Context) *ast.CallExpr {
	selector, ident, err := GetCallInfo(n, ctx)
	if err != nil {
		return nil
	}
	if !c.Contains(selector, ident) && !c.ContainsPointer(selector, ident) {
		return nil
	}

	return n.(*ast.CallExpr)
}
