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
	"go/types"
	"strings"
)

// ImportTracker is used to normalize the packages that have been imported
// by a source file. It is able to differentiate between plain imports, aliased
// imports and init only imports.
type ImportTracker struct {
	Imported map[string]string
	Aliased  map[string]string
	InitOnly map[string]bool
}

// NewImportTracker creates an empty Import tracker instance
func NewImportTracker() *ImportTracker {
	return &ImportTracker{
		make(map[string]string),
		make(map[string]string),
		make(map[string]bool),
	}
}

// TrackFile track all the imports used by the supplied file
func (t *ImportTracker) TrackFile(file *ast.File) {
	for _, imp := range file.Imports {
		path := strings.Trim(imp.Path.Value, `"`)
		parts := strings.Split(path, "/")
		if len(parts) > 0 {
			name := parts[len(parts)-1]
			t.Imported[path] = name
		}
	}
}

// TrackPackages tracks all the imports used by the supplied packages
func (t *ImportTracker) TrackPackages(pkgs ...*types.Package) {
	for _, pkg := range pkgs {
		t.Imported[pkg.Path()] = pkg.Name()
	}
}

// TrackImport tracks imports and handles the 'unsafe' import
func (t *ImportTracker) TrackImport(n ast.Node) {
	if imported, ok := n.(*ast.ImportSpec); ok {
		path := strings.Trim(imported.Path.Value, `"`)
		if imported.Name != nil {
			if imported.Name.Name == "_" {
				// Initialization only import
				t.InitOnly[path] = true
			} else {
				// Aliased import
				t.Aliased[path] = imported.Name.Name
			}
		}
		if path == "unsafe" {
			t.Imported[path] = path
		}
	}
}
