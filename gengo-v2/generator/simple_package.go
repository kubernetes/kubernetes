/*
Copyright 2015 The Kubernetes Authors.

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

package generator

import (
	"k8s.io/gengo/v2/types"
)

// SimplePackage is implements Package in terms of static configuration.
// The package name, path, and dir are required to be non-empty.
type SimplePackage struct {
	// PkgName is the name of the resulting package (as in "package xxxx").
	// Required.
	PkgName string
	// PkgPath is the canonical Go import-path of the resulting package (as in
	// "import example.com/xxxx/yyyy"). Required.
	PkgPath string
	// PkgDir is the location of the resulting package on disk (which may not
	// exist yet). Required.
	PkgDir string

	// HeaderComment is emitted at the top of every output file. Optional.
	HeaderComment []byte

	// PkgDocComment is emitted after the header comment for a "doc.go" file.
	// Optional.
	PkgDocComment []byte

	// FilterFunc will be called to implement Package.Filter. Optional.
	FilterFunc func(*Context, *types.Type) bool

	// GeneratorsFunc will be called to implement Package.Generators. Optional.
	GeneratorsFunc func(*Context) []Generator
}

func (d SimplePackage) Name() string       { return d.PkgName }
func (d SimplePackage) Path() string       { return d.PkgPath }
func (d SimplePackage) SourcePath() string { return d.PkgDir }

func (d SimplePackage) Filter(c *Context, t *types.Type) bool {
	if d.FilterFunc != nil {
		return d.FilterFunc(c, t)
	}
	return true
}

func (d SimplePackage) Generators(c *Context) []Generator {
	if d.GeneratorsFunc != nil {
		return d.GeneratorsFunc(c)
	}
	return nil
}

func (d SimplePackage) Header(filename string) []byte {
	if filename == "doc.go" {
		return append(d.HeaderComment, d.PkgDocComment...)
	}
	return d.HeaderComment
}

var (
	_ = Package(SimplePackage{})
)
