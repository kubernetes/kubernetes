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

// SimpleTarget is implements Target in terms of static configuration.
// The package name, path, and dir are required to be non-empty.
type SimpleTarget struct {
	// PkgName is the name of the resulting package (as in "package xxxx").
	// Required.
	PkgName string
	// PkgPath is the canonical Go import-path of the resulting package (as in
	// "import example.com/xxxx/yyyy"). Required.
	PkgPath string
	// PkgDir is the location of the resulting package on disk (which may not
	// exist yet). It may be absolute or relative to CWD. Required.
	PkgDir string

	// HeaderComment is emitted at the top of every output file. Optional.
	HeaderComment []byte

	// PkgDocComment is emitted after the header comment for a "doc.go" file.
	// Optional.
	PkgDocComment []byte

	// FilterFunc will be called to implement Target.Filter. Optional.
	FilterFunc func(*Context, *types.Type) bool

	// GeneratorsFunc will be called to implement Target.Generators. Optional.
	GeneratorsFunc func(*Context) []Generator
}

func (st SimpleTarget) Name() string { return st.PkgName }
func (st SimpleTarget) Path() string { return st.PkgPath }
func (st SimpleTarget) Dir() string  { return st.PkgDir }

func (st SimpleTarget) Filter(c *Context, t *types.Type) bool {
	if st.FilterFunc != nil {
		return st.FilterFunc(c, t)
	}
	return true
}

func (st SimpleTarget) Generators(c *Context) []Generator {
	if st.GeneratorsFunc != nil {
		return st.GeneratorsFunc(c)
	}
	return nil
}

func (st SimpleTarget) Header(filename string) []byte {
	if filename == "doc.go" {
		return append(st.HeaderComment, st.PkgDocComment...)
	}
	return st.HeaderComment
}

var (
	_ = Target(SimpleTarget{})
)
