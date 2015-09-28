/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

// DefaultPackage contains a default implentation of Package.
type DefaultPackage struct {
	PackageName string
	PackagePath string
	// emitted in every file.
	HeaderText []byte

	// emitted only for a "doc.go" file.
	PackageDocumentation []byte

	// If non-nil, will be called on "Generators"; otherwise, the static
	// list will be used.
	GeneratorFunc func(*Context) []Generator
	GeneratorList []Generator

	// Optional
	FilterFunc func(*Context, *types.Type) bool
}

func (d *DefaultPackage) Name() string { return d.PackageName }
func (d *DefaultPackage) Path() string { return d.PackagePath }

func (d *DefaultPackage) Filter(c *Context, t *types.Type) bool {
	if d.FilterFunc != nil {
		return d.FilterFunc(c, t)
	}
	return true
}

func (d *DefaultPackage) Generators(c *Context) []Generator {
	if d.GeneratorFunc != nil {
		return d.GeneratorFunc(c)
	}
	return d.GeneratorList
}

func (d *DefaultPackage) Header(filename string) []byte {
	if filename == "doc.go" {
		return append(d.HeaderText, d.PackageDocumentation...)
	}
	return d.HeaderText
}

var (
	_ = Package(&DefaultPackage{})
)
