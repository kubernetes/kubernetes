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
	"io"

	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
)

const (
	GolangFileType = "golang"
)

// GolangGenerator implements a do-nothing Generator for Go files.  It can be
// used as a base for custom Generators, which embed it and then define the
// methods they need to specialize.
type GolangGenerator struct {
	// OutputFilename is used as the Generator's name, and filename.
	OutputFilename string

	// Body, if present, will be used as the return from the "Init" method.
	// This causes it to be static content for the entire file if no other
	// generator touches the file.
	OptionalBody []byte
}

func (gg GolangGenerator) Name() string                                        { return gg.OutputFilename }
func (gg GolangGenerator) Filter(*Context, *types.Type) bool                   { return true }
func (gg GolangGenerator) Namers(*Context) namer.NameSystems                   { return nil }
func (gg GolangGenerator) Imports(*Context) []string                           { return []string{} }
func (gg GolangGenerator) PackageVars(*Context) []string                       { return []string{} }
func (gg GolangGenerator) PackageConsts(*Context) []string                     { return []string{} }
func (gg GolangGenerator) GenerateType(*Context, *types.Type, io.Writer) error { return nil }
func (gg GolangGenerator) Filename() string                                    { return gg.OutputFilename }
func (gg GolangGenerator) FileType() string                                    { return GolangFileType }
func (gg GolangGenerator) Finalize(*Context, io.Writer) error                  { return nil }

func (gg GolangGenerator) Init(c *Context, w io.Writer) error {
	_, err := w.Write(gg.OptionalBody)
	return err
}

var (
	_ = Generator(GolangGenerator{})
)
