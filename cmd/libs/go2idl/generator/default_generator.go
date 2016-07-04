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

	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

const (
	GolangFileType = "golang"
)

// DefaultGen implements a do-nothing Generator.
//
// It can be used to implement static content files.
type DefaultGen struct {
	// OptionalName, if present, will be used for the generator's name, and
	// the filename (with ".go" appended).
	OptionalName string

	// OptionalBody, if present, will be used as the return from the "Init"
	// method. This causes it to be static content for the entire file if
	// no other generator touches the file.
	OptionalBody []byte
}

func (d DefaultGen) Name() string                                        { return d.OptionalName }
func (d DefaultGen) Filter(*Context, *types.Type) bool                   { return true }
func (d DefaultGen) Namers(*Context) namer.NameSystems                   { return nil }
func (d DefaultGen) Imports(*Context) []string                           { return []string{} }
func (d DefaultGen) PackageVars(*Context) []string                       { return []string{} }
func (d DefaultGen) PackageConsts(*Context) []string                     { return []string{} }
func (d DefaultGen) GenerateType(*Context, *types.Type, io.Writer) error { return nil }
func (d DefaultGen) Filename() string                                    { return d.OptionalName + ".go" }
func (d DefaultGen) FileType() string                                    { return GolangFileType }

func (d DefaultGen) Init(c *Context, w io.Writer) error {
	_, err := w.Write(d.OptionalBody)
	return err
}

var (
	_ = Generator(DefaultGen{})
)
