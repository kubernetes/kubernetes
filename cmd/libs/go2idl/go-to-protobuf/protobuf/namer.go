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

package protobuf

import (
	"fmt"
	"reflect"
	"strings"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

type localNamer struct {
	localPackage types.Name
}

func (n localNamer) Name(t *types.Type) string {
	if t.Key != nil && t.Elem != nil {
		return fmt.Sprintf("map<%s, %s>", n.Name(t.Key), n.Name(t.Elem))
	}
	if len(n.localPackage.Package) != 0 && n.localPackage.Package == t.Name.Package {
		return t.Name.Name
	}
	return t.Name.String()
}

type protobufNamer struct {
	packages       []*protobufPackage
	packagesByPath map[string]*protobufPackage
}

func NewProtobufNamer() *protobufNamer {
	return &protobufNamer{
		packagesByPath: make(map[string]*protobufPackage),
	}
}

func (n *protobufNamer) Name(t *types.Type) string {
	if t.Kind == types.Map {
		return fmt.Sprintf("map<%s, %s>", n.Name(t.Key), n.Name(t.Elem))
	}
	return t.Name.String()
}

func (n *protobufNamer) List() []generator.Package {
	packages := make([]generator.Package, 0, len(n.packages))
	for i := range n.packages {
		packages = append(packages, n.packages[i])
	}
	return packages
}

func (n *protobufNamer) Add(p *protobufPackage) {
	if _, ok := n.packagesByPath[p.PackagePath]; !ok {
		n.packagesByPath[p.PackagePath] = p
		n.packages = append(n.packages, p)
	}
}

func (n *protobufNamer) GoNameToProtoName(name types.Name) types.Name {
	if p, ok := n.packagesByPath[name.Package]; ok {
		return types.Name{
			Name:    name.Name,
			Package: p.PackageName,
			Path:    p.ImportPath(),
		}
	}
	for _, p := range n.packages {
		if _, ok := p.FilterTypes[name]; ok {
			return types.Name{
				Name:    name.Name,
				Package: p.PackageName,
				Path:    p.ImportPath(),
			}
		}
	}
	return types.Name{Name: name.Name}
}

func protoSafePackage(name string) string {
	pkg := strings.Replace(name, "/", ".", -1)
	return strings.Replace(pkg, "-", "_", -1)
}

type typeNameSet map[types.Name]*protobufPackage

// assignGoTypeToProtoPackage looks for Go and Protobuf types that are referenced by a type in
// a package. It will not recurse into protobuf types.
func assignGoTypeToProtoPackage(p *protobufPackage, t *types.Type, local, global typeNameSet, optional map[types.Name]struct{}) {
	newT, isProto := isFundamentalProtoType(t)
	if isProto {
		t = newT
	}
	if otherP, ok := global[t.Name]; ok {
		if _, ok := local[t.Name]; !ok {
			p.Imports.AddType(&types.Type{
				Kind: types.Protobuf,
				Name: otherP.ProtoTypeName(),
			})
		}
		return
	}
	global[t.Name] = p
	if _, ok := local[t.Name]; ok {
		return
	}
	// don't recurse into existing proto types
	if isProto {
		p.Imports.AddType(t)
		return
	}

	local[t.Name] = p
	for _, m := range t.Members {
		if namer.IsPrivateGoName(m.Name) {
			continue
		}
		field := &protoField{}
		tag := reflect.StructTag(m.Tags).Get("protobuf")
		if tag == "-" {
			continue
		}
		if err := protobufTagToField(tag, field, m, t, p.ProtoTypeName()); err == nil && field.Type != nil {
			assignGoTypeToProtoPackage(p, field.Type, local, global, optional)
			continue
		}
		assignGoTypeToProtoPackage(p, m.Type, local, global, optional)
	}
	// TODO: should methods be walked?
	if t.Elem != nil {
		assignGoTypeToProtoPackage(p, t.Elem, local, global, optional)
	}
	if t.Key != nil {
		assignGoTypeToProtoPackage(p, t.Key, local, global, optional)
	}
	if t.Underlying != nil {
		if t.Kind == types.Alias && isOptionalAlias(t) {
			optional[t.Name] = struct{}{}
		}
		assignGoTypeToProtoPackage(p, t.Underlying, local, global, optional)
	}
}

func (n *protobufNamer) AssignTypesToPackages(c *generator.Context) error {
	global := make(typeNameSet)
	for _, p := range n.packages {
		local := make(typeNameSet)
		optional := make(map[types.Name]struct{})
		p.Imports = NewImportTracker(p.ProtoTypeName())
		for _, t := range c.Order {
			if t.Name.Package != p.PackagePath {
				continue
			}
			assignGoTypeToProtoPackage(p, t, local, global, optional)
		}
		p.FilterTypes = make(map[types.Name]struct{})
		p.LocalNames = make(map[string]struct{})
		p.OptionalTypeNames = make(map[string]struct{})
		for k, v := range local {
			if v == p {
				p.FilterTypes[k] = struct{}{}
				p.LocalNames[k.Name] = struct{}{}
				if _, ok := optional[k]; ok {
					p.OptionalTypeNames[k.Name] = struct{}{}
				}
			}
		}
	}
	return nil
}
