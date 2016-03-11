/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package fake

import (
	"fmt"
	"io"
	"path/filepath"

	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/generators/normalization"
	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// genClientset generates a package for a clientset.
type genClientset struct {
	generator.DefaultGen
	groupVersions      []unversioned.GroupVersion
	typedClientPath    string
	outputPackage      string
	imports            *generator.ImportTracker
	clientsetGenerated bool
	// the import path of the generated real clientset.
	clientsetPath string
}

var _ generator.Generator = &genClientset{}

func (g *genClientset) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

// We only want to call GenerateType() once.
func (g *genClientset) Filter(c *generator.Context, t *types.Type) bool {
	ret := !g.clientsetGenerated
	g.clientsetGenerated = true
	return ret
}

func (g *genClientset) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	for _, gv := range g.groupVersions {
		group := normalization.Group(gv.Group)
		version := normalization.Version(gv.Version)
		typedClientPath := filepath.Join(g.typedClientPath, group, version)
		imports = append(imports, fmt.Sprintf("%s%s \"%s\"", version, group, typedClientPath))
		fakeTypedClientPath := filepath.Join(typedClientPath, "fake")
		imports = append(imports, fmt.Sprintf("fake%s%s \"%s\"", version, group, fakeTypedClientPath))
	}
	// the package that has the clientset Interface
	imports = append(imports, fmt.Sprintf("clientset \"%s\"", g.clientsetPath))
	// imports for the code in commonTemplate
	imports = append(imports,
		"k8s.io/kubernetes/pkg/api",
		"k8s.io/kubernetes/pkg/apimachinery/registered",
		"k8s.io/kubernetes/pkg/client/testing/core",
		"k8s.io/kubernetes/pkg/client/typed/discovery",
		"k8s.io/kubernetes/pkg/runtime",
		"k8s.io/kubernetes/pkg/watch",
	)

	return
}

func (g *genClientset) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	// TODO: We actually don't need any type information to generate the clientset,
	// perhaps we can adapt the go2ild framework to this kind of usage.
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	sw.Do(common, nil)

	sw.Do(checkImpl, nil)

	type arg struct {
		Group       string
		PackageName string
	}
	allGroups := []arg{}
	for _, gv := range g.groupVersions {
		group := normalization.Group(gv.Group)
		version := normalization.Version(gv.Version)
		allGroups = append(allGroups, arg{namer.IC(group), version + group})
	}

	for _, g := range allGroups {
		sw.Do(clientsetInterfaceImplTemplate, g)
	}

	return sw.Error()
}

// This part of code is version-independent, unchanging.
var common = `
// Clientset returns a clientset that will respond with the provided objects
func NewSimpleClientset(objects ...runtime.Object) *Clientset {
	o := core.NewObjects(api.Scheme, api.Codecs.UniversalDecoder())
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	fakePtr := core.Fake{}
	fakePtr.AddReactor("*", "*", core.ObjectReaction(o, registered.RESTMapper()))

	fakePtr.AddWatchReactor("*", core.DefaultWatchReactor(watch.NewFake(), nil))

	return &Clientset{fakePtr}
}

// Clientset implements clientset.Interface. Meant to be embedded into a
// struct to get a default implementation. This makes faking out just the method
// you want to test easier.
type Clientset struct {
	core.Fake
}

func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	return &FakeDiscovery{&c.Fake}
}
`

var checkImpl = `
var _ clientset.Interface = &Clientset{}
`

var clientsetInterfaceImplTemplate = `
// $.Group$ retrieves the $.Group$Client
func (c *Clientset) $.Group$() $.PackageName$.$.Group$Interface {
	return &fake$.PackageName$.Fake$.Group${&c.Fake}
}
`
