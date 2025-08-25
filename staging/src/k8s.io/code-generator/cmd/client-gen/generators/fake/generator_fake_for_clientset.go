/*
Copyright 2016 The Kubernetes Authors.

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
	"path"
	"strings"

	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
)

// genClientset generates a package for a clientset.
type genClientset struct {
	generator.GoGenerator
	groups               []clientgentypes.GroupVersions
	groupGoNames         map[clientgentypes.GroupVersion]string
	fakeClientsetPackage string // must be a Go import-path
	imports              namer.ImportTracker
	clientsetGenerated   bool
	// the import path of the generated real clientset.
	realClientsetPackage      string // must be a Go import-path
	applyConfigurationPackage string
}

var _ generator.Generator = &genClientset{}

func (g *genClientset) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.fakeClientsetPackage, g.imports),
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
	for _, group := range g.groups {
		for _, version := range group.Versions {
			groupClientPackage := path.Join(g.fakeClientsetPackage, "typed", strings.ToLower(group.PackageName), strings.ToLower(version.NonEmpty()))
			fakeGroupClientPackage := path.Join(groupClientPackage, "fake")

			groupAlias := strings.ToLower(g.groupGoNames[clientgentypes.GroupVersion{Group: group.Group, Version: version.Version}])
			imports = append(imports, fmt.Sprintf("%s%s \"%s\"", groupAlias, strings.ToLower(version.NonEmpty()), groupClientPackage))
			imports = append(imports, fmt.Sprintf("fake%s%s \"%s\"", groupAlias, strings.ToLower(version.NonEmpty()), fakeGroupClientPackage))
		}
	}
	// the package that has the clientset Interface
	imports = append(imports, fmt.Sprintf("clientset \"%s\"", g.realClientsetPackage))
	// imports for the code in commonTemplate
	imports = append(imports,
		"k8s.io/client-go/testing",
		"k8s.io/client-go/discovery",
		"fakediscovery \"k8s.io/client-go/discovery/fake\"",
		"k8s.io/apimachinery/pkg/runtime",
		"k8s.io/apimachinery/pkg/watch",
		"metav1 \"k8s.io/apimachinery/pkg/apis/meta/v1\"",
	)

	return
}

func (g *genClientset) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	generateApply := len(g.applyConfigurationPackage) > 0

	// TODO: We actually don't need any type information to generate the clientset,
	// perhaps we can adapt the go2ild framework to this kind of usage.
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	allGroups := clientgentypes.ToGroupVersionInfo(g.groups, g.groupGoNames)

	sw.Do(common, nil)

	if generateApply {
		sw.Do(managedFieldsClientset, map[string]any{
			"newTypeConverter": types.Ref(g.applyConfigurationPackage, "NewTypeConverter"),
		})
	}

	sw.Do(checkImpl, nil)

	for _, group := range allGroups {
		m := map[string]interface{}{
			"group":        group.Group,
			"version":      group.Version,
			"PackageAlias": group.PackageAlias,
			"GroupGoName":  group.GroupGoName,
			"Version":      namer.IC(group.Version.String()),
		}

		sw.Do(clientsetInterfaceImplTemplate, m)
	}

	return sw.Error()
}

// This part of code is version-independent, unchanging.

var managedFieldsClientset = `
// NewClientset returns a clientset that will respond with the provided objects.
// It's backed by a very simple object tracker that processes creates, updates and deletions as-is,
// without applying any validations and/or defaults. It shouldn't be considered a replacement
// for a real clientset and is mostly useful in simple unit tests.
func NewClientset(objects ...runtime.Object) *Clientset {
	o := testing.NewFieldManagedObjectTracker(
		scheme,
		codecs.UniversalDecoder(),
		$.newTypeConverter|raw$(scheme),
	)
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	cs := &Clientset{tracker: o}
	cs.discovery = &fakediscovery.FakeDiscovery{Fake: &cs.Fake}
	cs.AddReactor("*", "*", testing.ObjectReaction(o))
	cs.AddWatchReactor("*", func(action testing.Action) (handled bool, ret watch.Interface, err error) {
        var opts metav1.ListOptions
        if watchAction, ok := action.(testing.WatchActionImpl); ok {
            opts = watchAction.ListOptions
        }
		gvr := action.GetResource()
		ns := action.GetNamespace()
		watch, err := o.Watch(gvr, ns, opts)
		if err != nil {
			return false, nil, err
		}
		return true, watch, nil
	})

	return cs
}
`

var common = `
// NewSimpleClientset returns a clientset that will respond with the provided objects.
// It's backed by a very simple object tracker that processes creates, updates and deletions as-is,
// without applying any field management, validations and/or defaults. It shouldn't be considered a replacement
// for a real clientset and is mostly useful in simple unit tests.
//
// DEPRECATED: NewClientset replaces this with support for field management, which significantly improves
// server side apply testing. NewClientset is only available when apply configurations are generated (e.g.
// via --with-applyconfig).
func NewSimpleClientset(objects ...runtime.Object) *Clientset {
	o := testing.NewObjectTracker(scheme, codecs.UniversalDecoder())
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	cs := &Clientset{tracker: o}
	cs.discovery = &fakediscovery.FakeDiscovery{Fake: &cs.Fake}
	cs.AddReactor("*", "*", testing.ObjectReaction(o))
	cs.AddWatchReactor("*", func(action testing.Action) (handled bool, ret watch.Interface, err error) {
        var opts metav1.ListOptions
        if watchActcion, ok := action.(testing.WatchActionImpl); ok {
            opts = watchActcion.ListOptions
        }
		gvr := action.GetResource()
		ns := action.GetNamespace()
		watch, err := o.Watch(gvr, ns, opts)
		if err != nil {
			return false, nil, err
		}
		return true, watch, nil
	})

	return cs
}

// Clientset implements clientset.Interface. Meant to be embedded into a
// struct to get a default implementation. This makes faking out just the method
// you want to test easier.
type Clientset struct {
	testing.Fake
	discovery *fakediscovery.FakeDiscovery
	tracker testing.ObjectTracker
}

func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	return c.discovery
}

func (c *Clientset) Tracker() testing.ObjectTracker {
	return c.tracker
}
`

var checkImpl = `
var (
	_ clientset.Interface = &Clientset{}
	_ testing.FakeClient  = &Clientset{}
)
`

var clientsetInterfaceImplTemplate = `
// $.GroupGoName$$.Version$ retrieves the $.GroupGoName$$.Version$Client
func (c *Clientset) $.GroupGoName$$.Version$() $.PackageAlias$.$.GroupGoName$$.Version$Interface {
	return &fake$.PackageAlias$.Fake$.GroupGoName$$.Version${Fake: &c.Fake}
}
`
