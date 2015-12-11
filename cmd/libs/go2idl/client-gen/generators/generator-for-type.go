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

package generators

import (
	"io"
	"path/filepath"

	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

// genClientForType produces a file for each top-level type.
type genClientForType struct {
	generator.DefaultGen
	outputPackage string
	group         string
	typeToMatch   *types.Type
	imports       *generator.ImportTracker
}

// Filter ignores all but one type because we're making a single file per type.
func (g *genClientForType) Filter(c *generator.Context, t *types.Type) bool { return t == g.typeToMatch }

func (g *genClientForType) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *genClientForType) Imports(c *generator.Context) (imports []string) {
	return g.imports.ImportLines()
}

// GenerateType makes the body of a file implementing a set for type t.
func (g *genClientForType) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	pkg := filepath.Base(t.Name.Package)
	m := map[string]interface{}{
		"type":             t,
		"package":          pkg,
		"Package":          namer.IC(pkg),
		"Group":            namer.IC(g.group),
		"watchInterface":   c.Universe.Type(types.Name{Package: "k8s.io/kubernetes/pkg/watch", Name: "Interface"}),
		"apiDeleteOptions": c.Universe.Type(types.Name{Package: "k8s.io/kubernetes/pkg/api", Name: "DeleteOptions"}),
		"unvListOptions":   c.Universe.Type(types.Name{Package: "k8s.io/kubernetes/pkg/api/unversioned", Name: "ListOptions"}),
	}
	sw.Do(namespacerTemplate, m)
	sw.Do(interfaceTemplate, m)
	sw.Do(structTemplate, m)
	sw.Do(newStructTemplate, m)
	sw.Do(createTemplate, m)
	sw.Do(updateTemplate, m)
	sw.Do(deleteTemplate, m)
	sw.Do(getTemplate, m)
	sw.Do(listTemplate, m)
	sw.Do(watchTemplate, m)

	return sw.Error()
}

// template for namespacer
var namespacerTemplate = `
// $.type|public$Namespacer has methods to work with $.type|public$ resources in a namespace
type $.type|public$Namespacer interface {
	$.type|publicPlural$(namespace string) $.type|public$Interface
}
`

// template for the Interface
var interfaceTemplate = `
// $.type|public$Interface has methods to work with $.type|public$ resources.
type $.type|public$Interface interface {
	Create(*$.type|raw$) (*$.type|raw$, error)
	Update(*$.type|raw$) (*$.type|raw$, error)
	Delete(name string, options *$.apiDeleteOptions|raw$) error
	Get(name string) (*$.type|raw$, error)
	List(opts $.unvListOptions|raw$) (*$.type|raw$List, error)
	Watch(opts $.unvListOptions|raw$) ($.watchInterface|raw$, error)
}
`

// template for the struct that implements the interface
var structTemplate = `
// $.type|privatePlural$ implements $.type|public$Interface
type $.type|privatePlural$ struct {
	client *$.Group$Client
	ns     string
}
`
var newStructTemplate = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.Group$Client, namespace string) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		client: c,
		ns:     namespace,
	}
}
`
var listTemplate = `
// List takes label and field selectors, and returns the list of $.type|publicPlural$ that match those selectors.
func (c *$.type|privatePlural$) List(opts $.unvListOptions|raw$) (result *$.type|raw$List, err error) {
	result = &$.type|raw$List{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("$.type|privatePlural$").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}
`
var getTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.type|private$ object, and an error if there is any.
func (c *$.type|privatePlural$) Get(name string) (result *$.type|raw$, err error) {
	result = &$.type|raw${}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("$.type|privatePlural$").
		Name(name).
		Do().
		Into(result)
	return
}
`

var deleteTemplate = `
// Delete takes name of the $.type|private$ and deletes it. Returns an error if one occurs.
func (c *$.type|privatePlural$) Delete(name string, options *$.apiDeleteOptions|raw$) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("$.type|privatePlural$").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("$.type|privatePlural$").
		Name(name).
		Body(body).
		Do().
		Error()
}
`

var createTemplate = `
// Create takes the representation of a $.type|private$ and creates it.  Returns the server's representation of the $.type|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Create($.type|private$ *$.type|raw$) (result *$.type|raw$, err error) {
	result = &$.type|raw${}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("$.type|privatePlural$").
		Body($.type|private$).
		Do().
		Into(result)
	return
}
`

var updateTemplate = `
// Update takes the representation of a $.type|private$ and updates it. Returns the server's representation of the $.type|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Update($.type|private$ *$.type|raw$) (result *$.type|raw$, err error) {
	result = &$.type|raw${}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("$.type|privatePlural$").
		Name($.type|private$.Name).
		Body($.type|private$).
		Do().
		Into(result)
	return
}
`

var watchTemplate = `
// Watch returns a $.watchInterface|raw$ that watches the requested $.type|privatePlural$.
func (c *$.type|privatePlural$) Watch(opts $.unvListOptions|raw$) ($.watchInterface|raw$, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("$.type|privatePlural$").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
`
