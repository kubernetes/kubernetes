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

package generators

import (
	"io"
	"path/filepath"
	"strings"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"

	"k8s.io/kube-gen/cmd/client-gen/generators/util"
)

// genClientForType produces a file for each top-level type.
type genClientForType struct {
	generator.DefaultGen
	outputPackage    string
	clientsetPackage string
	group            string
	version          string
	typeToMatch      *types.Type
	imports          namer.ImportTracker
}

var _ generator.Generator = &genClientForType{}

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

// Ideally, we'd like genStatus to return true if there is a subresource path
// registered for "status" in the API server, but we do not have that
// information, so genStatus returns true if the type has a status field.
func genStatus(t *types.Type) bool {
	// Default to true if we have a Status member
	hasStatus := false
	for _, m := range t.Members {
		if m.Name == "Status" {
			hasStatus = true
			break
		}
	}
	return hasStatus && !util.MustParseClientGenTags(t.SecondClosestCommentLines).NoStatus
}

// GenerateType makes the body of a file implementing the individual typed client for type t.
func (g *genClientForType) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	pkg := filepath.Base(t.Name.Package)
	tags, err := util.ParseClientGenTags(t.SecondClosestCommentLines)
	if err != nil {
		return err
	}
	m := map[string]interface{}{
		"type":                 t,
		"package":              pkg,
		"Package":              namer.IC(pkg),
		"namespaced":           !tags.NonNamespaced,
		"Group":                namer.IC(g.group),
		"GroupVersion":         namer.IC(g.group) + namer.IC(g.version),
		"DeleteOptions":        c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "DeleteOptions"}),
		"ListOptions":          c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ListOptions"}),
		"GetOptions":           c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "GetOptions"}),
		"PatchType":            c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "PatchType"}),
		"watchInterface":       c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/watch", Name: "Interface"}),
		"RESTClientInterface":  c.Universe.Type(types.Name{Package: "k8s.io/client-go/rest", Name: "Interface"}),
		"schemeParameterCodec": c.Universe.Variable(types.Name{Package: filepath.Join(g.clientsetPackage, "scheme"), Name: "ParameterCodec"}),
	}

	sw.Do(getterComment, m)
	if tags.NonNamespaced {
		sw.Do(getterNonNamespaced, m)
	} else {
		sw.Do(getterNamespaced, m)
	}

	sw.Do(interfaceTemplate1, m)
	if !tags.NoVerbs {
		if !genStatus(t) {
			tags.SkipVerbs = append(tags.SkipVerbs, "updateStatus")
		}
		sw.Do(generateInterface(tags), m)
	}
	sw.Do(interfaceTemplate4, m)

	if tags.NonNamespaced {
		sw.Do(structNonNamespaced, m)
		sw.Do(newStructNonNamespaced, m)
	} else {
		sw.Do(structNamespaced, m)
		sw.Do(newStructNamespaced, m)
	}

	if tags.NoVerbs {
		return sw.Error()
	}

	if tags.HasVerb("get") {
		sw.Do(getTemplate, m)
	}
	if tags.HasVerb("list") {
		sw.Do(listTemplate, m)
	}
	if tags.HasVerb("watch") {
		sw.Do(watchTemplate, m)
	}

	if tags.HasVerb("create") {
		sw.Do(createTemplate, m)
	}
	if tags.HasVerb("update") {
		sw.Do(updateTemplate, m)
	}
	if tags.HasVerb("updateStatus") {
		sw.Do(updateStatusTemplate, m)
	}
	if tags.HasVerb("delete") {
		sw.Do(deleteTemplate, m)
	}
	if tags.HasVerb("deleteCollection") {
		sw.Do(deleteCollectionTemplate, m)
	}
	if tags.HasVerb("patch") {
		sw.Do(patchTemplate, m)
	}

	return sw.Error()
}

func generateInterface(tags util.Tags) string {
	// need an ordered list here to guarantee order of generated methods.
	out := []string{}
	for _, m := range util.SupportedVerbs {
		if tags.HasVerb(m) {
			out = append(out, defaultVerbTemplates[m])
		}
	}
	return strings.Join(out, "\n")
}

var defaultVerbTemplates = map[string]string{
	"create":           `Create(*$.type|raw$) (*$.type|raw$, error)`,
	"update":           `Update(*$.type|raw$) (*$.type|raw$, error)`,
	"updateStatus":     `UpdateStatus(*$.type|raw$) (*$.type|raw$, error)`,
	"delete":           `Delete(name string, options *$.DeleteOptions|raw$) error`,
	"deleteCollection": `DeleteCollection(options *$.DeleteOptions|raw$, listOptions $.ListOptions|raw$) error`,
	"get":              `Get(name string, options $.GetOptions|raw$) (*$.type|raw$, error)`,
	"list":             `List(opts $.ListOptions|raw$) (*$.type|raw$List, error)`,
	"watch":            `Watch(opts $.ListOptions|raw$) ($.watchInterface|raw$, error)`,
	"patch":            `Patch(name string, pt $.PatchType|raw$, data []byte, subresources ...string) (result *$.type|raw$, err error)`,
}

// group client will implement this interface.
var getterComment = `
// $.type|publicPlural$Getter has a method to return a $.type|public$Interface.
// A group's client should implement this interface.`

var getterNamespaced = `
type $.type|publicPlural$Getter interface {
	$.type|publicPlural$(namespace string) $.type|public$Interface
}
`

var getterNonNamespaced = `
type $.type|publicPlural$Getter interface {
	$.type|publicPlural$() $.type|public$Interface
}
`

// this type's interface, typed client will implement this interface.
var interfaceTemplate1 = `
// $.type|public$Interface has methods to work with $.type|public$ resources.
type $.type|public$Interface interface {`

var interfaceTemplate4 = `
	$.type|public$Expansion
}
`

// template for the struct that implements the type's interface
var structNamespaced = `
// $.type|privatePlural$ implements $.type|public$Interface
type $.type|privatePlural$ struct {
	client $.RESTClientInterface|raw$
	ns     string
}
`

// template for the struct that implements the type's interface
var structNonNamespaced = `
// $.type|privatePlural$ implements $.type|public$Interface
type $.type|privatePlural$ struct {
	client $.RESTClientInterface|raw$
}
`

var newStructNamespaced = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupVersion$Client, namespace string) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		client: c.RESTClient(),
		ns:     namespace,
	}
}
`

var newStructNonNamespaced = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupVersion$Client) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		client: c.RESTClient(),
	}
}
`

var listTemplate = `
// List takes label and field selectors, and returns the list of $.type|publicPlural$ that match those selectors.
func (c *$.type|privatePlural$) List(opts $.ListOptions|raw$) (result *$.type|raw$List, err error) {
	result = &$.type|raw$List{}
	err = c.client.Get().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Do().
		Into(result)
	return
}
`
var getTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.type|private$ object, and an error if there is any.
func (c *$.type|privatePlural$) Get(name string, options $.GetOptions|raw$) (result *$.type|raw$, err error) {
	result = &$.type|raw${}
	err = c.client.Get().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Name(name).
		VersionedParams(&options, $.schemeParameterCodec|raw$).
		Do().
		Into(result)
	return
}
`

var deleteTemplate = `
// Delete takes name of the $.type|private$ and deletes it. Returns an error if one occurs.
func (c *$.type|privatePlural$) Delete(name string, options *$.DeleteOptions|raw$) error {
	return c.client.Delete().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Name(name).
		Body(options).
		Do().
		Error()
}
`

var deleteCollectionTemplate = `
// DeleteCollection deletes a collection of objects.
func (c *$.type|privatePlural$) DeleteCollection(options *$.DeleteOptions|raw$, listOptions $.ListOptions|raw$) error {
	return c.client.Delete().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		VersionedParams(&listOptions, $.schemeParameterCodec|raw$).
		Body(options).
		Do().
		Error()
}
`

var createTemplate = `
// Create takes the representation of a $.type|private$ and creates it.  Returns the server's representation of the $.type|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Create($.type|private$ *$.type|raw$) (result *$.type|raw$, err error) {
	result = &$.type|raw${}
	err = c.client.Post().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
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
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Name($.type|private$.Name).
		Body($.type|private$).
		Do().
		Into(result)
	return
}
`

var updateStatusTemplate = `
// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().

func (c *$.type|privatePlural$) UpdateStatus($.type|private$ *$.type|raw$) (result *$.type|raw$, err error) {
	result = &$.type|raw${}
	err = c.client.Put().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Name($.type|private$.Name).
		SubResource("status").
		Body($.type|private$).
		Do().
		Into(result)
	return
}
`

var watchTemplate = `
// Watch returns a $.watchInterface|raw$ that watches the requested $.type|privatePlural$.
func (c *$.type|privatePlural$) Watch(opts $.ListOptions|raw$) ($.watchInterface|raw$, error) {
	opts.Watch = true
	return c.client.Get().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Watch()
}
`

var patchTemplate = `
// Patch applies the patch and returns the patched $.type|private$.
func (c *$.type|privatePlural$) Patch(name string, pt $.PatchType|raw$, data []byte, subresources ...string) (result *$.type|raw$, err error) {
	result = &$.type|raw${}
	err = c.client.Patch(pt).
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
`
