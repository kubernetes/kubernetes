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

	"k8s.io/code-generator/cmd/client-gen/generators/util"
)

// genClientForType produces a file for each top-level type.
type genClientForType struct {
	generator.DefaultGen
	outputPackage    string
	clientsetPackage string
	group            string
	version          string
	groupGoName      string
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
	return hasStatus && !util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...)).NoStatus
}

// GenerateType makes the body of a file implementing the individual typed client for type t.
func (g *genClientForType) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	pkg := filepath.Base(t.Name.Package)
	tags, err := util.ParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
	if err != nil {
		return err
	}
	type extendedInterfaceMethod struct {
		template string
		args     map[string]interface{}
	}
	extendedMethods := []extendedInterfaceMethod{}
	for _, e := range tags.Extensions {
		inputType := *t
		resultType := *t
		// TODO: Extract this to some helper method as this code is copied into
		// 2 other places.
		if len(e.InputTypeOverride) > 0 {
			if name, pkg := e.Input(); len(pkg) > 0 {
				newType := c.Universe.Type(types.Name{Package: pkg, Name: name})
				inputType = *newType
			} else {
				inputType.Name.Name = e.InputTypeOverride
			}
		}
		if len(e.ResultTypeOverride) > 0 {
			if name, pkg := e.Result(); len(pkg) > 0 {
				newType := c.Universe.Type(types.Name{Package: pkg, Name: name})
				resultType = *newType
			} else {
				resultType.Name.Name = e.ResultTypeOverride
			}
		}
		var updatedVerbtemplate string
		if _, exists := subresourceDefaultVerbTemplates[e.VerbType]; e.IsSubresource() && exists {
			updatedVerbtemplate = e.VerbName + "(" + strings.TrimPrefix(subresourceDefaultVerbTemplates[e.VerbType], strings.Title(e.VerbType)+"(")
		} else {
			updatedVerbtemplate = e.VerbName + "(" + strings.TrimPrefix(defaultVerbTemplates[e.VerbType], strings.Title(e.VerbType)+"(")
		}
		extendedMethods = append(extendedMethods, extendedInterfaceMethod{
			template: updatedVerbtemplate,
			args: map[string]interface{}{
				"type":          t,
				"inputType":     &inputType,
				"resultType":    &resultType,
				"DeleteOptions": c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "DeleteOptions"}),
				"ListOptions":   c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ListOptions"}),
				"GetOptions":    c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "GetOptions"}),
				"PatchType":     c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "PatchType"}),
			},
		})
	}
	m := map[string]interface{}{
		"type":                 t,
		"inputType":            t,
		"resultType":           t,
		"package":              pkg,
		"Package":              namer.IC(pkg),
		"namespaced":           !tags.NonNamespaced,
		"Group":                namer.IC(g.group),
		"subresource":          false,
		"subresourcePath":      "",
		"GroupGoName":          g.groupGoName,
		"Version":              namer.IC(g.version),
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
		interfaceSuffix := ""
		if len(extendedMethods) > 0 {
			interfaceSuffix = "\n"
		}
		sw.Do("\n"+generateInterface(tags)+interfaceSuffix, m)
		// add extended verbs into interface
		for _, v := range extendedMethods {
			sw.Do(v.template+interfaceSuffix, v.args)
		}

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

	// generate expansion methods
	for _, e := range tags.Extensions {
		inputType := *t
		resultType := *t
		if len(e.InputTypeOverride) > 0 {
			if name, pkg := e.Input(); len(pkg) > 0 {
				newType := c.Universe.Type(types.Name{Package: pkg, Name: name})
				inputType = *newType
			} else {
				inputType.Name.Name = e.InputTypeOverride
			}
		}
		if len(e.ResultTypeOverride) > 0 {
			if name, pkg := e.Result(); len(pkg) > 0 {
				newType := c.Universe.Type(types.Name{Package: pkg, Name: name})
				resultType = *newType
			} else {
				resultType.Name.Name = e.ResultTypeOverride
			}
		}
		m["inputType"] = &inputType
		m["resultType"] = &resultType
		m["subresourcePath"] = e.SubResourcePath

		if e.HasVerb("get") {
			if e.IsSubresource() {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, getSubresourceTemplate), m)
			} else {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, getTemplate), m)
			}
		}

		if e.HasVerb("list") {
			if e.IsSubresource() {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, listSubresourceTemplate), m)
			} else {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, listTemplate), m)
			}
		}

		// TODO: Figure out schemantic for watching a sub-resource.
		if e.HasVerb("watch") {
			sw.Do(adjustTemplate(e.VerbName, e.VerbType, watchTemplate), m)
		}

		if e.HasVerb("create") {
			if e.IsSubresource() {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, createSubresourceTemplate), m)
			} else {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, createTemplate), m)
			}
		}

		if e.HasVerb("update") {
			if e.IsSubresource() {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, updateSubresourceTemplate), m)
			} else {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, updateTemplate), m)
			}
		}

		// TODO: Figure out schemantic for deleting a sub-resource (what arguments
		// are passed, does it need two names? etc.
		if e.HasVerb("delete") {
			sw.Do(adjustTemplate(e.VerbName, e.VerbType, deleteTemplate), m)
		}

		if e.HasVerb("patch") {
			sw.Do(adjustTemplate(e.VerbName, e.VerbType, patchTemplate), m)
		}
	}

	return sw.Error()
}

// adjustTemplate adjust the origin verb template using the expansion name.
// TODO: Make the verbs in templates parametrized so the strings.Replace() is
// not needed.
func adjustTemplate(name, verbType, template string) string {
	return strings.Replace(template, " "+strings.Title(verbType), " "+name, -1)
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

var subresourceDefaultVerbTemplates = map[string]string{
	"create": `Create($.type|private$Name string, $.inputType|private$ *$.inputType|raw$) (*$.resultType|raw$, error)`,
	"list":   `List($.type|private$Name string, opts $.ListOptions|raw$) (*$.resultType|raw$List, error)`,
	"update": `Update($.type|private$Name string, $.inputType|private$ *$.inputType|raw$) (*$.resultType|raw$, error)`,
	"get":    `Get($.type|private$Name string, options $.GetOptions|raw$) (*$.resultType|raw$, error)`,
}

var defaultVerbTemplates = map[string]string{
	"create":           `Create(*$.inputType|raw$) (*$.resultType|raw$, error)`,
	"update":           `Update(*$.inputType|raw$) (*$.resultType|raw$, error)`,
	"updateStatus":     `UpdateStatus(*$.type|raw$) (*$.type|raw$, error)`,
	"delete":           `Delete(name string, options *$.DeleteOptions|raw$) error`,
	"deleteCollection": `DeleteCollection(options *$.DeleteOptions|raw$, listOptions $.ListOptions|raw$) error`,
	"get":              `Get(name string, options $.GetOptions|raw$) (*$.resultType|raw$, error)`,
	"list":             `List(opts $.ListOptions|raw$) (*$.resultType|raw$List, error)`,
	"watch":            `Watch(opts $.ListOptions|raw$) ($.watchInterface|raw$, error)`,
	"patch":            `Patch(name string, pt $.PatchType|raw$, data []byte, subresources ...string) (result *$.resultType|raw$, err error)`,
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
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client, namespace string) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		client: c.RESTClient(),
		ns:     namespace,
	}
}
`

var newStructNonNamespaced = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		client: c.RESTClient(),
	}
}
`
var listTemplate = `
// List takes label and field selectors, and returns the list of $.resultType|publicPlural$ that match those selectors.
func (c *$.type|privatePlural$) List(opts $.ListOptions|raw$) (result *$.resultType|raw$List, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil{
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &$.resultType|raw$List{}
	err = c.client.Get().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Timeout(timeout).
		Do().
		Into(result)
	return
}
`

var listSubresourceTemplate = `
// List takes $.type|raw$ name, label and field selectors, and returns the list of $.resultType|publicPlural$ that match those selectors.
func (c *$.type|privatePlural$) List($.type|private$Name string, opts $.ListOptions|raw$) (result *$.resultType|raw$List, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil{
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &$.resultType|raw$List{}
	err = c.client.Get().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Name($.type|private$Name).
		SubResource("$.subresourcePath$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Timeout(timeout).
		Do().
		Into(result)
	return
}
`

var getTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.resultType|private$ object, and an error if there is any.
func (c *$.type|privatePlural$) Get(name string, options $.GetOptions|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
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

var getSubresourceTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.resultType|raw$ object, and an error if there is any.
func (c *$.type|privatePlural$) Get($.type|private$Name string, options $.GetOptions|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.client.Get().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Name($.type|private$Name).
		SubResource("$.subresourcePath$").
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
	var timeout time.Duration
	if listOptions.TimeoutSeconds != nil{
		timeout = time.Duration(*listOptions.TimeoutSeconds) * time.Second
	}
	return c.client.Delete().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		VersionedParams(&listOptions, $.schemeParameterCodec|raw$).
		Timeout(timeout).
		Body(options).
		Do().
		Error()
}
`

var createSubresourceTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Create($.type|private$Name string, $.inputType|private$ *$.inputType|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.client.Post().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Name($.type|private$Name).
		SubResource("$.subresourcePath$").
		Body($.inputType|private$).
		Do().
		Into(result)
	return
}
`

var createTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Create($.inputType|private$ *$.inputType|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.client.Post().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Body($.inputType|private$).
		Do().
		Into(result)
	return
}
`

var updateSubresourceTemplate = `
// Update takes the top resource name and the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Update($.type|private$Name string, $.inputType|private$ *$.inputType|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.client.Put().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Name($.type|private$Name).
		SubResource("$.subresourcePath$").
		Body($.inputType|private$).
		Do().
		Into(result)
	return
}
`

var updateTemplate = `
// Update takes the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Update($.inputType|private$ *$.inputType|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.client.Put().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		Name($.inputType|private$.Name).
		Body($.inputType|private$).
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
	var timeout time.Duration
	if opts.TimeoutSeconds != nil{
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.client.Get().
		$if .namespaced$Namespace(c.ns).$end$
		Resource("$.type|resource$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Timeout(timeout).
		Watch()
}
`

var patchTemplate = `
// Patch applies the patch and returns the patched $.resultType|private$.
func (c *$.type|privatePlural$) Patch(name string, pt $.PatchType|raw$, data []byte, subresources ...string) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
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
