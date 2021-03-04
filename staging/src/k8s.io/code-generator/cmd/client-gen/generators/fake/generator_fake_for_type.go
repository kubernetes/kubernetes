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

package fake

import (
	"io"
	"path/filepath"
	"strings"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
	"k8s.io/code-generator/cmd/client-gen/path"
)

// genFakeForType produces a file for each top-level type.
type genFakeForType struct {
	generator.DefaultGen
	outputPackage string
	group         string
	version       string
	groupGoName   string
	inputPackage  string
	typeToMatch   *types.Type
	imports       namer.ImportTracker
}

var _ generator.Generator = &genFakeForType{}

// Filter ignores all but one type because we're making a single file per type.
func (g *genFakeForType) Filter(c *generator.Context, t *types.Type) bool { return t == g.typeToMatch }

func (g *genFakeForType) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *genFakeForType) Imports(c *generator.Context) (imports []string) {
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

	tags := util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
	return hasStatus && !tags.NoStatus
}

// hasObjectMeta returns true if the type has a ObjectMeta field.
func hasObjectMeta(t *types.Type) bool {
	for _, m := range t.Members {
		if m.Embedded == true && m.Name == "ObjectMeta" {
			return true
		}
	}
	return false
}

// GenerateType makes the body of a file implementing the individual typed client for type t.
func (g *genFakeForType) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	pkg := filepath.Base(t.Name.Package)
	tags, err := util.ParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
	if err != nil {
		return err
	}
	canonicalGroup := g.group
	if canonicalGroup == "core" {
		canonicalGroup = ""
	}

	groupName := g.group
	if g.group == "core" {
		groupName = ""
	}

	// allow user to define a group name that's different from the one parsed from the directory.
	p := c.Universe.Package(path.Vendorless(g.inputPackage))
	if override := types.ExtractCommentTags("+", p.Comments)["groupName"]; override != nil {
		groupName = override[0]
	}

	const pkgClientGoTesting = "k8s.io/client-go/testing"
	m := map[string]interface{}{
		"type":                 t,
		"inputType":            t,
		"resultType":           t,
		"subresourcePath":      "",
		"package":              pkg,
		"Package":              namer.IC(pkg),
		"namespaced":           !tags.NonNamespaced,
		"Group":                namer.IC(g.group),
		"GroupGoName":          g.groupGoName,
		"Version":              namer.IC(g.version),
		"group":                canonicalGroup,
		"groupName":            groupName,
		"version":              g.version,
		"CreateOptions":        c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "CreateOptions"}),
		"DeleteOptions":        c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "DeleteOptions"}),
		"GetOptions":           c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "GetOptions"}),
		"ListOptions":          c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ListOptions"}),
		"PatchOptions":         c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "PatchOptions"}),
		"UpdateOptions":        c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "UpdateOptions"}),
		"Everything":           c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/labels", Name: "Everything"}),
		"GroupVersionResource": c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/schema", Name: "GroupVersionResource"}),
		"GroupVersionKind":     c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/schema", Name: "GroupVersionKind"}),
		"PatchType":            c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "PatchType"}),
		"watchInterface":       c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/watch", Name: "Interface"}),

		"NewRootListAction":              c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootListAction"}),
		"NewListAction":                  c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewListAction"}),
		"NewRootGetAction":               c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootGetAction"}),
		"NewGetAction":                   c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewGetAction"}),
		"NewRootDeleteAction":            c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootDeleteAction"}),
		"NewDeleteAction":                c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewDeleteAction"}),
		"NewRootDeleteCollectionAction":  c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootDeleteCollectionAction"}),
		"NewDeleteCollectionAction":      c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewDeleteCollectionAction"}),
		"NewRootUpdateAction":            c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootUpdateAction"}),
		"NewUpdateAction":                c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewUpdateAction"}),
		"NewRootCreateAction":            c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootCreateAction"}),
		"NewCreateAction":                c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewCreateAction"}),
		"NewRootWatchAction":             c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootWatchAction"}),
		"NewWatchAction":                 c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewWatchAction"}),
		"NewCreateSubresourceAction":     c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewCreateSubresourceAction"}),
		"NewRootCreateSubresourceAction": c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootCreateSubresourceAction"}),
		"NewUpdateSubresourceAction":     c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewUpdateSubresourceAction"}),
		"NewGetSubresourceAction":        c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewGetSubresourceAction"}),
		"NewRootGetSubresourceAction":    c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootGetSubresourceAction"}),
		"NewRootUpdateSubresourceAction": c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootUpdateSubresourceAction"}),
		"NewRootPatchAction":             c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootPatchAction"}),
		"NewPatchAction":                 c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewPatchAction"}),
		"NewRootPatchSubresourceAction":  c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootPatchSubresourceAction"}),
		"NewPatchSubresourceAction":      c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewPatchSubresourceAction"}),
		"ExtractFromListOptions":         c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "ExtractFromListOptions"}),
	}

	if tags.NonNamespaced {
		sw.Do(structNonNamespaced, m)
	} else {
		sw.Do(structNamespaced, m)
	}

	if tags.NoVerbs {
		return sw.Error()
	}
	sw.Do(resource, m)
	sw.Do(kind, m)

	if tags.HasVerb("get") {
		sw.Do(getTemplate, m)
	}
	if tags.HasVerb("list") {
		if hasObjectMeta(t) {
			sw.Do(listUsingOptionsTemplate, m)
		} else {
			sw.Do(listTemplate, m)
		}
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
	if tags.HasVerb("updateStatus") && genStatus(t) {
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

	// generate extended client methods
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

			sw.Do(adjustTemplate(e.VerbName, e.VerbType, listTemplate), m)
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

// template for the struct that implements the type's interface
var structNamespaced = `
// Fake$.type|publicPlural$ implements $.type|public$Interface
type Fake$.type|publicPlural$ struct {
	Fake *Fake$.GroupGoName$$.Version$
	ns     string
}
`

// template for the struct that implements the type's interface
var structNonNamespaced = `
// Fake$.type|publicPlural$ implements $.type|public$Interface
type Fake$.type|publicPlural$ struct {
	Fake *Fake$.GroupGoName$$.Version$
}
`

var resource = `
var $.type|allLowercasePlural$Resource = $.GroupVersionResource|raw${Group: "$.groupName$", Version: "$.version$", Resource: "$.type|resource$"}
`

var kind = `
var $.type|allLowercasePlural$Kind = $.GroupVersionKind|raw${Group: "$.groupName$", Version: "$.version$", Kind: "$.type|singularKind$"}
`

var listTemplate = `
// List takes label and field selectors, and returns the list of $.type|publicPlural$ that match those selectors.
func (c *Fake$.type|publicPlural$) List(ctx context.Context, opts $.ListOptions|raw$) (result *$.type|raw$List, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewListAction|raw$($.type|allLowercasePlural$Resource, $.type|allLowercasePlural$Kind, c.ns, opts), &$.type|raw$List{})
		$else$Invokes($.NewRootListAction|raw$($.type|allLowercasePlural$Resource, $.type|allLowercasePlural$Kind, opts), &$.type|raw$List{})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.type|raw$List), err
}
`

var listUsingOptionsTemplate = `
// List takes label and field selectors, and returns the list of $.type|publicPlural$ that match those selectors.
func (c *Fake$.type|publicPlural$) List(ctx context.Context, opts $.ListOptions|raw$) (result *$.type|raw$List, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewListAction|raw$($.type|allLowercasePlural$Resource, $.type|allLowercasePlural$Kind, c.ns, opts), &$.type|raw$List{})
		$else$Invokes($.NewRootListAction|raw$($.type|allLowercasePlural$Resource, $.type|allLowercasePlural$Kind, opts), &$.type|raw$List{})$end$
	if obj == nil {
		return nil, err
	}

	label, _, _ := $.ExtractFromListOptions|raw$(opts)
	if label == nil {
		label = $.Everything|raw$()
	}
	list := &$.type|raw$List{ListMeta: obj.(*$.type|raw$List).ListMeta}
	for _, item := range obj.(*$.type|raw$List).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}
`

var getTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.resultType|private$ object, and an error if there is any.
func (c *Fake$.type|publicPlural$) Get(ctx context.Context, name string, options $.GetOptions|raw$) (result *$.resultType|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewGetAction|raw$($.type|allLowercasePlural$Resource, c.ns, name), &$.resultType|raw${})
		$else$Invokes($.NewRootGetAction|raw$($.type|allLowercasePlural$Resource, name), &$.resultType|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var getSubresourceTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.resultType|private$ object, and an error if there is any.
func (c *Fake$.type|publicPlural$) Get(ctx context.Context, $.type|private$Name string, options $.GetOptions|raw$) (result *$.resultType|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewGetSubresourceAction|raw$($.type|allLowercasePlural$Resource, c.ns, "$.subresourcePath$", $.type|private$Name), &$.resultType|raw${})
		$else$Invokes($.NewRootGetSubresourceAction|raw$($.type|allLowercasePlural$Resource, "$.subresourcePath$", $.type|private$Name), &$.resultType|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var deleteTemplate = `
// Delete takes name of the $.type|private$ and deletes it. Returns an error if one occurs.
func (c *Fake$.type|publicPlural$) Delete(ctx context.Context, name string, opts $.DeleteOptions|raw$) error {
	_, err := c.Fake.
		$if .namespaced$Invokes($.NewDeleteAction|raw$($.type|allLowercasePlural$Resource, c.ns, name), &$.type|raw${})
		$else$Invokes($.NewRootDeleteAction|raw$($.type|allLowercasePlural$Resource, name), &$.type|raw${})$end$
	return err
}
`

var deleteCollectionTemplate = `
// DeleteCollection deletes a collection of objects.
func (c *Fake$.type|publicPlural$) DeleteCollection(ctx context.Context, opts $.DeleteOptions|raw$, listOpts $.ListOptions|raw$) error {
	$if .namespaced$action := $.NewDeleteCollectionAction|raw$($.type|allLowercasePlural$Resource, c.ns, listOpts)
	$else$action := $.NewRootDeleteCollectionAction|raw$($.type|allLowercasePlural$Resource, listOpts)
	$end$
	_, err := c.Fake.Invokes(action, &$.type|raw$List{})
	return err
}
`
var createTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *Fake$.type|publicPlural$) Create(ctx context.Context, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (result *$.resultType|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewCreateAction|raw$($.inputType|allLowercasePlural$Resource, c.ns, $.inputType|private$), &$.resultType|raw${})
		$else$Invokes($.NewRootCreateAction|raw$($.inputType|allLowercasePlural$Resource, $.inputType|private$), &$.resultType|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var createSubresourceTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *Fake$.type|publicPlural$) Create(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (result *$.resultType|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewCreateSubresourceAction|raw$($.type|allLowercasePlural$Resource, $.type|private$Name, "$.subresourcePath$", c.ns, $.inputType|private$), &$.resultType|raw${})
		$else$Invokes($.NewRootCreateSubresourceAction|raw$($.type|allLowercasePlural$Resource, $.type|private$Name, "$.subresourcePath$", $.inputType|private$), &$.resultType|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var updateTemplate = `
// Update takes the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *Fake$.type|publicPlural$) Update(ctx context.Context, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (result *$.resultType|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewUpdateAction|raw$($.inputType|allLowercasePlural$Resource, c.ns, $.inputType|private$), &$.resultType|raw${})
		$else$Invokes($.NewRootUpdateAction|raw$($.inputType|allLowercasePlural$Resource, $.inputType|private$), &$.resultType|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var updateSubresourceTemplate = `
// Update takes the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *Fake$.type|publicPlural$) Update(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (result *$.resultType|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewUpdateSubresourceAction|raw$($.type|allLowercasePlural$Resource, "$.subresourcePath$", c.ns, $.inputType|private$), &$.inputType|raw${})
		$else$Invokes($.NewRootUpdateSubresourceAction|raw$($.type|allLowercasePlural$Resource, "$.subresourcePath$", $.inputType|private$), &$.resultType|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var updateStatusTemplate = `
// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *Fake$.type|publicPlural$) UpdateStatus(ctx context.Context, $.type|private$ *$.type|raw$, opts $.UpdateOptions|raw$) (*$.type|raw$, error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewUpdateSubresourceAction|raw$($.type|allLowercasePlural$Resource, "status", c.ns, $.type|private$), &$.type|raw${})
		$else$Invokes($.NewRootUpdateSubresourceAction|raw$($.type|allLowercasePlural$Resource, "status", $.type|private$), &$.type|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.type|raw$), err
}
`

var watchTemplate = `
// Watch returns a $.watchInterface|raw$ that watches the requested $.type|privatePlural$.
func (c *Fake$.type|publicPlural$) Watch(ctx context.Context, opts $.ListOptions|raw$) ($.watchInterface|raw$, error) {
	return c.Fake.
		$if .namespaced$InvokesWatch($.NewWatchAction|raw$($.type|allLowercasePlural$Resource, c.ns, opts))
		$else$InvokesWatch($.NewRootWatchAction|raw$($.type|allLowercasePlural$Resource, opts))$end$
}
`

var patchTemplate = `
// Patch applies the patch and returns the patched $.resultType|private$.
func (c *Fake$.type|publicPlural$) Patch(ctx context.Context, name string, pt $.PatchType|raw$, data []byte, opts $.PatchOptions|raw$, subresources ...string) (result *$.resultType|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewPatchSubresourceAction|raw$($.type|allLowercasePlural$Resource, c.ns, name, pt, data, subresources... ), &$.resultType|raw${})
		$else$Invokes($.NewRootPatchSubresourceAction|raw$($.type|allLowercasePlural$Resource, name, pt, data, subresources...), &$.resultType|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.resultType|raw$), err
}
`
