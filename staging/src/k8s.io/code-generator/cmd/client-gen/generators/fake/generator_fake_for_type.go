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
	"path"
	"strings"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"

	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
)

// genFakeForType produces a file for each top-level type.
type genFakeForType struct {
	generator.GoGenerator
	outputPackage             string // Must be a Go import-path
	group                     string
	version                   string
	groupGoName               string
	inputPackage              string
	typeToMatch               *types.Type
	imports                   namer.ImportTracker
	applyConfigurationPackage string
}

var _ generator.Generator = &genFakeForType{}

var titler = cases.Title(language.Und)

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
		if m.Embedded && m.Name == "ObjectMeta" {
			return true
		}
	}
	return false
}

// GenerateType makes the body of a file implementing the individual typed client for type t.
func (g *genFakeForType) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	pkg := path.Base(t.Name.Package)
	tags, err := util.ParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
	if err != nil {
		return err
	}

	const pkgClientGoTesting = "k8s.io/client-go/testing"
	m := map[string]interface{}{
		"type":               t,
		"inputType":          t,
		"resultType":         t,
		"subresourcePath":    "",
		"package":            pkg,
		"Package":            namer.IC(pkg),
		"namespaced":         !tags.NonNamespaced,
		"Group":              namer.IC(g.group),
		"GroupGoName":        g.groupGoName,
		"Version":            namer.IC(g.version),
		"version":            g.version,
		"SchemeGroupVersion": c.Universe.Type(types.Name{Package: t.Name.Package, Name: "SchemeGroupVersion"}),
		"CreateOptions":      c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "CreateOptions"}),
		"DeleteOptions":      c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "DeleteOptions"}),
		"GetOptions":         c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "GetOptions"}),
		"ListOptions":        c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ListOptions"}),
		"PatchOptions":       c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "PatchOptions"}),
		"ApplyOptions":       c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ApplyOptions"}),
		"UpdateOptions":      c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "UpdateOptions"}),
		"Everything":         c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/labels", Name: "Everything"}),
		"PatchType":          c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "PatchType"}),
		"ApplyPatchType":     c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "ApplyPatchType"}),
		"watchInterface":     c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/watch", Name: "Interface"}),
		"jsonMarshal":        c.Universe.Type(types.Name{Package: "encoding/json", Name: "Marshal"}),

		"NewRootListActionWithOptions":              c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootListActionWithOptions"}),
		"NewListActionWithOptions":                  c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewListActionWithOptions"}),
		"NewRootGetActionWithOptions":               c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootGetActionWithOptions"}),
		"NewGetActionWithOptions":                   c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewGetActionWithOptions"}),
		"NewRootDeleteActionWithOptions":            c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootDeleteActionWithOptions"}),
		"NewDeleteActionWithOptions":                c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewDeleteActionWithOptions"}),
		"NewRootDeleteCollectionActionWithOptions":  c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootDeleteCollectionActionWithOptions"}),
		"NewDeleteCollectionActionWithOptions":      c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewDeleteCollectionActionWithOptions"}),
		"NewRootUpdateActionWithOptions":            c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootUpdateActionWithOptions"}),
		"NewUpdateActionWithOptions":                c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewUpdateActionWithOptions"}),
		"NewRootCreateActionWithOptions":            c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootCreateActionWithOptions"}),
		"NewCreateActionWithOptions":                c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewCreateActionWithOptions"}),
		"NewRootWatchActionWithOptions":             c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootWatchActionWithOptions"}),
		"NewWatchActionWithOptions":                 c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewWatchActionWithOptions"}),
		"NewCreateSubresourceActionWithOptions":     c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewCreateSubresourceActionWithOptions"}),
		"NewRootCreateSubresourceActionWithOptions": c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootCreateSubresourceActionWithOptions"}),
		"NewUpdateSubresourceActionWithOptions":     c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewUpdateSubresourceActionWithOptions"}),
		"NewGetSubresourceActionWithOptions":        c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewGetSubresourceActionWithOptions"}),
		"NewRootGetSubresourceActionWithOptions":    c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootGetSubresourceActionWithOptions"}),
		"NewRootUpdateSubresourceActionWithOptions": c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootUpdateSubresourceActionWithOptions"}),
		"NewRootPatchActionWithOptions":             c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootPatchActionWithOptions"}),
		"NewPatchActionWithOptions":                 c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewPatchActionWithOptions"}),
		"NewRootPatchSubresourceActionWithOptions":  c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootPatchSubresourceActionWithOptions"}),
		"NewPatchSubresourceActionWithOptions":      c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewPatchSubresourceActionWithOptions"}),
		"ExtractFromListOptions":                    c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "ExtractFromListOptions"}),
	}

	generateApply := len(g.applyConfigurationPackage) > 0
	if generateApply {
		// Generated apply builder type references required for generated Apply function
		_, gvString := util.ParsePathGroupVersion(g.inputPackage)
		m["inputApplyConfig"] = types.Ref(path.Join(g.applyConfigurationPackage, gvString), t.Name.Name+"ApplyConfiguration")
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
	if tags.HasVerb("apply") && generateApply {
		sw.Do(applyTemplate, m)
	}
	if tags.HasVerb("applyStatus") && generateApply && genStatus(t) {
		sw.Do(applyStatusTemplate, m)
	}
	_, typeGVString := util.ParsePathGroupVersion(g.inputPackage)

	// generate extended client methods
	for _, e := range tags.Extensions {
		if e.HasVerb("apply") && !generateApply {
			continue
		}
		inputType := *t
		resultType := *t
		inputGVString := typeGVString
		if len(e.InputTypeOverride) > 0 {
			if name, pkg := e.Input(); len(pkg) > 0 {
				_, inputGVString = util.ParsePathGroupVersion(pkg)
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
		if e.HasVerb("apply") {
			m["inputApplyConfig"] = types.Ref(path.Join(g.applyConfigurationPackage, inputGVString), inputType.Name.Name+"ApplyConfiguration")
		}

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

		if e.HasVerb("apply") && generateApply {
			if e.IsSubresource() {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, applySubresourceTemplate), m)
			} else {
				sw.Do(adjustTemplate(e.VerbName, e.VerbType, applyTemplate), m)
			}
		}
	}

	return sw.Error()
}

// adjustTemplate adjust the origin verb template using the expansion name.
// TODO: Make the verbs in templates parametrized so the strings.Replace() is
// not needed.
func adjustTemplate(name, verbType, template string) string {
	return strings.ReplaceAll(template, " "+titler.String(verbType), " "+name)
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
var $.type|allLowercasePlural$Resource = $.SchemeGroupVersion|raw$.WithResource("$.type|resource$")
`

var kind = `
var $.type|allLowercasePlural$Kind = $.SchemeGroupVersion|raw$.WithKind("$.type|singularKind$")
`

var listTemplate = `
// List takes label and field selectors, and returns the list of $.type|publicPlural$ that match those selectors.
func (c *Fake$.type|publicPlural$) List(ctx context.Context, opts $.ListOptions|raw$) (result *$.type|raw$List, err error) {
	emptyResult := &$.type|raw$List{}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewListActionWithOptions|raw$($.type|allLowercasePlural$Resource, $.type|allLowercasePlural$Kind, c.ns, opts), emptyResult)
		$else$Invokes($.NewRootListActionWithOptions|raw$($.type|allLowercasePlural$Resource, $.type|allLowercasePlural$Kind, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.type|raw$List), err
}
`

var listUsingOptionsTemplate = `
// List takes label and field selectors, and returns the list of $.type|publicPlural$ that match those selectors.
func (c *Fake$.type|publicPlural$) List(ctx context.Context, opts $.ListOptions|raw$) (result *$.type|raw$List, err error) {
	emptyResult := &$.type|raw$List{}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewListActionWithOptions|raw$($.type|allLowercasePlural$Resource, $.type|allLowercasePlural$Kind, c.ns, opts), emptyResult)
		$else$Invokes($.NewRootListActionWithOptions|raw$($.type|allLowercasePlural$Resource, $.type|allLowercasePlural$Kind, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
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
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewGetActionWithOptions|raw$($.type|allLowercasePlural$Resource, c.ns, name, options), emptyResult)
		$else$Invokes($.NewRootGetActionWithOptions|raw$($.type|allLowercasePlural$Resource, name, options), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var getSubresourceTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.resultType|private$ object, and an error if there is any.
func (c *Fake$.type|publicPlural$) Get(ctx context.Context, $.type|private$Name string, options $.GetOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewGetSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, c.ns, "$.subresourcePath$", $.type|private$Name, options), emptyResult)
		$else$Invokes($.NewRootGetSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, "$.subresourcePath$", $.type|private$Name, options), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var deleteTemplate = `
// Delete takes name of the $.type|private$ and deletes it. Returns an error if one occurs.
func (c *Fake$.type|publicPlural$) Delete(ctx context.Context, name string, opts $.DeleteOptions|raw$) error {
	_, err := c.Fake.
		$if .namespaced$Invokes($.NewDeleteActionWithOptions|raw$($.type|allLowercasePlural$Resource, c.ns, name, opts), &$.type|raw${})
		$else$Invokes($.NewRootDeleteActionWithOptions|raw$($.type|allLowercasePlural$Resource, name, opts), &$.type|raw${})$end$
	return err
}
`

var deleteCollectionTemplate = `
// DeleteCollection deletes a collection of objects.
func (c *Fake$.type|publicPlural$) DeleteCollection(ctx context.Context, opts $.DeleteOptions|raw$, listOpts $.ListOptions|raw$) error {
	$if .namespaced$action := $.NewDeleteCollectionActionWithOptions|raw$($.type|allLowercasePlural$Resource, c.ns, opts, listOpts)
	$else$action := $.NewRootDeleteCollectionActionWithOptions|raw$($.type|allLowercasePlural$Resource, opts, listOpts)
	$end$
	_, err := c.Fake.Invokes(action, &$.type|raw$List{})
	return err
}
`
var createTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *Fake$.type|publicPlural$) Create(ctx context.Context, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewCreateActionWithOptions|raw$($.inputType|allLowercasePlural$Resource, c.ns, $.inputType|private$, opts), emptyResult)
		$else$Invokes($.NewRootCreateActionWithOptions|raw$($.inputType|allLowercasePlural$Resource, $.inputType|private$, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var createSubresourceTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *Fake$.type|publicPlural$) Create(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewCreateSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, $.type|private$Name, "$.subresourcePath$", c.ns, $.inputType|private$, opts), emptyResult)
		$else$Invokes($.NewRootCreateSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, $.type|private$Name, "$.subresourcePath$", $.inputType|private$, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var updateTemplate = `
// Update takes the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *Fake$.type|publicPlural$) Update(ctx context.Context, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewUpdateActionWithOptions|raw$($.inputType|allLowercasePlural$Resource, c.ns, $.inputType|private$, opts), emptyResult)
		$else$Invokes($.NewRootUpdateActionWithOptions|raw$($.inputType|allLowercasePlural$Resource, $.inputType|private$, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var updateSubresourceTemplate = `
// Update takes the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *Fake$.type|publicPlural$) Update(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewUpdateSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, "$.subresourcePath$", c.ns, $.inputType|private$, opts), &$.inputType|raw${})
		$else$Invokes($.NewRootUpdateSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, "$.subresourcePath$", $.inputType|private$, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var updateStatusTemplate = `
// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *Fake$.type|publicPlural$) UpdateStatus(ctx context.Context, $.type|private$ *$.type|raw$, opts $.UpdateOptions|raw$) (result *$.type|raw$, err error) {
	emptyResult := &$.type|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewUpdateSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, "status", c.ns, $.type|private$, opts), emptyResult)
		$else$Invokes($.NewRootUpdateSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, "status", $.type|private$, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.type|raw$), err
}
`

var watchTemplate = `
// Watch returns a $.watchInterface|raw$ that watches the requested $.type|privatePlural$.
func (c *Fake$.type|publicPlural$) Watch(ctx context.Context, opts $.ListOptions|raw$) ($.watchInterface|raw$, error) {
	return c.Fake.
		$if .namespaced$InvokesWatch($.NewWatchActionWithOptions|raw$($.type|allLowercasePlural$Resource, c.ns, opts))
		$else$InvokesWatch($.NewRootWatchActionWithOptions|raw$($.type|allLowercasePlural$Resource, opts))$end$
}
`

var patchTemplate = `
// Patch applies the patch and returns the patched $.resultType|private$.
func (c *Fake$.type|publicPlural$) Patch(ctx context.Context, name string, pt $.PatchType|raw$, data []byte, opts $.PatchOptions|raw$, subresources ...string) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewPatchSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, c.ns, name, pt, data, opts, subresources... ), emptyResult)
		$else$Invokes($.NewRootPatchSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, name, pt, data, opts, subresources...), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var applyTemplate = `
// Apply takes the given apply declarative configuration, applies it and returns the applied $.resultType|private$.
func (c *Fake$.type|publicPlural$) Apply(ctx context.Context, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error) {
	if $.inputType|private$ == nil {
		return nil, fmt.Errorf("$.inputType|private$ provided to Apply must not be nil")
	}
	data, err := $.jsonMarshal|raw$($.inputType|private$)
	if err != nil {
		return nil, err
	}
	name := $.inputType|private$.Name
	if name == nil {
		return nil, fmt.Errorf("$.inputType|private$.Name must be provided to Apply")
	}
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewPatchSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, c.ns, *name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions()), emptyResult)
		$else$Invokes($.NewRootPatchSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, *name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions()), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var applyStatusTemplate = `
// ApplyStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating ApplyStatus().
func (c *Fake$.type|publicPlural$) ApplyStatus(ctx context.Context, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error) {
	if $.inputType|private$ == nil {
		return nil, fmt.Errorf("$.inputType|private$ provided to Apply must not be nil")
	}
	data, err := $.jsonMarshal|raw$($.inputType|private$)
	if err != nil {
		return nil, err
	}
	name := $.inputType|private$.Name
	if name == nil {
		return nil, fmt.Errorf("$.inputType|private$.Name must be provided to Apply")
	}
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewPatchSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, c.ns, *name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions(), "status"), emptyResult)
		$else$Invokes($.NewRootPatchSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, *name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions(), "status"), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var applySubresourceTemplate = `
// Apply takes top resource name and the apply declarative configuration for $.subresourcePath$,
// applies it and returns the applied $.resultType|private$, and an error, if there is any.
func (c *Fake$.type|publicPlural$) Apply(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error) {
	if $.inputType|private$ == nil {
		return nil, fmt.Errorf("$.inputType|private$ provided to Apply must not be nil")
	}
	data, err := $.jsonMarshal|raw$($.inputType|private$)
	if err != nil {
		return nil, err
	}
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewPatchSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, c.ns, $.type|private$Name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions(), "status"), emptyResult)
		$else$Invokes($.NewRootPatchSubresourceActionWithOptions|raw$($.type|allLowercasePlural$Resource, $.type|private$Name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions(), "status"), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`
