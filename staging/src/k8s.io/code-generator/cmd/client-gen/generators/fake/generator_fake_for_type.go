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
	realClientPackage         string // Must be a Go import-path
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

// GenerateType makes the body of a file implementing the individual typed client for type t.
func (g *genFakeForType) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	tags, err := util.ParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
	if err != nil {
		return err
	}

	const pkgClientGoTesting = "k8s.io/client-go/testing"
	m := map[string]interface{}{
		"type":                t,
		"inputType":           t,
		"resultType":          t,
		"subresourcePath":     "",
		"namespaced":          !tags.NonNamespaced,
		"GroupGoName":         g.groupGoName,
		"Version":             namer.IC(g.version),
		"realClientInterface": c.Universe.Type(types.Name{Package: g.realClientPackage, Name: t.Name.Name + "Interface"}),
		"SchemeGroupVersion":  c.Universe.Type(types.Name{Package: t.Name.Package, Name: "SchemeGroupVersion"}),
		"CreateOptions":       c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "CreateOptions"}),
		"DeleteOptions":       c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "DeleteOptions"}),
		"GetOptions":          c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "GetOptions"}),
		"ListOptions":         c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ListOptions"}),
		"PatchOptions":        c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "PatchOptions"}),
		"ApplyOptions":        c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ApplyOptions"}),
		"UpdateOptions":       c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "UpdateOptions"}),
		"PatchType":           c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "PatchType"}),
		"ApplyPatchType":      c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "ApplyPatchType"}),
		"watchInterface":      c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/watch", Name: "Interface"}),
		"jsonMarshal":         c.Universe.Type(types.Name{Package: "encoding/json", Name: "Marshal"}),
		"fmtErrorf":           c.Universe.Type(types.Name{Package: "fmt", Name: "Errorf"}),
		"contextContext":      c.Universe.Type(types.Name{Package: "context", Name: "Context"}),

		"NewRootListActionWithOptions":              c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootListActionWithOptions"}),
		"NewListActionWithOptions":                  c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewListActionWithOptions"}),
		"NewRootGetActionWithOptions":               c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootGetActionWithOptions"}),
		"NewGetActionWithOptions":                   c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewGetActionWithOptions"}),
		"NewRootDeleteActionWithOptions":            c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootDeleteActionWithOptions"}),
		"NewDeleteActionWithOptions":                c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewDeleteActionWithOptions"}),
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
		"NewRootPatchSubresourceActionWithOptions":  c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewRootPatchSubresourceActionWithOptions"}),
		"NewPatchSubresourceActionWithOptions":      c.Universe.Function(types.Name{Package: pkgClientGoTesting, Name: "NewPatchSubresourceActionWithOptions"}),
		"FakeClient":                                c.Universe.Type(types.Name{Package: "k8s.io/client-go/gentype", Name: "FakeClient"}),
		"NewFakeClient":                             c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype", Name: "NewFakeClient"}),
		"FakeClientWithApply":                       c.Universe.Type(types.Name{Package: "k8s.io/client-go/gentype", Name: "FakeClientWithApply"}),
		"NewFakeClientWithApply":                    c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype", Name: "NewFakeClientWithApply"}),
		"FakeClientWithList":                        c.Universe.Type(types.Name{Package: "k8s.io/client-go/gentype", Name: "FakeClientWithList"}),
		"NewFakeClientWithList":                     c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype", Name: "NewFakeClientWithList"}),
		"FakeClientWithListAndApply":                c.Universe.Type(types.Name{Package: "k8s.io/client-go/gentype", Name: "FakeClientWithListAndApply"}),
		"NewFakeClientWithListAndApply":             c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype", Name: "NewFakeClientWithListAndApply"}),
	}

	generateApply := len(g.applyConfigurationPackage) > 0
	if generateApply {
		// Generated apply builder type references required for generated Apply function
		_, gvString := util.ParsePathGroupVersion(g.inputPackage)
		m["inputApplyConfig"] = types.Ref(path.Join(g.applyConfigurationPackage, gvString), t.Name.Name+"ApplyConfiguration")
	}

	listableOrAppliable := noList | noApply

	if !tags.NoVerbs && tags.HasVerb("list") {
		listableOrAppliable |= withList
	}

	if !tags.NoVerbs && tags.HasVerb("apply") && generateApply {
		listableOrAppliable |= withApply
	}

	sw.Do(structType[listableOrAppliable], m)
	sw.Do(newStruct[listableOrAppliable], m)

	if tags.NoVerbs {
		return sw.Error()
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

// struct and constructor variants
const (
	// The following values are bits in a bitmask.
	// The values which can be set indicate list support and apply support;
	// to make the declarations easier to read (like a truth table), corresponding zero-values
	// are also declared.
	noList   = 0
	noApply  = 0
	withList = 1 << iota
	withApply
)

// The following string slices are similar to maps, but with combinable keys used as indices.
// Each entry defines whether it supports lists and/or apply; each bit is then toggled:
// * noList, noApply: index 0;
// * withList, noApply: index 1;
// * noList, withApply: index 2;
// * withList, withApply: index 3.
// Go enforces index unicity in these kinds of declarations.

// struct declarations
var structType = []string{
	noList | noApply: `
	// fake$.type|publicPlural$ implements $.type|public$Interface
	type fake$.type|publicPlural$ struct {
		*$.FakeClient|raw$[*$.type|raw$]
		Fake *Fake$.GroupGoName$$.Version$
	}
	`,
	withList | noApply: `
	// fake$.type|publicPlural$ implements $.type|public$Interface
	type fake$.type|publicPlural$ struct {
		*$.FakeClientWithList|raw$[*$.type|raw$, *$.type|raw$List]
		Fake *Fake$.GroupGoName$$.Version$
	}
	`,
	noList | withApply: `
	// fake$.type|publicPlural$ implements $.type|public$Interface
	type fake$.type|publicPlural$ struct {
		*$.FakeClientWithApply|raw$[*$.type|raw$, *$.inputApplyConfig|raw$]
		Fake *Fake$.GroupGoName$$.Version$
	}
	`,
	withList | withApply: `
	// fake$.type|publicPlural$ implements $.type|public$Interface
	type fake$.type|publicPlural$ struct {
		*$.FakeClientWithListAndApply|raw$[*$.type|raw$, *$.type|raw$List, *$.inputApplyConfig|raw$]
		Fake *Fake$.GroupGoName$$.Version$
	}
	`,
}

// Constructors for the struct, in all variants
var newStruct = []string{
	noList | noApply: `
	func newFake$.type|publicPlural$(fake *Fake$.GroupGoName$$.Version$$if .namespaced$, namespace string$end$) $.realClientInterface|raw$ {
		return &fake$.type|publicPlural${
			$.NewFakeClient|raw$[*$.type|raw$](
				fake.Fake,
				$if .namespaced$namespace$else$""$end$,
				$.SchemeGroupVersion|raw$.WithResource("$.type|resource$"),
				$.SchemeGroupVersion|raw$.WithKind("$.type|singularKind$"),
				func() *$.type|raw$ {return &$.type|raw${}},
			),
			fake,
		}
	}
	`,
	noList | withApply: `
	func newFake$.type|publicPlural$(fake *Fake$.GroupGoName$$.Version$$if .namespaced$, namespace string$end$) $.realClientInterface|raw$ {
		return &fake$.type|publicPlural${
			$.NewFakeClientWithApply|raw$[*$.type|raw$, *$.inputApplyConfig|raw$](
				fake.Fake,
				$if .namespaced$namespace$else$""$end$,
				$.SchemeGroupVersion|raw$.WithResource("$.type|resource$"),
				$.SchemeGroupVersion|raw$.WithKind("$.type|singularKind$"),
				func() *$.type|raw$ {return &$.type|raw${}},
			),
			fake,
		}
	}
	`,
	withList | noApply: `
	func newFake$.type|publicPlural$(fake *Fake$.GroupGoName$$.Version$$if .namespaced$, namespace string$end$) $.realClientInterface|raw$ {
		return &fake$.type|publicPlural${
			$.NewFakeClientWithList|raw$[*$.type|raw$, *$.type|raw$List](
				fake.Fake,
				$if .namespaced$namespace$else$""$end$,
				$.SchemeGroupVersion|raw$.WithResource("$.type|resource$"),
				$.SchemeGroupVersion|raw$.WithKind("$.type|singularKind$"),
				func() *$.type|raw$ {return &$.type|raw${}},
				func() *$.type|raw$List {return &$.type|raw$List{}},
				func(dst, src *$.type|raw$List) {dst.ListMeta = src.ListMeta},
				func(list *$.type|raw$List) []*$.type|raw$ {return gentype.ToPointerSlice(list.Items)},
				func(list *$.type|raw$List, items []*$.type|raw$) {list.Items = gentype.FromPointerSlice(items)},
			),
			fake,
		}
	}
	`,
	withList | withApply: `
	func newFake$.type|publicPlural$(fake *Fake$.GroupGoName$$.Version$$if .namespaced$, namespace string$end$) $.realClientInterface|raw$ {
		return &fake$.type|publicPlural${
			$.NewFakeClientWithListAndApply|raw$[*$.type|raw$, *$.type|raw$List, *$.inputApplyConfig|raw$](
				fake.Fake,
				$if .namespaced$namespace$else$""$end$,
				$.SchemeGroupVersion|raw$.WithResource("$.type|resource$"),
				$.SchemeGroupVersion|raw$.WithKind("$.type|singularKind$"),
				func() *$.type|raw$ {return &$.type|raw${}},
				func() *$.type|raw$List {return &$.type|raw$List{}},
				func(dst, src *$.type|raw$List) {dst.ListMeta = src.ListMeta},
				func(list *$.type|raw$List) []*$.type|raw$ {return gentype.ToPointerSlice(list.Items)},
				func(list *$.type|raw$List, items []*$.type|raw$) {list.Items = gentype.FromPointerSlice(items)},
			),
			fake,
		}
	}
	`,
}

var listTemplate = `
// List takes label and field selectors, and returns the list of $.type|publicPlural$ that match those selectors.
func (c *fake$.type|publicPlural$) List(ctx $.contextContext|raw$, opts $.ListOptions|raw$) (result *$.type|raw$List, err error) {
	emptyResult := &$.type|raw$List{}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewListActionWithOptions|raw$(c.Resource(), c.Kind(), c.Namespace(), opts), emptyResult)
		$else$Invokes($.NewRootListActionWithOptions|raw$(c.Resource(), c.Kind(), opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.type|raw$List), err
}
`

var getTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.resultType|private$ object, and an error if there is any.
func (c *fake$.type|publicPlural$) Get(ctx $.contextContext|raw$, name string, options $.GetOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewGetActionWithOptions|raw$(c.Resource(), c.Namespace(), name, options), emptyResult)
		$else$Invokes($.NewRootGetActionWithOptions|raw$(c.Resource(), name, options), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var getSubresourceTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.resultType|private$ object, and an error if there is any.
func (c *fake$.type|publicPlural$) Get(ctx $.contextContext|raw$, $.type|private$Name string, options $.GetOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewGetSubresourceActionWithOptions|raw$(c.Resource(), c.Namespace(), "$.subresourcePath$", $.type|private$Name, options), emptyResult)
		$else$Invokes($.NewRootGetSubresourceActionWithOptions|raw$(c.Resource(), "$.subresourcePath$", $.type|private$Name, options), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var deleteTemplate = `
// Delete takes name of the $.type|private$ and deletes it. Returns an error if one occurs.
func (c *fake$.type|publicPlural$) Delete(ctx $.contextContext|raw$, name string, opts $.DeleteOptions|raw$) error {
	_, err := c.Fake.
		$if .namespaced$Invokes($.NewDeleteActionWithOptions|raw$(c.Resource(), c.Namespace(), name, opts), &$.type|raw${})
		$else$Invokes($.NewRootDeleteActionWithOptions|raw$(c.Resource(), name, opts), &$.type|raw${})$end$
	return err
}
`

var createTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *fake$.type|publicPlural$) Create(ctx $.contextContext|raw$, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewCreateActionWithOptions|raw$(c.Resource(), c.Namespace(), $.inputType|private$, opts), emptyResult)
		$else$Invokes($.NewRootCreateActionWithOptions|raw$(c.Resource(), $.inputType|private$, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var createSubresourceTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *fake$.type|publicPlural$) Create(ctx $.contextContext|raw$, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewCreateSubresourceActionWithOptions|raw$(c.Resource(), $.type|private$Name, "$.subresourcePath$", c.Namespace(), $.inputType|private$, opts), emptyResult)
		$else$Invokes($.NewRootCreateSubresourceActionWithOptions|raw$(c.Resource(), $.type|private$Name, "$.subresourcePath$", $.inputType|private$, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var updateTemplate = `
// Update takes the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *fake$.type|publicPlural$) Update(ctx $.contextContext|raw$, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewUpdateActionWithOptions|raw$(c.Resource(), c.Namespace(), $.inputType|private$, opts), emptyResult)
		$else$Invokes($.NewRootUpdateActionWithOptions|raw$(c.Resource(), $.inputType|private$, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var updateSubresourceTemplate = `
// Update takes the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *fake$.type|publicPlural$) Update(ctx $.contextContext|raw$, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewUpdateSubresourceActionWithOptions|raw$(c.Resource(), "$.subresourcePath$", c.Namespace(), $.inputType|private$, opts), &$.inputType|raw${})
		$else$Invokes($.NewRootUpdateSubresourceActionWithOptions|raw$(c.Resource(), "$.subresourcePath$", $.inputType|private$, opts), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var watchTemplate = `
// Watch returns a $.watchInterface|raw$ that watches the requested $.type|privatePlural$.
func (c *fake$.type|publicPlural$) Watch(ctx $.contextContext|raw$, opts $.ListOptions|raw$) ($.watchInterface|raw$, error) {
	return c.Fake.
		$if .namespaced$InvokesWatch($.NewWatchActionWithOptions|raw$(c.Resource(), c.Namespace(), opts))
		$else$InvokesWatch($.NewRootWatchActionWithOptions|raw$(c.Resource(), opts))$end$
}
`

var patchTemplate = `
// Patch applies the patch and returns the patched $.resultType|private$.
func (c *fake$.type|publicPlural$) Patch(ctx $.contextContext|raw$, name string, pt $.PatchType|raw$, data []byte, opts $.PatchOptions|raw$, subresources ...string) (result *$.resultType|raw$, err error) {
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewPatchSubresourceActionWithOptions|raw$(c.Resource(), c.Namespace(), name, pt, data, opts, subresources... ), emptyResult)
		$else$Invokes($.NewRootPatchSubresourceActionWithOptions|raw$(c.Resource(), name, pt, data, opts, subresources...), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var applyTemplate = `
// Apply takes the given apply declarative configuration, applies it and returns the applied $.resultType|private$.
func (c *fake$.type|publicPlural$) Apply(ctx $.contextContext|raw$, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error) {
	if $.inputType|private$ == nil {
		return nil, $.fmtErrorf|raw$("$.inputType|private$ provided to Apply must not be nil")
	}
	data, err := $.jsonMarshal|raw$($.inputType|private$)
	if err != nil {
		return nil, err
	}
	name := $.inputType|private$.Name
	if name == nil {
		return nil, $.fmtErrorf|raw$("$.inputType|private$.Name must be provided to Apply")
	}
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewPatchSubresourceActionWithOptions|raw$(c.Resource(), c.Namespace(), *name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions()), emptyResult)
		$else$Invokes($.NewRootPatchSubresourceActionWithOptions|raw$(c.Resource(), *name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions()), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`

var applySubresourceTemplate = `
// Apply takes top resource name and the apply declarative configuration for $.subresourcePath$,
// applies it and returns the applied $.resultType|private$, and an error, if there is any.
func (c *fake$.type|publicPlural$) Apply(ctx $.contextContext|raw$, $.type|private$Name string, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error) {
	if $.inputType|private$ == nil {
		return nil, $.fmtErrorf|raw$("$.inputType|private$ provided to Apply must not be nil")
	}
	data, err := $.jsonMarshal|raw$($.inputType|private$)
	if err != nil {
		return nil, err
	}
	emptyResult := &$.resultType|raw${}
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewPatchSubresourceActionWithOptions|raw$(c.Resource(), c.Namespace(), $.type|private$Name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions(), "$.inputType|private$"), emptyResult)
		$else$Invokes($.NewRootPatchSubresourceActionWithOptions|raw$(c.Resource(), $.type|private$Name, $.ApplyPatchType|raw$, data, opts.ToPatchOptions(), "$.inputType|private$"), emptyResult)$end$
	if obj == nil {
		return emptyResult, err
	}
	return obj.(*$.resultType|raw$), err
}
`
