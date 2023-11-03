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
	"path"
	"strings"

	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
)

// genClientForType produces a file for each top-level type.
type genClientForType struct {
	generator.GoGenerator
	outputPackage             string // must be a Go import-path
	inputPackage              string
	clientsetPackage          string // must be a Go import-path
	applyConfigurationPackage string // must be a Go import-path
	group                     string
	version                   string
	groupGoName               string
	typeToMatch               *types.Type
	imports                   namer.ImportTracker
}

var _ generator.Generator = &genClientForType{}

// Filter ignores all but one type because we're making a single file per type.
func (g *genClientForType) Filter(c *generator.Context, t *types.Type) bool {
	return t == g.typeToMatch
}

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
	generateApply := len(g.applyConfigurationPackage) > 0
	genericVerbTemplates := buildGenericVerbTemplates(generateApply)
	defaultVerbTemplates := buildDefaultVerbTemplates(generateApply)
	subresourceDefaultVerbTemplates := buildSubresourceDefaultVerbTemplates(generateApply)
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	pkg := path.Base(t.Name.Package)
	tags, err := util.ParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
	if err != nil {
		return err
	}
	type extendedInterfaceMethod struct {
		template string
		args     map[string]interface{}
	}
	_, typeGVString := util.ParsePathGroupVersion(g.inputPackage)
	extendedMethods := []extendedInterfaceMethod{}
	for _, e := range tags.Extensions {
		if e.HasVerb("apply") && !generateApply {
			continue
		}
		inputType := *t
		resultType := *t
		inputGVString := typeGVString
		// TODO: Extract this to some helper method as this code is copied into
		// 2 other places.
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
		var updatedVerbtemplate string
		if _, exists := subresourceDefaultVerbTemplates[e.VerbType]; e.IsSubresource() && exists {
			//nolint:staticcheck
			// TODO: convert this to use golang.org/x/text/cases
			updatedVerbtemplate = e.VerbName + "(" + strings.TrimPrefix(subresourceDefaultVerbTemplates[e.VerbType], strings.Title(e.VerbType)+"(")
		} else {
			//nolint:staticcheck
			// TODO: convert this to use golang.org/x/text/cases
			updatedVerbtemplate = e.VerbName + "(" + strings.TrimPrefix(defaultVerbTemplates[e.VerbType], strings.Title(e.VerbType)+"(")
		}
		extendedMethod := extendedInterfaceMethod{
			template: updatedVerbtemplate,
			args: map[string]interface{}{
				"type":          t,
				"inputType":     &inputType,
				"resultType":    &resultType,
				"CreateOptions": c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "CreateOptions"}),
				"GetOptions":    c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "GetOptions"}),
				"ListOptions":   c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ListOptions"}),
				"UpdateOptions": c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "UpdateOptions"}),
				"ApplyOptions":  c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ApplyOptions"}),
				"PatchType":     c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "PatchType"}),
				"jsonMarshal":   c.Universe.Type(types.Name{Package: "encoding/json", Name: "Marshal"}),
			},
		}
		if e.HasVerb("apply") {
			extendedMethod.args["inputApplyConfig"] = types.Ref(path.Join(g.applyConfigurationPackage, inputGVString), inputType.Name.Name+"ApplyConfiguration")
		}
		extendedMethods = append(extendedMethods, extendedMethod)
	}
	m := map[string]interface{}{
		"type":                       t,
		"inputType":                  t,
		"resultType":                 t,
		"package":                    pkg,
		"Package":                    namer.IC(pkg),
		"namespaced":                 !tags.NonNamespaced,
		"Group":                      namer.IC(g.group),
		"subresource":                false,
		"subresourcePath":            "",
		"GroupGoName":                g.groupGoName,
		"Version":                    namer.IC(g.version),
		"CreateOptions":              c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "CreateOptions"}),
		"DeleteOptions":              c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "DeleteOptions"}),
		"GetOptions":                 c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "GetOptions"}),
		"ListOptions":                c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ListOptions"}),
		"PatchOptions":               c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "PatchOptions"}),
		"ApplyOptions":               c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ApplyOptions"}),
		"UpdateOptions":              c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "UpdateOptions"}),
		"PatchType":                  c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "PatchType"}),
		"ApplyPatchType":             c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "ApplyPatchType"}),
		"watchInterface":             c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/watch", Name: "Interface"}),
		"RESTClientInterface":        c.Universe.Type(types.Name{Package: "k8s.io/client-go/rest", Name: "Interface"}),
		"schemeParameterCodec":       c.Universe.Variable(types.Name{Package: path.Join(g.clientsetPackage, "scheme"), Name: "ParameterCodec"}),
		"jsonMarshal":                c.Universe.Type(types.Name{Package: "encoding/json", Name: "Marshal"}),
		"TypeClient":                 c.Universe.Type(types.Name{Package: "k8s.io/client-go/generic", Name: "TypeClient"}),
		"TypeClientWithList":         c.Universe.Type(types.Name{Package: "k8s.io/client-go/generic", Name: "TypeClientWithList"}),
		"TypeClientWithApply":        c.Universe.Type(types.Name{Package: "k8s.io/client-go/generic", Name: "TypeClientWithApply"}),
		"TypeClientWithListAndApply": c.Universe.Type(types.Name{Package: "k8s.io/client-go/generic", Name: "TypeClientWithListAndApply"}),
		"Interface":                  c.Universe.Type(types.Name{Package: "k8s.io/client-go/generic", Name: "Interface"}),
		"InterfaceWithApply":         c.Universe.Type(types.Name{Package: "k8s.io/client-go/generic", Name: "InterfaceWithApply"}),
	}

	if generateApply {
		// Generated apply configuration type references required for generated Apply function
		_, gvString := util.ParsePathGroupVersion(g.inputPackage)
		m["inputApplyConfig"] = types.Ref(path.Join(g.applyConfigurationPackage, gvString), t.Name.Name+"ApplyConfiguration")
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
			tags.SkipVerbs = append(tags.SkipVerbs, "applyStatus")
		}
		interfaceSuffix := ""
		if len(extendedMethods) > 0 {
			interfaceSuffix = "\n"
		}
		if hasAllDefaultVerbs(tags) {
			sw.Do(commonInterface, m)
			skipDefaultVerbs(genericVerbTemplates)
		}
		sw.Do("\n"+generateInterface(genericVerbTemplates, tags)+interfaceSuffix, m)
		// add extended verbs into interface
		for _, v := range extendedMethods {
			sw.Do(v.template+interfaceSuffix, v.args)
		}

	}
	sw.Do(interfaceTemplate4, m)

	if tags.NoVerbs {
		sw.Do(structType, m)
		if tags.NonNamespaced {
			sw.Do(newStructNonNamespaced, m)
		} else {
			sw.Do(newStructNamespaced, m)
		}

		return sw.Error()
	}

	if !tags.HasVerb("list") {
		if !tags.HasVerb("apply") || !generateApply {
			sw.Do(structType, m)
			if tags.NonNamespaced {
				sw.Do(newStructNonNamespaced, m)
			} else {
				sw.Do(newStructNamespaced, m)
			}
		} else {
			sw.Do(structTypeWithApply, m)
			if tags.NonNamespaced {
				sw.Do(newStructNonNamespacedWithApply, m)
			} else {
				sw.Do(newStructNamespacedWithApply, m)
			}
		}
	} else {
		if !tags.HasVerb("apply") || !generateApply {
			sw.Do(structTypeWithList, m)
			if tags.NonNamespaced {
				sw.Do(newStructNonNamespacedWithList, m)
			} else {
				sw.Do(newStructNamespacedWithList, m)
			}
		} else {
			sw.Do(structTypeWithListAndApply, m)
			if tags.NonNamespaced {
				sw.Do(newStructNonNamespacedWithListAndApply, m)
			} else {
				sw.Do(newStructNamespacedWithListAndApply, m)
			}
		}
	}

	// generate expansion methods
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

		if e.HasVerb("apply") {
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
	//nolint:staticcheck
	// TODO: convert this to use golang.org/x/text/cases
	return strings.ReplaceAll(template, " "+strings.Title(verbType), " "+name)
}

func generateInterface(defaultVerbTemplates map[string]string, tags util.Tags) string {
	// need an ordered list here to guarantee order of generated methods.
	out := []string{}
	for _, m := range util.SupportedVerbs {
		if tags.HasVerb(m) && len(defaultVerbTemplates[m]) > 0 {
			out = append(out, defaultVerbTemplates[m])
		}
	}
	return strings.Join(out, "\n")
}

func buildSubresourceDefaultVerbTemplates(generateApply bool) map[string]string {
	m := map[string]string{
		"create": `Create(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (*$.resultType|raw$, error)`,
		"list":   `List(ctx context.Context, $.type|private$Name string, opts $.ListOptions|raw$) (*$.resultType|raw$List, error)`,
		"update": `Update(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (*$.resultType|raw$, error)`,
		"get":    `Get(ctx context.Context, $.type|private$Name string, options $.GetOptions|raw$) (*$.resultType|raw$, error)`,
	}
	if generateApply {
		m["apply"] = `Apply(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (*$.resultType|raw$, error)`
	}
	return m
}

func buildDefaultVerbTemplates(generateApply bool) map[string]string {
	m := map[string]string{
		"create":           `Create(ctx context.Context, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (*$.resultType|raw$, error)`,
		"update":           `Update(ctx context.Context, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (*$.resultType|raw$, error)`,
		"updateStatus":     `UpdateStatus(ctx context.Context, $.inputType|private$ *$.type|raw$, opts $.UpdateOptions|raw$) (*$.type|raw$, error)`,
		"delete":           `Delete(ctx context.Context, name string, opts $.DeleteOptions|raw$) error`,
		"deleteCollection": `DeleteCollection(ctx context.Context, opts $.DeleteOptions|raw$, listOpts $.ListOptions|raw$) error`,
		"get":              `Get(ctx context.Context, name string, opts $.GetOptions|raw$) (*$.resultType|raw$, error)`,
		"list":             `List(ctx context.Context, opts $.ListOptions|raw$) (*$.resultType|raw$List, error)`,
		"watch":            `Watch(ctx context.Context, opts $.ListOptions|raw$) ($.watchInterface|raw$, error)`,
		"patch":            `Patch(ctx context.Context, name string, pt $.PatchType|raw$, data []byte, opts $.PatchOptions|raw$, subresources ...string) (result *$.resultType|raw$, err error)`,
	}
	if generateApply {
		m["apply"] = `Apply(ctx context.Context, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error)`
		m["applyStatus"] = `ApplyStatus(ctx context.Context, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error)`
	}
	return m
}

func buildGenericVerbTemplates(generateApply bool) map[string]string {
	m := map[string]string{
		"create": `generic.Creator[*$.resultType|raw$]`,
		"update": `generic.Updater[*$.resultType|raw$]`,
		"updateStatus": `// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
generic.StatusUpdater[*$.type|raw$]`,
		"delete":           `generic.Deleter`,
		"deleteCollection": `generic.CollectionDeleter`,
		"get":              `generic.Getter[*$.resultType|raw$]`,
		"list":             `generic.Lister[*$.resultType|raw$, *$.resultType|raw$List]`,
		"watch":            `generic.Watcher`,
		"patch":            `generic.Patcher[*$.resultType|raw$]`,
	}
	if generateApply {
		m["apply"] = `generic.Applier[*$.resultType|raw$, *$.inputApplyConfig|raw$]`
		m["applyStatus"] = `// Add a +genclient:noStatus comment above the type to avoid generating ApplyStatus().
generic.StatusApplier[*$.resultType|raw$, *$.inputApplyConfig|raw$]`
	}
	return m
}

func hasAllDefaultVerbs(tags util.Tags) bool {
	// This must match the basic Interface in generic/type.go
	for _, verb := range []string{"create", "update", "delete", "deleteCollection", "get", "list", "watch", "patch"} {
		if !tags.HasVerb(verb) {
			return false
		}
	}

	return true
}

func skipDefaultVerbs(verbTemplates map[string]string) {
	for _, verb := range []string{"create", "update", "delete", "deleteCollection", "get", "list", "watch", "patch"} {
		delete(verbTemplates, verb)
	}
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

var commonInterface = `
	generic.Interface[*$.resultType|raw$, *$.resultType|raw$List]`

var interfaceTemplate4 = `
	$.type|public$Expansion
}
`

// template for the struct that implements the type's interface
var structType = `
// $.type|privatePlural$ implements $.type|public$Interface
type $.type|privatePlural$ struct {
	*$.TypeClient|raw$[*$.resultType|raw$]
}
`

var structTypeWithList = `
// $.type|privatePlural$ implements $.type|public$Interface
type $.type|privatePlural$ struct {
	*$.TypeClientWithList|raw$[*$.resultType|raw$, *$.resultType|raw$List]
}
`

var structTypeWithApply = `
// $.type|privatePlural$ implements $.type|public$Interface
type $.type|privatePlural$ struct {
	*$.TypeClientWithApply|raw$[*$.resultType|raw$, *$.inputApplyConfig|raw$]
}
`

var structTypeWithListAndApply = `
// $.type|privatePlural$ implements $.type|public$Interface
type $.type|privatePlural$ struct {
	*$.TypeClientWithListAndApply|raw$[*$.resultType|raw$, *$.resultType|raw$List, *$.inputApplyConfig|raw$]
}
`

var newStructNamespaced = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client, namespace string) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		generic.NewNamespaced[*$.resultType|raw$](
			"$.type|resource$",
			c.RESTClient(),
			$.schemeParameterCodec|raw$,
			namespace,
			func() *$.resultType|raw$ { return &$.resultType|raw${} }),
	}
}
`

var newStructNamespacedWithList = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client, namespace string) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		generic.NewNamespacedWithList[*$.resultType|raw$, *$.resultType|raw$List](
			"$.type|resource$",
			c.RESTClient(),
			$.schemeParameterCodec|raw$,
			namespace,
			func() *$.resultType|raw$ { return &$.resultType|raw${} },
			func() *$.resultType|raw$List { return &$.resultType|raw$List{} }),
	}
}
`

var newStructNamespacedWithApply = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client, namespace string) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		generic.NewNamespacedWithApply[*$.resultType|raw$, *$.inputApplyConfig|raw$](
			"$.type|resource$",
			c.RESTClient(),
			$.schemeParameterCodec|raw$,
			namespace,
			func() *$.resultType|raw$ { return &$.resultType|raw${} }),
	}
}
`

var newStructNamespacedWithListAndApply = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client, namespace string) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		generic.NewNamespacedWithListAndApply[*$.resultType|raw$, *$.resultType|raw$List, *$.inputApplyConfig|raw$](
			"$.type|resource$",
			c.RESTClient(),
			$.schemeParameterCodec|raw$,
			namespace,
			func() *$.resultType|raw$ { return &$.resultType|raw${} },
			func() *$.resultType|raw$List { return &$.resultType|raw$List{} }),
	}
}
`

var newStructNonNamespaced = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		generic.NewNonNamespaced[*$.resultType|raw$](
			"$.type|resource$",
			c.RESTClient(),
			$.schemeParameterCodec|raw$,
			func() *$.resultType|raw$ { return &$.resultType|raw${} }),
	}
}
`

var newStructNonNamespacedWithList = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		generic.NewNonNamespacedWithList[*$.resultType|raw$, *$.resultType|raw$List](
			"$.type|resource$",
			c.RESTClient(),
			$.schemeParameterCodec|raw$,
			func() *$.resultType|raw$ { return &$.resultType|raw${} },
			func() *$.resultType|raw$List { return &$.resultType|raw$List{} }),
	}
}
`

var newStructNonNamespacedWithApply = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		generic.NewNonNamespacedWithApply[*$.resultType|raw$, *$.inputApplyConfig|raw$](
			"$.type|resource$",
			c.RESTClient(),
			$.schemeParameterCodec|raw$,
			func() *$.resultType|raw$ { return &$.resultType|raw${} }),
	}
}
`

var newStructNonNamespacedWithListAndApply = `
// new$.type|publicPlural$ returns a $.type|publicPlural$
func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client) *$.type|privatePlural$ {
	return &$.type|privatePlural${
		generic.NewNonNamespacedWithListAndApply[*$.resultType|raw$, *$.resultType|raw$List, *$.inputApplyConfig|raw$](
			"$.type|resource$",
			c.RESTClient(),
			$.schemeParameterCodec|raw$,
			func() *$.resultType|raw$ { return &$.resultType|raw${} },
			func() *$.resultType|raw$List { return &$.resultType|raw$List{} }),
	}
}
`

var listTemplate = `
// List takes label and field selectors, and returns the list of $.resultType|publicPlural$ that match those selectors.
func (c *$.type|privatePlural$) List(ctx context.Context, opts $.ListOptions|raw$) (result *$.resultType|raw$List, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil{
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &$.resultType|raw$List{}
	err = c.Client.Get().
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Timeout(timeout).
		Do(ctx).
		Into(result)
	return
}
`

var listSubresourceTemplate = `
// List takes $.type|raw$ name, label and field selectors, and returns the list of $.resultType|publicPlural$ that match those selectors.
func (c *$.type|privatePlural$) List(ctx context.Context, $.type|private$Name string, opts $.ListOptions|raw$) (result *$.resultType|raw$List, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil{
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &$.resultType|raw$List{}
	err = c.Client.Get().
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name($.type|private$Name).
		SubResource("$.subresourcePath$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Timeout(timeout).
		Do(ctx).
		Into(result)
	return
}
`

var getTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.resultType|private$ object, and an error if there is any.
func (c *$.type|privatePlural$) Get(ctx context.Context, name string, options $.GetOptions|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.Client.Get().
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name(name).
		VersionedParams(&options, $.schemeParameterCodec|raw$).
		Do(ctx).
		Into(result)
	return
}
`

var getSubresourceTemplate = `
// Get takes name of the $.type|private$, and returns the corresponding $.resultType|raw$ object, and an error if there is any.
func (c *$.type|privatePlural$) Get(ctx context.Context, $.type|private$Name string, options $.GetOptions|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.Client.Get().
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name($.type|private$Name).
		SubResource("$.subresourcePath$").
		VersionedParams(&options, $.schemeParameterCodec|raw$).
		Do(ctx).
		Into(result)
	return
}
`

var deleteTemplate = `
// Delete takes name of the $.type|private$ and deletes it. Returns an error if one occurs.
func (c *$.type|privatePlural$) Delete(ctx context.Context, name string, opts $.DeleteOptions|raw$) error {
	return c.Client.Delete().
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name(name).
		Body(&opts).
		Do(ctx).
		Error()
}
`

var createSubresourceTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Create(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.Client.Post().
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name($.type|private$Name).
		SubResource("$.subresourcePath$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Body($.inputType|private$).
		Do(ctx).
		Into(result)
	return
}
`

var createTemplate = `
// Create takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Create(ctx context.Context, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.Client.Post().
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Body($.inputType|private$).
		Do(ctx).
		Into(result)
	return
}
`

var updateSubresourceTemplate = `
// Update takes the top resource name and the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Update(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.Client.Put().
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name($.type|private$Name).
		SubResource("$.subresourcePath$").
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Body($.inputType|private$).
		Do(ctx).
		Into(result)
	return
}
`

var updateTemplate = `
// Update takes the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Update(ctx context.Context, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.Client.Put().
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name($.inputType|private$.Name).
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Body($.inputType|private$).
		Do(ctx).
		Into(result)
	return
}
`

var watchTemplate = `
// Watch returns a $.watchInterface|raw$ that watches the requested $.type|privatePlural$.
func (c *$.type|privatePlural$) Watch(ctx context.Context, opts $.ListOptions|raw$) ($.watchInterface|raw$, error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil{
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.Client.Get().
		$if .namespaced$Namespace(c.Namespace).$end$
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Timeout(timeout).
		Watch(ctx)
}
`

var patchTemplate = `
// Patch applies the patch and returns the patched $.resultType|private$.
func (c *$.type|privatePlural$) Patch(ctx context.Context, name string, pt $.PatchType|raw$, data []byte, opts $.PatchOptions|raw$, subresources ...string) (result *$.resultType|raw$, err error) {
	result = &$.resultType|raw${}
	err = c.Client.Patch(pt).
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name(name).
		SubResource(subresources...).
		VersionedParams(&opts, $.schemeParameterCodec|raw$).
		Body(data).
		Do(ctx).
		Into(result)
	return
}
`

var applyTemplate = `
// Apply takes the given apply declarative configuration, applies it and returns the applied $.resultType|private$.
func (c *$.type|privatePlural$) Apply(ctx context.Context, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error) {
	if $.inputType|private$ == nil {
		return nil, fmt.Errorf("$.inputType|private$ provided to Apply must not be nil")
	}
	patchOpts := opts.ToPatchOptions()
	data, err := $.jsonMarshal|raw$($.inputType|private$)
	if err != nil {
		return nil, err
	}
    name := $.inputType|private$.Name
	if name == nil {
		return nil, fmt.Errorf("$.inputType|private$.Name must be provided to Apply")
	}
	result = &$.resultType|raw${}
	err = c.Client.Patch($.ApplyPatchType|raw$).
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name(*name).
		VersionedParams(&patchOpts, $.schemeParameterCodec|raw$).
		Body(data).
		Do(ctx).
		Into(result)
	return
}
`

var applySubresourceTemplate = `
// Apply takes top resource name and the apply declarative configuration for $.subresourcePath$,
// applies it and returns the applied $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) Apply(ctx context.Context, $.type|private$Name string, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error) {
	if $.inputType|private$ == nil {
		return nil, fmt.Errorf("$.inputType|private$ provided to Apply must not be nil")
	}
	patchOpts := opts.ToPatchOptions()
	data, err := $.jsonMarshal|raw$($.inputType|private$)
	if err != nil {
		return nil, err
	}

	result = &$.resultType|raw${}
	err = c.Client.Patch($.ApplyPatchType|raw$).
		$if .namespaced$Namespace(c.Namespace).$end$
		Resource("$.type|resource$").
		Name($.type|private$Name).
		SubResource("$.subresourcePath$").
		VersionedParams(&patchOpts, $.schemeParameterCodec|raw$).
		Body(data).
		Do(ctx).
		Into(result)
	return
}
`
