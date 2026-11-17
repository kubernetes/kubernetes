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

	"golang.org/x/text/cases"
	"golang.org/x/text/language"

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
	prefersProtobuf           bool
	typeToMatch               *types.Type
	imports                   namer.ImportTracker
}

var _ generator.Generator = &genClientForType{}

var titler = cases.Title(language.Und)

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
			updatedVerbtemplate = e.VerbName + "(" + strings.TrimPrefix(subresourceDefaultVerbTemplates[e.VerbType], titler.String(e.VerbType)+"(")
		} else {
			updatedVerbtemplate = e.VerbName + "(" + strings.TrimPrefix(defaultVerbTemplates[e.VerbType], titler.String(e.VerbType)+"(")
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
				"PatchOptions":  c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "PatchOptions"}),
				"context":       c.Universe.Type(types.Name{Package: "context", Name: "Context"}),
			},
		}
		if e.HasVerb("apply") {
			extendedMethod.args["inputApplyConfig"] = types.Ref(path.Join(g.applyConfigurationPackage, inputGVString), inputType.Name.Name+"ApplyConfiguration")
		}
		extendedMethods = append(extendedMethods, extendedMethod)
	}
	m := map[string]interface{}{
		"type":                           t,
		"inputType":                      t,
		"resultType":                     t,
		"package":                        pkg,
		"Package":                        namer.IC(pkg),
		"namespaced":                     !tags.NonNamespaced,
		"Group":                          namer.IC(g.group),
		"subresource":                    false,
		"subresourcePath":                "",
		"GroupGoName":                    g.groupGoName,
		"prefersProtobuf":                g.prefersProtobuf,
		"Version":                        namer.IC(g.version),
		"runtimeObject":                  c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "Object"}),
		"APIResult":                      c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "APIResult"}),
		"CreateOptions":                  c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "CreateOptions"}),
		"DeleteOptions":                  c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "DeleteOptions"}),
		"GetOptions":                     c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "GetOptions"}),
		"ListOptions":                    c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ListOptions"}),
		"PatchOptions":                   c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "PatchOptions"}),
		"ApplyOptions":                   c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ApplyOptions"}),
		"UpdateOptions":                  c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "UpdateOptions"}),
		"PatchType":                      c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/types", Name: "PatchType"}),
		"watchInterface":                 c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/watch", Name: "Interface"}),
		"RESTClientInterface":            c.Universe.Type(types.Name{Package: "k8s.io/client-go/rest", Name: "Interface"}),
		"schemeParameterCodec":           c.Universe.Variable(types.Name{Package: path.Join(g.clientsetPackage, "scheme"), Name: "ParameterCodec"}),
		"fmtErrorf":                      c.Universe.Function(types.Name{Package: "fmt", Name: "Errorf"}),
		"klogWarningf":                   c.Universe.Function(types.Name{Package: "k8s.io/klog/v2", Name: "Warningf"}),
		"context":                        c.Universe.Type(types.Name{Package: "context", Name: "Context"}),
		"timeDuration":                   c.Universe.Type(types.Name{Package: "time", Name: "Duration"}),
		"timeSecond":                     c.Universe.Type(types.Name{Package: "time", Name: "Second"}),
		"applyNewRequest":                c.Universe.Function(types.Name{Package: "k8s.io/client-go/util/apply", Name: "NewRequest"}),
		"TypedClient":                    c.Universe.Type(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "TypedClient"}),
		"TypedClientWithList":            c.Universe.Type(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "TypedClientWithList"}),
		"TypedClientWithApply":           c.Universe.Type(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "TypedClientWithApply"}),
		"TypedClientWithListAndApply":    c.Universe.Type(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "TypedClientWithListAndApply"}),
		"NewTypedClient":                 c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "NewTypedClient"}),
		"NewTypedClientWithApply":        c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "NewTypedClientWithApply"}),
		"NewTypedClientWithList":         c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "NewTypedClientWithList"}),
		"NewTypedClientWithListAndApply": c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "NewTypedClientWithListAndApply"}),
		"ForResource":                    c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "ForResource"}),
		"InNamespace":                    c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "InNamespace"}),
		"PrefersProtobuf":                c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "PrefersProtobuf"}),
		"UsingClient":                    c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "UsingClient"}),
		"UsingParameterCodec":            c.Universe.Function(types.Name{Package: "k8s.io/client-go/gentype/v2", Name: "UsingParameterCodec"}),
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
		sw.Do("\n"+generateInterface(defaultVerbTemplates, tags)+interfaceSuffix, m)
		// add extended verbs into interface
		for _, v := range extendedMethods {
			sw.Do(v.template+interfaceSuffix, v.args)
		}

	}
	sw.Do(interfaceTemplate4, m)

	if tags.NoVerbs {
		sw.Do(structType[noList|noApply], m)
		sw.Do(newStruct[noList|noApply], m)

		return sw.Error()
	}

	listableOrAppliable := noList | noApply

	if tags.HasVerb("list") {
		listableOrAppliable |= withList
	}

	if tags.HasVerb("apply") && generateApply {
		listableOrAppliable |= withApply
	}

	sw.Do(structType[listableOrAppliable], m)
	sw.Do(newStruct[listableOrAppliable], m)

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
		m["verb"] = e.VerbName
		if e.HasVerb("apply") {
			m["inputApplyConfig"] = types.Ref(path.Join(g.applyConfigurationPackage, inputGVString), inputType.Name.Name+"ApplyConfiguration")
		}

		if e.HasVerb("get") {
			if e.IsSubresource() {
				sw.Do(getSubresourceTemplate, m)
			} else {
				sw.Do(getTemplate, m)
			}
		}

		if e.HasVerb("list") {
			if e.IsSubresource() {
				sw.Do(listSubresourceTemplate, m)
			} else {
				sw.Do(listTemplate, m)
			}
		}

		// TODO: Figure out schemantic for watching a sub-resource.
		if e.HasVerb("watch") {
			sw.Do(watchTemplate, m)
		}

		if e.HasVerb("create") {
			if e.IsSubresource() {
				sw.Do(createSubresourceTemplate, m)
			} else {
				sw.Do(createTemplate, m)
			}
		}

		if e.HasVerb("update") {
			if e.IsSubresource() {
				sw.Do(updateSubresourceTemplate, m)
			} else {
				sw.Do(updateTemplate, m)
			}
		}

		// TODO: Figure out schemantic for deleting a sub-resource (what arguments
		// are passed, does it need two names? etc.
		if e.HasVerb("delete") {
			sw.Do(deleteTemplate, m)
		}

		if e.HasVerb("patch") {
			sw.Do(patchTemplate, m)
		}

		if e.HasVerb("apply") {
			if e.IsSubresource() {
				sw.Do(applySubresourceTemplate, m)
			} else {
				sw.Do(applyTemplate, m)
			}
		}
	}

	return sw.Error()
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
		"create": `Create(ctx $.context|raw$, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (*$.resultType|raw$, error)`,
		"list":   `List(ctx $.context|raw$, $.type|private$Name string, opts $.ListOptions|raw$) (*$.resultType|raw$List, error)`,
		"update": `Update(ctx $.context|raw$, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (*$.resultType|raw$, error)`,
		"get":    `Get(ctx $.context|raw$, $.type|private$Name string, opts $.GetOptions|raw$) (*$.resultType|raw$, error)`,
	}
	if generateApply {
		m["apply"] = `Apply(ctx $.context|raw$, $.type|private$Name string, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (*$.resultType|raw$, error)`
	}
	return m
}

func buildDefaultVerbTemplates(generateApply bool) map[string]string {
	m := map[string]string{
		"create": `Create(ctx $.context|raw$, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (*$.resultType|raw$, error)`,
		"update": `Update(ctx $.context|raw$, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (*$.resultType|raw$, error)`,
		"updateStatus": `// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
UpdateStatus(ctx $.context|raw$, $.inputType|private$ *$.type|raw$, opts $.UpdateOptions|raw$) (*$.type|raw$, error)`,
		"delete":           `Delete(ctx $.context|raw$, name string, opts $.DeleteOptions|raw$) error`,
		"deleteWithResult": `DeleteWithResult(ctx $.context|raw$, name string, opts $.DeleteOptions|raw$) ($.APIResult|raw$, error)`,
		"deleteCollection": `DeleteCollection(ctx $.context|raw$, opts $.DeleteOptions|raw$, listOpts $.ListOptions|raw$) error`,
		"get":              `Get(ctx $.context|raw$, name string, opts $.GetOptions|raw$) (*$.resultType|raw$, error)`,
		"list":             `List(ctx $.context|raw$, opts $.ListOptions|raw$) (*$.resultType|raw$List, error)`,
		"watch":            `Watch(ctx $.context|raw$, opts $.ListOptions|raw$) ($.watchInterface|raw$, error)`,
		"patch":            `Patch(ctx $.context|raw$, name string, pt $.PatchType|raw$, data []byte, opts $.PatchOptions|raw$, subresources ...string) (result *$.resultType|raw$, err error)`,
	}
	if generateApply {
		m["apply"] = `Apply(ctx $.context|raw$, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error)`
		m["applyStatus"] = `// Add a +genclient:noStatus comment above the type to avoid generating ApplyStatus().
ApplyStatus(ctx $.context|raw$, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (result *$.resultType|raw$, err error)`
	}
	return m
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

// struct and constructor variants
const (
	// The following values are bits in a bitmask.
	// The values which can be set indicate namespace support, list support, and apply support;
	// to make the declarations easier to read (like a truth table), corresponding zero-values
	// are also declared.
	noList   = 0
	noApply  = 0
	withList = 1 << iota
	withApply
)

// The following string slices are similar to maps, but with combinable keys used as indices.
// Each entry defines whether it supports lists and/or apply, and if namespacedness matters,
// namespaces; each bit is then toggled:
// * noList, noApply: index 0;
// * withList, noApply: index 2;
// * noList, withApply: index 4;
// * withList, withApply: index 6.
// Go enforces index unicity in these kinds of declarations.

// struct declarations
var structType = []string{
	noList | noApply: `
	// $.type|privatePlural$ implements $.type|public$Interface
	type $.type|privatePlural$ struct {
		*$.TypedClient|raw$[$.resultType|raw$, *$.resultType|raw$]
	}
	`,
	withList | noApply: `
	// $.type|privatePlural$ implements $.type|public$Interface
	type $.type|privatePlural$ struct {
		*$.TypedClientWithList|raw$[$.resultType|raw$, $.resultType|raw$List, *$.resultType|raw$, *$.resultType|raw$List]
	}
	`,
	noList | withApply: `
	// $.type|privatePlural$ implements $.type|public$Interface
	type $.type|privatePlural$ struct {
		*$.TypedClientWithApply|raw$[$.resultType|raw$, $.inputApplyConfig|raw$, *$.resultType|raw$, *$.inputApplyConfig|raw$]
	}
	`,
	withList | withApply: `
	// $.type|privatePlural$ implements $.type|public$Interface
	type $.type|privatePlural$ struct {
		*$.TypedClientWithListAndApply|raw$[$.resultType|raw$, $.resultType|raw$List, $.inputApplyConfig|raw$, *$.resultType|raw$, *$.resultType|raw$List, *$.inputApplyConfig|raw$]
	}
	`,
}

// Constructors for the struct, in all variants
var newStruct = []string{
	noList | noApply: `
	// new$.type|publicPlural$ returns a $.type|publicPlural$
	func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client$if .namespaced$, namespace string$end$) *$.type|privatePlural$ {
		return &$.type|privatePlural${
			$.NewTypedClient|raw$[$.resultType|raw$](
				$.ForResource|raw$("$.type|resource$"),
				$.UsingClient|raw$(c.RESTClient()),
				$.UsingParameterCodec|raw$($.schemeParameterCodec|raw$),$if .namespaced$
				$.InNamespace|raw$(namespace),$end$$if .prefersProtobuf$
				$.PrefersProtobuf|raw$(),$end$
			),
		}
	}
	`,
	noList | withApply: `
	// new$.type|publicPlural$ returns a $.type|publicPlural$
	func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client$if .namespaced$, namespace string$end$) *$.type|privatePlural$ {
		return &$.type|privatePlural${
			$.NewTypedClientWithApply|raw$[$.resultType|raw$, $.inputApplyConfig|raw$](
				$.ForResource|raw$("$.type|resource$"),
				$.UsingClient|raw$(c.RESTClient()),
				$.UsingParameterCodec|raw$($.schemeParameterCodec|raw$),$if .namespaced$
				$.InNamespace|raw$(namespace),$end$$if .prefersProtobuf$
				$.PrefersProtobuf|raw$(),$end$
			),
		}
	}
	`,
	withList | noApply: `
	// new$.type|publicPlural$ returns a $.type|publicPlural$
	func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client$if .namespaced$, namespace string$end$) *$.type|privatePlural$ {
		return &$.type|privatePlural${
			$.NewTypedClientWithList|raw$[$.resultType|raw$, $.resultType|raw$List](
				$.ForResource|raw$("$.type|resource$"),
				$.UsingClient|raw$(c.RESTClient()),
				$.UsingParameterCodec|raw$($.schemeParameterCodec|raw$),$if .namespaced$
				$.InNamespace|raw$(namespace),$end$$if .prefersProtobuf$
				$.PrefersProtobuf|raw$(),$end$
			),
		}
	}
	`,
	withList | withApply: `
	// new$.type|publicPlural$ returns a $.type|publicPlural$
	func new$.type|publicPlural$(c *$.GroupGoName$$.Version$Client$if .namespaced$, namespace string$end$) *$.type|privatePlural$ {
		return &$.type|privatePlural${
			$.NewTypedClientWithListAndApply|raw$[$.resultType|raw$, $.resultType|raw$List, $.inputApplyConfig|raw$](
				$.ForResource|raw$("$.type|resource$"),
				$.UsingClient|raw$(c.RESTClient()),
				$.UsingParameterCodec|raw$($.schemeParameterCodec|raw$),$if .namespaced$
				$.InNamespace|raw$(namespace),$end$$if .prefersProtobuf$
				$.PrefersProtobuf|raw$(),$end$
			),
		}
	}
	`,
}

var listTemplate = `
// $.verb$ takes label and field selectors, and returns the list of $.resultType|publicPlural$ that match those selectors.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, opts $.ListOptions|raw$) (*$.resultType|raw$List, error) {
	return c.Client.List[$.resultType|raw$List](ctx, opts)
}
`

var listSubresourceTemplate = `
// $.verb$ takes $.type|raw$ name, label and field selectors, and returns the list of $.resultType|publicPlural$ that match those selectors.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, $.type|private$Name string, opts $.ListOptions|raw$) (*$.resultType|raw$List, error) {
	return c.SubResourceList[$.resultType|raw$List](ctx, $.type|privateName$, "$.subresourcePath$", opts)
}
`

var getTemplate = `
// $.verb$ takes name of the $.type|private$, and returns the corresponding $.resultType|private$ object, and an error if there is any.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, name string, opts $.GetOptions|raw$) (*$.resultType|raw$, error) {
	return c.Client.Get[$.resultType|raw$](ctx, name, opts)
}
`

var getSubresourceTemplate = `
// $.verb$ takes name of the $.type|private$, and returns the corresponding $.resultType|raw$ object, and an error if there is any.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, $.type|private$Name string, opts $.GetOptions|raw$) (*$.resultType|raw$, error) {
	return c.SubResourceGet[$.resultType|raw$](ctx, $.type|private$Name, "$.subresourcePath$", opts)
}
`

var deleteTemplate = `
// $.verb$ takes name of the $.type|private$ and deletes it. Returns an error if one occurs.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, name string, opts $.DeleteOptions|raw$) error {
	return c.Client.Delete(ctx, name, opts)
}
`

var createSubresourceTemplate = `
// $.verb$ takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (*$.resultType|raw$, error) {
	return c.SubResourceCreate(ctx, $.type|private$Name, "$.subresourcePath$", $.inputType|private$, opts)
}
`

var createTemplate = `
// $.verb$ takes the representation of a $.inputType|private$ and creates it.  Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, $.inputType|private$ *$.inputType|raw$, opts $.CreateOptions|raw$) (*$.resultType|raw$, error) {
	return c.Client.Create(ctx, $.inputType|private$, opts)
}
`

var updateSubresourceTemplate = `
// $.verb$ takes the top resource name and the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, $.type|private$Name string, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (*$.resultType|raw$, error) {
	return c.SubResourceUpdate(ctx, $.type|private$Name, "$.subresourcePath$", $.inputType|private$, opts)
}
`

var updateTemplate = `
// $.verb$ takes the representation of a $.inputType|private$ and updates it. Returns the server's representation of the $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, $.inputType|private$ *$.inputType|raw$, opts $.UpdateOptions|raw$) (*$.resultType|raw$, error) {
	return c.Client.Update(ctx, $.inputType|private$, opts)
}
`

var watchTemplate = `
// $.verb$ returns a $.watchInterface|raw$ that watches the requested $.type|privatePlural$.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, opts $.ListOptions|raw$) ($.watchInterface|raw$, error) {
	return c.Client.Watch(ctx, opts)
}
`
var patchTemplate = `
// $.verb$ applies the patch and returns the patched $.resultType|private$.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, name string, pt $.PatchType|raw$, data []byte, opts $.PatchOptions|raw$, subresources ...string) (*$.resultType|raw$, error) {
	return c.Client.Patch[$.resultType|raw$](ctx, name, pt, data, opts, subresources...)
}
`

var applyTemplate = `
// $.verb$ takes the given apply declarative configuration, applies it and returns the applied $.resultType|private$.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (*$.resultType|raw$, error) {
	return c.Client.Apply[$.resultType|raw$](ctx, $.inputType|private$, opts)
}
`

var applySubresourceTemplate = `
// $.verb$ takes top resource name and the apply declarative configuration for $.subresourcePath$,
// applies it and returns the applied $.resultType|private$, and an error, if there is any.
func (c *$.type|privatePlural$) $.verb$(ctx $.context|raw$, $.type|private$Name string, $.inputType|private$ *$.inputApplyConfig|raw$, opts $.ApplyOptions|raw$) (*$.resultType|raw$, error) {
	return c.SubResourceApply[$.resultType|raw$](ctx, $.type|private$Name, "$.subresourcePath$", $.inputType|private$, opts)
}
`
