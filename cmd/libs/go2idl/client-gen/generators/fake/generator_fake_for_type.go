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
	"fmt"
	"io"
	"path/filepath"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// genFakeForType produces a file for each top-level type.
type genFakeForType struct {
	generator.DefaultGen
	outputPackage string
	group         string
	version       string
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

	// Allow overriding via a comment on the type
	genStatus, err := types.ExtractSingleBoolCommentTag("+", "genclientstatus", hasStatus, t.SecondClosestCommentLines)
	if err != nil {
		fmt.Printf("error looking up +genclientstatus: %v\n", err)
	}
	return genStatus
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
	const pkgTestingCore = "k8s.io/kubernetes/pkg/client/testing/core"
	namespaced := !extractBoolTagOrDie("nonNamespaced", t.SecondClosestCommentLines)
	canonicalGroup := g.group
	if canonicalGroup == "core" {
		canonicalGroup = ""
	}

	groupName := g.group
	if g.group == "core" {
		groupName = ""
	}

	// allow user to define a group name that's different from the one parsed from the directory.
	p := c.Universe.Package(g.inputPackage)
	if override := types.ExtractCommentTags("+", p.DocComments)["groupName"]; override != nil {
		groupName = override[0]
	}

	m := map[string]interface{}{
		"type":                 t,
		"package":              pkg,
		"Package":              namer.IC(pkg),
		"namespaced":           namespaced,
		"Group":                namer.IC(g.group),
		"GroupVersion":         namer.IC(g.group) + namer.IC(g.version),
		"group":                canonicalGroup,
		"groupName":            groupName,
		"version":              g.version,
		"watchInterface":       c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/watch", Name: "Interface"}),
		"GroupVersionResource": c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/schema", Name: "GroupVersionResource"}),
		"PatchType":            c.Universe.Type(types.Name{Package: "k8s.io/kubernetes/pkg/api", Name: "PatchType"}),
		"Everything":           c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/labels", Name: "Everything"}),

		"NewRootListAction":              c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootListAction"}),
		"NewListAction":                  c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewListAction"}),
		"NewRootGetAction":               c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootGetAction"}),
		"NewGetAction":                   c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewGetAction"}),
		"NewRootDeleteAction":            c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootDeleteAction"}),
		"NewDeleteAction":                c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewDeleteAction"}),
		"NewRootDeleteCollectionAction":  c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootDeleteCollectionAction"}),
		"NewDeleteCollectionAction":      c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewDeleteCollectionAction"}),
		"NewRootUpdateAction":            c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootUpdateAction"}),
		"NewUpdateAction":                c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewUpdateAction"}),
		"NewRootCreateAction":            c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootCreateAction"}),
		"NewCreateAction":                c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewCreateAction"}),
		"NewRootWatchAction":             c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootWatchAction"}),
		"NewWatchAction":                 c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewWatchAction"}),
		"NewUpdateSubresourceAction":     c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewUpdateSubresourceAction"}),
		"NewRootUpdateSubresourceAction": c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootUpdateSubresourceAction"}),
		"NewRootPatchAction":             c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootPatchAction"}),
		"NewPatchAction":                 c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewPatchAction"}),
		"NewRootPatchSubresourceAction":  c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewRootPatchSubresourceAction"}),
		"NewPatchSubresourceAction":      c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "NewPatchSubresourceAction"}),
		"ExtractFromListOptions":         c.Universe.Function(types.Name{Package: pkgTestingCore, Name: "ExtractFromListOptions"}),
	}

	if g.version == "" {
		m["DeleteOptions"] = c.Universe.Type(types.Name{Package: "k8s.io/kubernetes/pkg/api", Name: "DeleteOptions"})
		m["ListOptions"] = c.Universe.Type(types.Name{Package: "k8s.io/kubernetes/pkg/api", Name: "ListOptions"})
	} else {
		m["DeleteOptions"] = c.Universe.Type(types.Name{Package: "k8s.io/kubernetes/pkg/api/v1", Name: "DeleteOptions"})
		m["ListOptions"] = c.Universe.Type(types.Name{Package: "k8s.io/kubernetes/pkg/api/v1", Name: "ListOptions"})
	}
	m["GetOptions"] = c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "GetOptions"})

	noMethods := extractBoolTagOrDie("noMethods", t.SecondClosestCommentLines) == true

	if namespaced {
		sw.Do(structNamespaced, m)
	} else {
		sw.Do(structNonNamespaced, m)
	}

	if !noMethods {
		sw.Do(resource, m)
		sw.Do(createTemplate, m)
		sw.Do(updateTemplate, m)
		// Generate the UpdateStatus method if the type has a status
		if genStatus(t) {
			sw.Do(updateStatusTemplate, m)
		}
		sw.Do(deleteTemplate, m)
		sw.Do(deleteCollectionTemplate, m)
		sw.Do(getTemplate, m)
		if hasObjectMeta(t) {
			sw.Do(listUsingOptionsTemplate, m)
		} else {
			sw.Do(listTemplate, m)
		}
		sw.Do(watchTemplate, m)
		sw.Do(patchTemplate, m)
	}

	return sw.Error()
}

// template for the struct that implements the type's interface
var structNamespaced = `
// Fake$.type|publicPlural$ implements $.type|public$Interface
type Fake$.type|publicPlural$ struct {
	Fake *Fake$.GroupVersion$
	ns     string
}
`

// template for the struct that implements the type's interface
var structNonNamespaced = `
// Fake$.type|publicPlural$ implements $.type|public$Interface
type Fake$.type|publicPlural$ struct {
	Fake *Fake$.GroupVersion$
}
`

var resource = `
var $.type|allLowercasePlural$Resource = $.GroupVersionResource|raw${Group: "$.groupName$", Version: "$.version$", Resource: "$.type|allLowercasePlural$"}
`

var listTemplate = `
func (c *Fake$.type|publicPlural$) List(opts $.ListOptions|raw$) (result *$.type|raw$List, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewListAction|raw$($.type|allLowercasePlural$Resource, c.ns, opts), &$.type|raw$List{})
		$else$Invokes($.NewRootListAction|raw$($.type|allLowercasePlural$Resource, opts), &$.type|raw$List{})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.type|raw$List), err
}
`

var listUsingOptionsTemplate = `
func (c *Fake$.type|publicPlural$) List(opts $.ListOptions|raw$) (result *$.type|raw$List, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewListAction|raw$($.type|allLowercasePlural$Resource, c.ns, opts), &$.type|raw$List{})
		$else$Invokes($.NewRootListAction|raw$($.type|allLowercasePlural$Resource, opts), &$.type|raw$List{})$end$
	if obj == nil {
		return nil, err
	}

	label, _, _ := $.ExtractFromListOptions|raw$(opts)
	if label == nil {
		label = $.Everything|raw$()
	}
	list := &$.type|raw$List{}
	for _, item := range obj.(*$.type|raw$List).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}
`

var getTemplate = `
func (c *Fake$.type|publicPlural$) Get(name string, options $.GetOptions|raw$) (result *$.type|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewGetAction|raw$($.type|allLowercasePlural$Resource, c.ns, name), &$.type|raw${})
		$else$Invokes($.NewRootGetAction|raw$($.type|allLowercasePlural$Resource, name), &$.type|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.type|raw$), err
}
`

var deleteTemplate = `
func (c *Fake$.type|publicPlural$) Delete(name string, options *$.DeleteOptions|raw$) error {
	_, err := c.Fake.
		$if .namespaced$Invokes($.NewDeleteAction|raw$($.type|allLowercasePlural$Resource, c.ns, name), &$.type|raw${})
		$else$Invokes($.NewRootDeleteAction|raw$($.type|allLowercasePlural$Resource, name), &$.type|raw${})$end$
	return err
}
`

var deleteCollectionTemplate = `
func (c *Fake$.type|publicPlural$) DeleteCollection(options *$.DeleteOptions|raw$, listOptions $.ListOptions|raw$) error {
	$if .namespaced$action := $.NewDeleteCollectionAction|raw$($.type|allLowercasePlural$Resource, c.ns, listOptions)
	$else$action := $.NewRootDeleteCollectionAction|raw$($.type|allLowercasePlural$Resource, listOptions)
	$end$
	_, err := c.Fake.Invokes(action, &$.type|raw$List{})
	return err
}
`

var createTemplate = `
func (c *Fake$.type|publicPlural$) Create($.type|private$ *$.type|raw$) (result *$.type|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewCreateAction|raw$($.type|allLowercasePlural$Resource, c.ns, $.type|private$), &$.type|raw${})
		$else$Invokes($.NewRootCreateAction|raw$($.type|allLowercasePlural$Resource, $.type|private$), &$.type|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.type|raw$), err
}
`

var updateTemplate = `
func (c *Fake$.type|publicPlural$) Update($.type|private$ *$.type|raw$) (result *$.type|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewUpdateAction|raw$($.type|allLowercasePlural$Resource, c.ns, $.type|private$), &$.type|raw${})
		$else$Invokes($.NewRootUpdateAction|raw$($.type|allLowercasePlural$Resource, $.type|private$), &$.type|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.type|raw$), err
}
`

var updateStatusTemplate = `
func (c *Fake$.type|publicPlural$) UpdateStatus($.type|private$ *$.type|raw$) (*$.type|raw$, error) {
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
func (c *Fake$.type|publicPlural$) Watch(opts $.ListOptions|raw$) ($.watchInterface|raw$, error) {
	return c.Fake.
		$if .namespaced$InvokesWatch($.NewWatchAction|raw$($.type|allLowercasePlural$Resource, c.ns, opts))
		$else$InvokesWatch($.NewRootWatchAction|raw$($.type|allLowercasePlural$Resource, opts))$end$
}
`

var patchTemplate = `
// Patch applies the patch and returns the patched $.type|private$.
func (c *Fake$.type|publicPlural$) Patch(name string, pt $.PatchType|raw$, data []byte, subresources ...string) (result *$.type|raw$, err error) {
	obj, err := c.Fake.
		$if .namespaced$Invokes($.NewPatchSubresourceAction|raw$($.type|allLowercasePlural$Resource, c.ns, name, data, subresources... ), &$.type|raw${})
		$else$Invokes($.NewRootPatchSubresourceAction|raw$($.type|allLowercasePlural$Resource, name, data, subresources...), &$.type|raw${})$end$
	if obj == nil {
		return nil, err
	}
	return obj.(*$.type|raw$), err
}
`
