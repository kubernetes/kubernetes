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

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// genGroup produces a file for a group client, e.g. ExtensionsClient for the extension group.
type genGroup struct {
	generator.DefaultGen
	outputPackage string
	group         string
	version       string
	apiPath       string
	// types in this group
	types            []*types.Type
	imports          namer.ImportTracker
	inputPackage     string
	clientsetPackage string
}

var _ generator.Generator = &genGroup{}

// We only want to call GenerateType() once per group.
func (g *genGroup) Filter(c *generator.Context, t *types.Type) bool {
	return t == g.types[0]
}

func (g *genGroup) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *genGroup) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	imports = append(imports, filepath.Join(g.clientsetPackage, "scheme"))
	return
}

func (g *genGroup) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	apiPath := func(group string) string {
		if len(g.apiPath) > 0 {
			return `"` + g.apiPath + `"`
		}
		if group == "core" {
			return `"/api"`
		}
		return `"/apis"`
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
		"group":                          g.group,
		"version":                        g.version,
		"GroupVersion":                   namer.IC(g.group) + namer.IC(g.version),
		"groupName":                      groupName,
		"types":                          g.types,
		"apiPath":                        apiPath(g.group),
		"schemaGroupVersion":             c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/schema", Name: "GroupVersion"}),
		"runtimeAPIVersionInternal":      c.Universe.Variable(types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "APIVersionInternal"}),
		"serializerDirectCodecFactory":   c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/serializer", Name: "DirectCodecFactory"}),
		"restConfig":                     c.Universe.Type(types.Name{Package: "k8s.io/client-go/rest", Name: "Config"}),
		"restDefaultKubernetesUserAgent": c.Universe.Function(types.Name{Package: "k8s.io/client-go/rest", Name: "DefaultKubernetesUserAgent"}),
		"restRESTClientInterface":        c.Universe.Type(types.Name{Package: "k8s.io/client-go/rest", Name: "Interface"}),
		"restRESTClientFor":              c.Universe.Function(types.Name{Package: "k8s.io/client-go/rest", Name: "RESTClientFor"}),
		"SchemeGroupVersion":             c.Universe.Variable(types.Name{Package: g.inputPackage, Name: "SchemeGroupVersion"}),
	}
	sw.Do(groupInterfaceTemplate, m)
	sw.Do(groupClientTemplate, m)
	for _, t := range g.types {
		wrapper := map[string]interface{}{
			"type":         t,
			"GroupVersion": namer.IC(g.group) + namer.IC(g.version),
		}
		namespaced := !extractBoolTagOrDie("nonNamespaced", t.SecondClosestCommentLines)
		if namespaced {
			sw.Do(getterImplNamespaced, wrapper)
		} else {
			sw.Do(getterImplNonNamespaced, wrapper)
		}
	}
	sw.Do(newClientForConfigTemplate, m)
	sw.Do(newClientForConfigOrDieTemplate, m)
	sw.Do(newClientForRESTClientTemplate, m)
	if g.version == "" {
		sw.Do(setInternalVersionClientDefaultsTemplate, m)
	} else {
		sw.Do(setClientDefaultsTemplate, m)
	}
	sw.Do(getRESTClient, m)

	return sw.Error()
}

var groupInterfaceTemplate = `
type $.GroupVersion$Interface interface {
    RESTClient() $.restRESTClientInterface|raw$
    $range .types$ $.|publicPlural$Getter
    $end$
}
`

var groupClientTemplate = `
// $.GroupVersion$Client is used to interact with features provided by the $.groupName$ group.
type $.GroupVersion$Client struct {
	restClient $.restRESTClientInterface|raw$
}
`

var getterImplNamespaced = `
func (c *$.GroupVersion$Client) $.type|publicPlural$(namespace string) $.type|public$Interface {
	return new$.type|publicPlural$(c, namespace)
}
`

var getterImplNonNamespaced = `
func (c *$.GroupVersion$Client) $.type|publicPlural$() $.type|public$Interface {
	return new$.type|publicPlural$(c)
}
`

var newClientForConfigTemplate = `
// NewForConfig creates a new $.GroupVersion$Client for the given config.
func NewForConfig(c *$.restConfig|raw$) (*$.GroupVersion$Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := $.restRESTClientFor|raw$(&config)
	if err != nil {
		return nil, err
	}
	return &$.GroupVersion$Client{client}, nil
}
`

var newClientForConfigOrDieTemplate = `
// NewForConfigOrDie creates a new $.GroupVersion$Client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *$.restConfig|raw$) *$.GroupVersion$Client {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}
`

var getRESTClient = `
// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *$.GroupVersion$Client) RESTClient() $.restRESTClientInterface|raw$ {
	if c == nil {
		return nil
	}
	return c.restClient
}
`

var newClientForRESTClientTemplate = `
// New creates a new $.GroupVersion$Client for the given RESTClient.
func New(c $.restRESTClientInterface|raw$) *$.GroupVersion$Client {
	return &$.GroupVersion$Client{c}
}
`

var setInternalVersionClientDefaultsTemplate = `
func setConfigDefaults(config *$.restConfig|raw$) error {
	g, err := scheme.Registry.Group("$.groupName$")
	if err != nil {
		return err
	}

	config.APIPath = $.apiPath$
	if config.UserAgent == "" {
		config.UserAgent = $.restDefaultKubernetesUserAgent|raw$()
	}
	if config.GroupVersion == nil || config.GroupVersion.Group != g.GroupVersion.Group {
		gv := g.GroupVersion
		config.GroupVersion = &gv
	}
	config.NegotiatedSerializer = scheme.Codecs

	if config.QPS == 0 {
		config.QPS = 5
	}
	if config.Burst == 0 {
		config.Burst = 10
	}

	return nil
}
`

var setClientDefaultsTemplate = `
func setConfigDefaults(config *$.restConfig|raw$) error {
	gv := $.SchemeGroupVersion|raw$
	config.GroupVersion =  &gv
	config.APIPath = $.apiPath$
	config.NegotiatedSerializer = $.serializerDirectCodecFactory|raw${CodecFactory: scheme.Codecs}

	if config.UserAgent == "" {
		config.UserAgent = $.restDefaultKubernetesUserAgent|raw$()
	}

	return nil
}
`
