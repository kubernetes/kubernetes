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

	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
)

// genGroup produces a file for a group client, e.g. ExtensionsClient for the extension group.
type genGroup struct {
	generator.GoGenerator
	outputPackage string
	group         string
	version       string
	groupGoName   string
	apiPath       string
	// types in this group
	types            []*types.Type
	imports          namer.ImportTracker
	inputPackage     string
	clientsetPackage string // must be a Go import-path
	// If the genGroup has been called. This generator should only execute once.
	called bool
}

var _ generator.Generator = &genGroup{}

// We only want to call GenerateType() once per group.
func (g *genGroup) Filter(c *generator.Context, t *types.Type) bool {
	if !g.called {
		g.called = true
		return true
	}
	return false
}

func (g *genGroup) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *genGroup) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	return
}

func (g *genGroup) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	// allow user to define a group name that's different from the one parsed from the directory.
	p := c.Universe.Package(g.inputPackage)
	groupName := g.group
	if override := gengo.ExtractCommentTags("+", p.Comments)["groupName"]; override != nil {
		groupName = override[0]
	}

	apiPath := `"` + g.apiPath + `"`
	if groupName == "" {
		apiPath = `"/api"`
	}
	schemePackage := path.Join(g.clientsetPackage, "scheme")
	m := map[string]interface{}{
		"version":                            g.version,
		"groupName":                          groupName,
		"GroupGoName":                        g.groupGoName,
		"Version":                            namer.IC(g.version),
		"types":                              g.types,
		"apiPath":                            apiPath,
		"httpClient":                         c.Universe.Type(types.Name{Package: "net/http", Name: "Client"}),
		"schemaGroupVersion":                 c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/schema", Name: "GroupVersion"}),
		"runtimeAPIVersionInternal":          c.Universe.Variable(types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "APIVersionInternal"}),
		"restConfig":                         c.Universe.Type(types.Name{Package: "k8s.io/client-go/rest", Name: "Config"}),
		"restDefaultKubernetesUserAgent":     c.Universe.Function(types.Name{Package: "k8s.io/client-go/rest", Name: "DefaultKubernetesUserAgent"}),
		"restRESTClientInterface":            c.Universe.Type(types.Name{Package: "k8s.io/client-go/rest", Name: "Interface"}),
		"RESTHTTPClientFor":                  c.Universe.Function(types.Name{Package: "k8s.io/client-go/rest", Name: "HTTPClientFor"}),
		"restRESTClientFor":                  c.Universe.Function(types.Name{Package: "k8s.io/client-go/rest", Name: "RESTClientFor"}),
		"restRESTClientForConfigAndClient":   c.Universe.Function(types.Name{Package: "k8s.io/client-go/rest", Name: "RESTClientForConfigAndClient"}),
		"restCodecFactoryForGeneratedClient": c.Universe.Function(types.Name{Package: "k8s.io/client-go/rest", Name: "CodecFactoryForGeneratedClient"}),
		"SchemeGroupVersion":                 c.Universe.Variable(types.Name{Package: g.inputPackage, Name: "SchemeGroupVersion"}),
		"SchemePrioritizedVersionsForGroup":  c.Universe.Variable(types.Name{Package: schemePackage, Name: "Scheme.PrioritizedVersionsForGroup"}),
		"Codecs":                             c.Universe.Variable(types.Name{Package: schemePackage, Name: "Codecs"}),
		"Scheme":                             c.Universe.Variable(types.Name{Package: schemePackage, Name: "Scheme"}),
	}
	sw.Do(groupInterfaceTemplate, m)
	sw.Do(groupClientTemplate, m)
	for _, t := range g.types {
		tags, err := util.ParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
		if err != nil {
			return err
		}
		wrapper := map[string]interface{}{
			"type":        t,
			"GroupGoName": g.groupGoName,
			"Version":     namer.IC(g.version),
		}
		if tags.NonNamespaced {
			sw.Do(getterImplNonNamespaced, wrapper)
		} else {
			sw.Do(getterImplNamespaced, wrapper)
		}
	}
	sw.Do(newClientForConfigTemplate, m)
	sw.Do(newClientForConfigAndClientTemplate, m)
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
type $.GroupGoName$$.Version$Interface interface {
    RESTClient() $.restRESTClientInterface|raw$
    $range .types$ $.|publicPlural$Getter
    $end$
}
`

var groupClientTemplate = `
// $.GroupGoName$$.Version$Client is used to interact with features provided by the $.groupName$ group.
type $.GroupGoName$$.Version$Client struct {
	restClient $.restRESTClientInterface|raw$
}
`

var getterImplNamespaced = `
func (c *$.GroupGoName$$.Version$Client) $.type|publicPlural$(namespace string) $.type|public$Interface {
	return new$.type|publicPlural$(c, namespace)
}
`

var getterImplNonNamespaced = `
func (c *$.GroupGoName$$.Version$Client) $.type|publicPlural$() $.type|public$Interface {
	return new$.type|publicPlural$(c)
}
`

var newClientForConfigTemplate = `
// NewForConfig creates a new $.GroupGoName$$.Version$Client for the given config.
// NewForConfig is equivalent to NewForConfigAndClient(c, httpClient),
// where httpClient was generated with rest.HTTPClientFor(c).
func NewForConfig(c *$.restConfig|raw$) (*$.GroupGoName$$.Version$Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	httpClient, err := $.RESTHTTPClientFor|raw$(&config)
	if err != nil {
		return nil, err
	}
	return NewForConfigAndClient(&config, httpClient)
}
`

var newClientForConfigAndClientTemplate = `
// NewForConfigAndClient creates a new $.GroupGoName$$.Version$Client for the given config and http client.
// Note the http client provided takes precedence over the configured transport values.
func NewForConfigAndClient(c *$.restConfig|raw$, h *$.httpClient|raw$) (*$.GroupGoName$$.Version$Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := $.restRESTClientForConfigAndClient|raw$(&config, h)
	if err != nil {
		return nil, err
	}
	return &$.GroupGoName$$.Version$Client{client}, nil
}
`

var newClientForConfigOrDieTemplate = `
// NewForConfigOrDie creates a new $.GroupGoName$$.Version$Client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *$.restConfig|raw$) *$.GroupGoName$$.Version$Client {
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
func (c *$.GroupGoName$$.Version$Client) RESTClient() $.restRESTClientInterface|raw$ {
	if c == nil {
		return nil
	}
	return c.restClient
}
`

var newClientForRESTClientTemplate = `
// New creates a new $.GroupGoName$$.Version$Client for the given RESTClient.
func New(c $.restRESTClientInterface|raw$) *$.GroupGoName$$.Version$Client {
	return &$.GroupGoName$$.Version$Client{c}
}
`

var setInternalVersionClientDefaultsTemplate = `
func setConfigDefaults(config *$.restConfig|raw$) error {
	config.APIPath = $.apiPath$
	if config.UserAgent == "" {
		config.UserAgent = $.restDefaultKubernetesUserAgent|raw$()
	}
	if config.GroupVersion == nil || config.GroupVersion.Group != $.SchemePrioritizedVersionsForGroup|raw$("$.groupName$")[0].Group {
		gv := $.SchemePrioritizedVersionsForGroup|raw$("$.groupName$")[0]
		config.GroupVersion = &gv
	}
	config.NegotiatedSerializer = $.restCodecFactoryForGeneratedClient|raw$($.Scheme|raw$, $.Codecs|raw$)

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
	config.NegotiatedSerializer = $.restCodecFactoryForGeneratedClient|raw$($.Scheme|raw$, $.Codecs|raw$).WithoutConversion()

	if config.UserAgent == "" {
		config.UserAgent = $.restDefaultKubernetesUserAgent|raw$()
	}

	return nil
}
`
