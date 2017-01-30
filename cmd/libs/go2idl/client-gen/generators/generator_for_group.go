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
	types        []*types.Type
	imports      namer.ImportTracker
	inputPacakge string
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
	return
}

func (g *genGroup) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	const pkgRESTClient = "k8s.io/kubernetes/pkg/client/restclient"
	const pkgAPI = "k8s.io/kubernetes/pkg/api"
	const pkgSerializer = "k8s.io/apimachinery/pkg/runtime/serializer"
	const pkgUnversioned = "k8s.io/kubernetes/pkg/api/unversioned"
	const pkgSchema = "k8s.io/apimachinery/pkg/runtime/schema"

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
	p := c.Universe.Package(g.inputPacakge)
	if override := types.ExtractCommentTags("+", p.DocComments)["groupName"]; override != nil {
		groupName = override[0]
	}

	m := map[string]interface{}{
		"group":                      g.group,
		"version":                    g.version,
		"GroupVersion":               namer.IC(g.group) + namer.IC(g.version),
		"groupName":                  groupName,
		"types":                      g.types,
		"Config":                     c.Universe.Type(types.Name{Package: pkgRESTClient, Name: "Config"}),
		"DefaultKubernetesUserAgent": c.Universe.Function(types.Name{Package: pkgRESTClient, Name: "DefaultKubernetesUserAgent"}),
		"RESTClientInterface":        c.Universe.Type(types.Name{Package: pkgRESTClient, Name: "Interface"}),
		"RESTClientFor":              c.Universe.Function(types.Name{Package: pkgRESTClient, Name: "RESTClientFor"}),
		"ParseGroupVersion":          c.Universe.Function(types.Name{Package: pkgSchema, Name: "ParseGroupVersion"}),
		"apiPath":                    apiPath(g.group),
		"apiRegistry":                c.Universe.Variable(types.Name{Package: pkgAPI, Name: "Registry"}),
		"codecs":                     c.Universe.Variable(types.Name{Package: pkgAPI, Name: "Codecs"}),
		"directCodecFactory":         c.Universe.Variable(types.Name{Package: pkgSerializer, Name: "DirectCodecFactory"}),
		"Errorf":                     c.Universe.Variable(types.Name{Package: "fmt", Name: "Errorf"}),
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
    RESTClient() $.RESTClientInterface|raw$
    $range .types$ $.|publicPlural$Getter
    $end$
}
`

var groupClientTemplate = `
// $.GroupVersion$Client is used to interact with features provided by the $.groupName$ group.
type $.GroupVersion$Client struct {
	restClient $.RESTClientInterface|raw$
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
func NewForConfig(c *$.Config|raw$) (*$.GroupVersion$Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := $.RESTClientFor|raw$(&config)
	if err != nil {
		return nil, err
	}
	return &$.GroupVersion$Client{client}, nil
}
`

var newClientForConfigOrDieTemplate = `
// NewForConfigOrDie creates a new $.GroupVersion$Client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *$.Config|raw$) *$.GroupVersion$Client {
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
func (c *$.GroupVersion$Client) RESTClient() $.RESTClientInterface|raw$ {
	if c == nil {
		return nil
	}
	return c.restClient
}
`

var newClientForRESTClientTemplate = `
// New creates a new $.GroupVersion$Client for the given RESTClient.
func New(c $.RESTClientInterface|raw$) *$.GroupVersion$Client {
	return &$.GroupVersion$Client{c}
}
`
var setInternalVersionClientDefaultsTemplate = `
func setConfigDefaults(config *$.Config|raw$) error {
	// if $.group$ group is not registered, return an error
	g, err := $.apiRegistry|raw$.Group("$.groupName$")
	if err != nil {
		return err
	}
	config.APIPath = $.apiPath$
	if config.UserAgent == "" {
		config.UserAgent = $.DefaultKubernetesUserAgent|raw$()
	}
	if config.GroupVersion == nil || config.GroupVersion.Group != g.GroupVersion.Group {
		copyGroupVersion := g.GroupVersion
		config.GroupVersion = &copyGroupVersion
	}
	config.NegotiatedSerializer = $.codecs|raw$

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
func setConfigDefaults(config *$.Config|raw$) error {
	gv, err := $.ParseGroupVersion|raw$("$.groupName$/$.version$")
	if err != nil {
		return err
	}
	// if $.groupName$/$.version$ is not enabled, return an error
	if ! $.apiRegistry|raw$.IsEnabledVersion(gv) {
		return $.Errorf|raw$("$.groupName$/$.version$ is not enabled")
	}
	config.APIPath = $.apiPath$
	if config.UserAgent == "" {
		config.UserAgent = $.DefaultKubernetesUserAgent|raw$()
	}
	copyGroupVersion := gv
	config.GroupVersion = &copyGroupVersion

	config.NegotiatedSerializer = $.directCodecFactory|raw${CodecFactory: $.codecs|raw$}

	return nil
}
`
