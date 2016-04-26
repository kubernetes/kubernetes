/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/generators/normalization"
	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

// genGroup produces a file for a group client, e.g. ExtensionsClient for the extension group.
type genGroup struct {
	generator.DefaultGen
	outputPackage string
	group         string
	version       string
	// types in this group
	types   []*types.Type
	imports namer.ImportTracker
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
	const pkgRegistered = "k8s.io/kubernetes/pkg/apimachinery/registered"
	const pkgAPI = "k8s.io/kubernetes/pkg/api"
	apiPath := func(group string) string {
		if group == "core" {
			return `"/api"`
		}
		return `"/apis"`
	}

	canonize := func(group string) string {
		if group == "core" {
			return ""
		}
		return group
	}

	m := map[string]interface{}{
		"group":                      normalization.BeforeFirstDot(g.group),
		"Group":                      namer.IC(normalization.BeforeFirstDot(g.group)),
		"canonicalGroup":             canonize(g.group),
		"types":                      g.types,
		"Config":                     c.Universe.Type(types.Name{Package: pkgRESTClient, Name: "Config"}),
		"DefaultKubernetesUserAgent": c.Universe.Function(types.Name{Package: pkgRESTClient, Name: "DefaultKubernetesUserAgent"}),
		"RESTClient":                 c.Universe.Type(types.Name{Package: pkgRESTClient, Name: "RESTClient"}),
		"RESTClientFor":              c.Universe.Function(types.Name{Package: pkgRESTClient, Name: "RESTClientFor"}),
		"latestGroup":                c.Universe.Variable(types.Name{Package: pkgRegistered, Name: "Group"}),
		"GroupOrDie":                 c.Universe.Variable(types.Name{Package: pkgRegistered, Name: "GroupOrDie"}),
		"apiPath":                    apiPath(g.group),
		"codecs":                     c.Universe.Variable(types.Name{Package: pkgAPI, Name: "Codecs"}),
		"Errorf":                     c.Universe.Variable(types.Name{Package: "fmt", Name: "Errorf"}),
	}
	sw.Do(groupInterfaceTemplate, m)
	sw.Do(groupClientTemplate, m)
	for _, t := range g.types {
		wrapper := map[string]interface{}{
			"type":  t,
			"Group": namer.IC(normalization.BeforeFirstDot(g.group)),
		}
		namespaced := !(types.ExtractCommentTags("+", t.SecondClosestCommentLines)["nonNamespaced"] == "true")
		if namespaced {
			sw.Do(getterImplNamespaced, wrapper)
		} else {
			sw.Do(getterImplNonNamespaced, wrapper)

		}
	}
	sw.Do(newClientForConfigTemplate, m)
	sw.Do(newClientForConfigOrDieTemplate, m)
	sw.Do(newClientForRESTClientTemplate, m)
	if g.version == "unversioned" {
		sw.Do(setInternalVersionClientDefaultsTemplate, m)
	} else {
		sw.Do(setClientDefaultsTemplate, m)
	}
	sw.Do(getRESTClient, m)

	return sw.Error()
}

var groupInterfaceTemplate = `
type $.Group$Interface interface {
    GetRESTClient() *$.RESTClient|raw$
    $range .types$ $.|publicPlural$Getter
    $end$
}
`

var groupClientTemplate = `
// $.Group$Client is used to interact with features provided by the $.Group$ group.
type $.Group$Client struct {
	*$.RESTClient|raw$
}
`

var getterImplNamespaced = `
func (c *$.Group$Client) $.type|publicPlural$(namespace string) $.type|public$Interface {
	return new$.type|publicPlural$(c, namespace)
}
`

var getterImplNonNamespaced = `
func (c *$.Group$Client) $.type|publicPlural$() $.type|public$Interface {
	return new$.type|publicPlural$(c)
}
`

var newClientForConfigTemplate = `
// NewForConfig creates a new $.Group$Client for the given config.
func NewForConfig(c *$.Config|raw$) (*$.Group$Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := $.RESTClientFor|raw$(&config)
	if err != nil {
		return nil, err
	}
	return &$.Group$Client{client}, nil
}
`

var newClientForConfigOrDieTemplate = `
// NewForConfigOrDie creates a new $.Group$Client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *$.Config|raw$) *$.Group$Client {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}
`

var getRESTClient = `
// GetRESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *$.Group$Client) GetRESTClient() *$.RESTClient|raw$ {
	if c == nil {
		return nil
	}
	return c.RESTClient
}
`

var newClientForRESTClientTemplate = `
// New creates a new $.Group$Client for the given RESTClient.
func New(c *$.RESTClient|raw$) *$.Group$Client {
	return &$.Group$Client{c}
}
`
var setInternalVersionClientDefaultsTemplate = `
func setConfigDefaults(config *$.Config|raw$) error {
	// if $.group$ group is not registered, return an error
	g, err := $.latestGroup|raw$("$.canonicalGroup$")
	if err != nil {
		return err
	}
	config.APIPath = $.apiPath$
	if config.UserAgent == "" {
		config.UserAgent = $.DefaultKubernetesUserAgent|raw$()
	}
	// TODO: Unconditionally set the config.Version, until we fix the config.
	//if config.Version == "" {
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	//}

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
	// if $.group$ group is not registered, return an error
	g, err := $.latestGroup|raw$("$.canonicalGroup$")
	if err != nil {
		return err
	}
	config.APIPath = $.apiPath$
	if config.UserAgent == "" {
		config.UserAgent = $.DefaultKubernetesUserAgent|raw$()
	}
	// TODO: Unconditionally set the config.Version, until we fix the config.
	//if config.Version == "" {
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	//}

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
