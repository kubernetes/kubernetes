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

	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

// genFakeForGroup produces a file for a group client, e.g. ExtensionsClient for the extension group.
type genFakeForGroup struct {
	generator.DefaultGen
	outputPackage  string
	realClientPath string
	group          string
	// types in this group
	types   []*types.Type
	imports namer.ImportTracker
}

var _ generator.Generator = &genFakeForGroup{}

// We only want to call GenerateType() once per group.
func (g *genFakeForGroup) Filter(c *generator.Context, t *types.Type) bool {
	return t == g.types[0]
}

func (g *genFakeForGroup) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *genFakeForGroup) Imports(c *generator.Context) (imports []string) {
	imports = append(g.imports.ImportLines(), fmt.Sprintf("%s \"%s\"", filepath.Base(g.realClientPath), g.realClientPath))
	return imports
}

func (g *genFakeForGroup) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	const pkgTestingCore = "k8s.io/kubernetes/pkg/client/testing/core"
	const pkgRESTClient = "k8s.io/kubernetes/pkg/client/restclient"
	m := map[string]interface{}{
		"group":      g.group,
		"Group":      namer.IC(g.group),
		"Fake":       c.Universe.Type(types.Name{Package: pkgTestingCore, Name: "Fake"}),
		"RESTClient": c.Universe.Type(types.Name{Package: pkgRESTClient, Name: "RESTClient"}),
	}
	sw.Do(groupClientTemplate, m)
	for _, t := range g.types {
		wrapper := map[string]interface{}{
			"type":              t,
			"Group":             namer.IC(g.group),
			"realClientPackage": filepath.Base(g.realClientPath),
		}
		namespaced := !extractBoolTagOrDie("nonNamespaced", t.SecondClosestCommentLines)
		if namespaced {
			sw.Do(getterImplNamespaced, wrapper)
		} else {
			sw.Do(getterImplNonNamespaced, wrapper)

		}
	}
	sw.Do(getRESTClient, m)
	return sw.Error()
}

var groupClientTemplate = `
type Fake$.Group$ struct {
	*$.Fake|raw$
}
`

var getterImplNamespaced = `
func (c *Fake$.Group$) $.type|publicPlural$(namespace string) $.realClientPackage$.$.type|public$Interface {
	return &Fake$.type|publicPlural${c, namespace}
}
`

var getterImplNonNamespaced = `
func (c *Fake$.Group$) $.type|publicPlural$() $.realClientPackage$.$.type|public$Interface {
	return &Fake$.type|publicPlural${c}
}
`

var getRESTClient = `
// GetRESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *Fake$.Group$) GetRESTClient() *$.RESTClient|raw$ {
  return nil
}
`
