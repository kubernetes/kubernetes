/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io"
	"path/filepath"

	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// genClientset generates a package for a clientset.
type genClientset struct {
	generator.DefaultGen
	groupVersions   []unversioned.GroupVersion
	typedClientPath string
	outputPackage   string
	imports         *generator.ImportTracker
}

var _ generator.Generator = &genClientset{}

func (g *genClientset) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

var generate_clientset = true

// We only want to call GenerateType() once.
func (g *genClientset) Filter(c *generator.Context, t *types.Type) bool {
	ret := generate_clientset
	generate_clientset = false
	return ret
}

func normalizeGroup(group string) string {
	if group == "api" {
		return "legacy"
	}
	return group
}

func normalizeVersion(version string) string {
	if version == "" {
		return "unversioned"
	}
	return version
}

func (g *genClientset) Imports(c *generator.Context) (imports []string) {
	for _, gv := range g.groupVersions {
		group := normalizeGroup(gv.Group)
		version := normalizeVersion(gv.Version)
		typedClientPath := filepath.Join(g.typedClientPath, group, version)
		imports = append(imports, g.imports.ImportLines()...)
		imports = append(imports, fmt.Sprintf("%s_%s \"%s\"", group, version, typedClientPath))
	}
	return
}

func (g *genClientset) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	// TODO: We actually don't need any type information to generate the clientset,
	// perhaps we can adapt the go2ild framework to this kind of usage.
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	const pkgUnversioned = "k8s.io/kubernetes/pkg/client/unversioned"

	type arg struct {
		Group       string
		PackageName string
	}

	allGroups := []arg{}
	for _, gv := range g.groupVersions {
		group := normalizeGroup(gv.Group)
		version := normalizeVersion(gv.Version)
		allGroups = append(allGroups, arg{namer.IC(group), group + "_" + version})
	}

	m := map[string]interface{}{
		"allGroups":                  allGroups,
		"Config":                     c.Universe.Type(types.Name{Package: pkgUnversioned, Name: "Config"}),
		"DefaultKubernetesUserAgent": c.Universe.Function(types.Name{Package: pkgUnversioned, Name: "DefaultKubernetesUserAgent"}),
		"RESTClient":                 c.Universe.Type(types.Name{Package: pkgUnversioned, Name: "RESTClient"}),
	}
	sw.Do(clientsetInterfaceTemplate, m)
	sw.Do(clientsetTemplate, m)
	for _, g := range allGroups {
		sw.Do(clientsetInterfaceImplTemplate, g)
	}
	sw.Do(newClientsetForConfigTemplate, m)
	sw.Do(newClientsetForConfigOrDieTemplate, m)
	sw.Do(newClientsetForRESTClientTemplate, m)

	return sw.Error()
}

var clientsetInterfaceTemplate = `
type Interface interface {
    $range .allGroups$$.Group$() $.PackageName$.$.Group$Interface
    $end$
}
`

var clientsetTemplate = `
// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
    $range .allGroups$*$.PackageName$.$.Group$Client
    $end$
}
`

var clientsetInterfaceImplTemplate = `
// $.Group$ retrieves the $.Group$Client
func (c *Clientset) $.Group$() $.PackageName$.$.Group$Interface {
	return c.$.Group$Client
}
`

var newClientsetForConfigTemplate = `
// NewForConfig creates a new Clientset for the given config.
func NewForConfig(c *$.Config|raw$) (*Clientset, error) {
	var clientset Clientset
	var err error
$range .allGroups$    clientset.$.Group$Client, err =$.PackageName$.NewForConfig(c)
	if err!=nil {
		return nil, err
	}
$end$
	return &clientset, nil
}
`

var newClientsetForConfigOrDieTemplate = `
// NewForConfigOrDie creates a new Clientset for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *$.Config|raw$) *Clientset {
	var clientset Clientset
$range .allGroups$    clientset.$.Group$Client =$.PackageName$.NewForConfigOrDie(c)
$end$
	return &clientset
}
`

var newClientsetForRESTClientTemplate = `
// New creates a new Clientset for the given RESTClient.
func New(c *$.RESTClient|raw$) *Clientset {
	var clientset Clientset
$range .allGroups$    clientset.$.Group$Client =$.PackageName$.New(c)
$end$

	return &clientset
}
`
