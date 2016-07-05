/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/generators/normalization"
	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// genClientset generates a package for a clientset.
type genClientset struct {
	generator.DefaultGen
	groupVersions      []unversioned.GroupVersion
	typedClientPath    string
	outputPackage      string
	imports            namer.ImportTracker
	clientsetGenerated bool
}

var _ generator.Generator = &genClientset{}

func (g *genClientset) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

// We only want to call GenerateType() once.
func (g *genClientset) Filter(c *generator.Context, t *types.Type) bool {
	ret := !g.clientsetGenerated
	g.clientsetGenerated = true
	return ret
}

func (g *genClientset) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	for _, gv := range g.groupVersions {
		group := normalization.Group(gv.Group)
		version := normalization.Version(gv.Version)
		typedClientPath := filepath.Join(g.typedClientPath, group, version)
		group = normalization.BeforeFirstDot(group)
		imports = append(imports, fmt.Sprintf("%s%s \"%s\"", version, group, typedClientPath))
	}
	imports = append(imports, "github.com/golang/glog")
	imports = append(imports, "k8s.io/kubernetes/pkg/util/flowcontrol")
	return
}

func (g *genClientset) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	// TODO: We actually don't need any type information to generate the clientset,
	// perhaps we can adapt the go2ild framework to this kind of usage.
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	const pkgDiscovery = "k8s.io/kubernetes/pkg/client/typed/discovery"
	const pkgRESTClient = "k8s.io/kubernetes/pkg/client/restclient"

	type arg struct {
		Group       string
		PackageName string
	}

	allGroups := []arg{}
	for _, gv := range g.groupVersions {
		group := normalization.BeforeFirstDot(normalization.Group(gv.Group))
		version := normalization.Version(gv.Version)
		allGroups = append(allGroups, arg{namer.IC(group), version + group})
	}

	m := map[string]interface{}{
		"allGroups":                        allGroups,
		"Config":                           c.Universe.Type(types.Name{Package: pkgRESTClient, Name: "Config"}),
		"DefaultKubernetesUserAgent":       c.Universe.Function(types.Name{Package: pkgRESTClient, Name: "DefaultKubernetesUserAgent"}),
		"RESTClient":                       c.Universe.Type(types.Name{Package: pkgRESTClient, Name: "RESTClient"}),
		"DiscoveryInterface":               c.Universe.Type(types.Name{Package: pkgDiscovery, Name: "DiscoveryInterface"}),
		"DiscoveryClient":                  c.Universe.Type(types.Name{Package: pkgDiscovery, Name: "DiscoveryClient"}),
		"NewDiscoveryClientForConfig":      c.Universe.Function(types.Name{Package: pkgDiscovery, Name: "NewDiscoveryClientForConfig"}),
		"NewDiscoveryClientForConfigOrDie": c.Universe.Function(types.Name{Package: pkgDiscovery, Name: "NewDiscoveryClientForConfigOrDie"}),
		"NewDiscoveryClient":               c.Universe.Function(types.Name{Package: pkgDiscovery, Name: "NewDiscoveryClient"}),
	}
	sw.Do(clientsetInterfaceTemplate, m)
	sw.Do(clientsetTemplate, m)
	for _, g := range allGroups {
		sw.Do(clientsetInterfaceImplTemplate, g)
	}
	sw.Do(getDiscoveryTemplate, m)
	sw.Do(newClientsetForConfigTemplate, m)
	sw.Do(newClientsetForConfigOrDieTemplate, m)
	sw.Do(newClientsetForRESTClientTemplate, m)

	return sw.Error()
}

var clientsetInterfaceTemplate = `
type Interface interface {
	Discovery() $.DiscoveryInterface|raw$
    $range .allGroups$$.Group$() $.PackageName$.$.Group$Interface
    $end$
}
`

var clientsetTemplate = `
// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*$.DiscoveryClient|raw$
    $range .allGroups$*$.PackageName$.$.Group$Client
    $end$
}
`

var clientsetInterfaceImplTemplate = `
// $.Group$ retrieves the $.Group$Client
func (c *Clientset) $.Group$() $.PackageName$.$.Group$Interface {
	if c == nil {
		return nil
	}
	return c.$.Group$Client
}
`
var getDiscoveryTemplate = `
// Discovery retrieves the DiscoveryClient
func (c *Clientset) Discovery() $.DiscoveryInterface|raw$ {
	return c.DiscoveryClient
}
`

var newClientsetForConfigTemplate = `
// NewForConfig creates a new Clientset for the given config.
func NewForConfig(c *$.Config|raw$) (*Clientset, error) {
	configShallowCopy := *c
	if configShallowCopy.RateLimiter == nil && configShallowCopy.QPS > 0 {
		configShallowCopy.RateLimiter = flowcontrol.NewTokenBucketRateLimiter(configShallowCopy.QPS, configShallowCopy.Burst)
	}
	var clientset Clientset
	var err error
$range .allGroups$    clientset.$.Group$Client, err =$.PackageName$.NewForConfig(&configShallowCopy)
	if err!=nil {
		return nil, err
	}
$end$
	clientset.DiscoveryClient, err = $.NewDiscoveryClientForConfig|raw$(&configShallowCopy)
	if err!=nil {
		glog.Errorf("failed to create the DiscoveryClient: %v", err)
		return nil, err
	}
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
	clientset.DiscoveryClient = $.NewDiscoveryClientForConfigOrDie|raw$(c)
	return &clientset
}
`

var newClientsetForRESTClientTemplate = `
// New creates a new Clientset for the given RESTClient.
func New(c *$.RESTClient|raw$) *Clientset {
	var clientset Clientset
$range .allGroups$    clientset.$.Group$Client =$.PackageName$.New(c)
$end$
	clientset.DiscoveryClient = $.NewDiscoveryClient|raw$(c)
	return &clientset
}
`
