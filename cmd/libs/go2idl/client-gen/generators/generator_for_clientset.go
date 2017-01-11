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
	"strings"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	clientgentypes "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/types"
)

// genClientset generates a package for a clientset.
type genClientset struct {
	generator.DefaultGen
	groups             []clientgentypes.GroupVersions
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
	for _, group := range g.groups {
		for _, version := range group.Versions {
			typedClientPath := filepath.Join(g.typedClientPath, group.Group.NonEmpty(), version.NonEmpty())
			imports = append(imports, strings.ToLower(fmt.Sprintf("%s%s \"%s\"", version.NonEmpty(), group.Group.NonEmpty(), typedClientPath)))
		}
	}
	imports = append(imports, "github.com/golang/glog")
	imports = append(imports, "k8s.io/kubernetes/pkg/util/flowcontrol")
	// import solely to initialize client auth plugins.
	imports = append(imports, "_ \"k8s.io/kubernetes/plugin/pkg/client/auth\"")
	return
}

func (g *genClientset) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	// TODO: We actually don't need any type information to generate the clientset,
	// perhaps we can adapt the go2ild framework to this kind of usage.
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	const pkgDiscovery = "k8s.io/kubernetes/pkg/client/typed/discovery"
	const pkgRESTClient = "k8s.io/kubernetes/pkg/client/restclient"

	allGroups := clientgentypes.ToGroupVersionPackages(g.groups)

	m := map[string]interface{}{
		"allGroups":                        allGroups,
		"Config":                           c.Universe.Type(types.Name{Package: pkgRESTClient, Name: "Config"}),
		"DefaultKubernetesUserAgent":       c.Universe.Function(types.Name{Package: pkgRESTClient, Name: "DefaultKubernetesUserAgent"}),
		"RESTClientInterface":              c.Universe.Type(types.Name{Package: pkgRESTClient, Name: "Interface"}),
		"DiscoveryInterface":               c.Universe.Type(types.Name{Package: pkgDiscovery, Name: "DiscoveryInterface"}),
		"DiscoveryClient":                  c.Universe.Type(types.Name{Package: pkgDiscovery, Name: "DiscoveryClient"}),
		"NewDiscoveryClientForConfig":      c.Universe.Function(types.Name{Package: pkgDiscovery, Name: "NewDiscoveryClientForConfig"}),
		"NewDiscoveryClientForConfigOrDie": c.Universe.Function(types.Name{Package: pkgDiscovery, Name: "NewDiscoveryClientForConfigOrDie"}),
		"NewDiscoveryClient":               c.Universe.Function(types.Name{Package: pkgDiscovery, Name: "NewDiscoveryClient"}),
	}
	sw.Do(clientsetInterface, m)
	sw.Do(clientsetTemplate, m)
	for _, g := range allGroups {
		sw.Do(clientsetInterfaceImplTemplate, g)
		// don't generated the default method if generating internalversion clientset
		if g.IsDefaultVersion && g.Version != "" {
			sw.Do(clientsetInterfaceDefaultVersionImpl, g)
		}
	}
	sw.Do(getDiscoveryTemplate, m)
	sw.Do(newClientsetForConfigTemplate, m)
	sw.Do(newClientsetForConfigOrDieTemplate, m)
	sw.Do(newClientsetForRESTClientTemplate, m)

	return sw.Error()
}

var clientsetInterface = `
type Interface interface {
	Discovery() $.DiscoveryInterface|raw$
    $range .allGroups$$.GroupVersion$() $.PackageName$.$.GroupVersion$Interface
	$if .IsDefaultVersion$// Deprecated: please explicitly pick a version if possible.
	$.Group$() $.PackageName$.$.GroupVersion$Interface$end$
    $end$
}
`

var clientsetTemplate = `
// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*$.DiscoveryClient|raw$
    $range .allGroups$*$.PackageName$.$.GroupVersion$Client
    $end$
}
`

var clientsetInterfaceImplTemplate = `
// $.GroupVersion$ retrieves the $.GroupVersion$Client
func (c *Clientset) $.GroupVersion$() $.PackageName$.$.GroupVersion$Interface {
	if c == nil {
		return nil
	}
	return c.$.GroupVersion$Client
}
`

var clientsetInterfaceDefaultVersionImpl = `
// Deprecated: $.Group$ retrieves the default version of $.Group$Client.
// Please explicitly pick a version.
func (c *Clientset) $.Group$() $.PackageName$.$.GroupVersion$Interface {
	if c == nil {
		return nil
	}
	return c.$.GroupVersion$Client
}
`

var getDiscoveryTemplate = `
// Discovery retrieves the DiscoveryClient
func (c *Clientset) Discovery() $.DiscoveryInterface|raw$ {
	if c == nil {
		return nil
	}
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
	var cs Clientset
	var err error
$range .allGroups$    cs.$.GroupVersion$Client, err =$.PackageName$.NewForConfig(&configShallowCopy)
	if err!=nil {
		return nil, err
	}
$end$
	cs.DiscoveryClient, err = $.NewDiscoveryClientForConfig|raw$(&configShallowCopy)
	if err!=nil {
		glog.Errorf("failed to create the DiscoveryClient: %v", err)
		return nil, err
	}
	return &cs, nil
}
`

var newClientsetForConfigOrDieTemplate = `
// NewForConfigOrDie creates a new Clientset for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *$.Config|raw$) *Clientset {
	var cs Clientset
$range .allGroups$    cs.$.GroupVersion$Client =$.PackageName$.NewForConfigOrDie(c)
$end$
	cs.DiscoveryClient = $.NewDiscoveryClientForConfigOrDie|raw$(c)
	return &cs
}
`

var newClientsetForRESTClientTemplate = `
// New creates a new Clientset for the given RESTClient.
func New(c $.RESTClientInterface|raw$) *Clientset {
	var cs Clientset
$range .allGroups$    cs.$.GroupVersion$Client =$.PackageName$.New(c)
$end$
	cs.DiscoveryClient = $.NewDiscoveryClient|raw$(c)
	return &cs
}
`
