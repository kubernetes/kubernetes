/*
Copyright 2017 The Kubernetes Authors.

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

package scheme

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/code-generator/cmd/client-gen/path"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// GenScheme produces a package for a clientset with the scheme, codecs and parameter codecs.
type GenScheme struct {
	generator.DefaultGen
	OutputPackage   string
	Groups          []clientgentypes.GroupVersions
	GroupGoNames    map[clientgentypes.GroupVersion]string
	InputPackages   map[clientgentypes.GroupVersion]string
	OutputPath      string
	ImportTracker   namer.ImportTracker
	PrivateScheme   bool
	CreateRegistry  bool
	schemeGenerated bool
}

func (g *GenScheme) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.OutputPackage, g.ImportTracker),
	}
}

// We only want to call GenerateType() once.
func (g *GenScheme) Filter(c *generator.Context, t *types.Type) bool {
	ret := !g.schemeGenerated
	g.schemeGenerated = true
	return ret
}

func (g *GenScheme) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.ImportTracker.ImportLines()...)
	for _, group := range g.Groups {
		for _, version := range group.Versions {
			packagePath := g.InputPackages[clientgentypes.GroupVersion{Group: group.Group, Version: version.Version}]
			groupAlias := strings.ToLower(g.GroupGoNames[clientgentypes.GroupVersion{Group: group.Group, Version: version.Version}])
			if g.CreateRegistry {
				// import the install package for internal clientsets instead of the type package with register.go
				if version.Version != "" {
					packagePath = filepath.Dir(packagePath)
				}
				packagePath = filepath.Join(packagePath, "install")

				imports = append(imports, fmt.Sprintf("%s \"%s\"", groupAlias, path.Vendorless(packagePath)))
				break
			} else {
				imports = append(imports, fmt.Sprintf("%s%s \"%s\"", groupAlias, strings.ToLower(version.Version.NonEmpty()), path.Vendorless(packagePath)))
			}
		}
	}
	return
}

func (g *GenScheme) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	allGroupVersions := clientgentypes.ToGroupVersionInfo(g.Groups, g.GroupGoNames)
	allInstallGroups := clientgentypes.ToGroupInstallPackages(g.Groups, g.GroupGoNames)

	m := map[string]interface{}{
		"allGroupVersions":          allGroupVersions,
		"allInstallGroups":          allInstallGroups,
		"customRegister":            false,
		"runtimeNewParameterCodec":  c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "NewParameterCodec"}),
		"runtimeNewScheme":          c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "NewScheme"}),
		"serializerNewCodecFactory": c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/serializer", Name: "NewCodecFactory"}),
		"runtimeScheme":             c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "Scheme"}),
		"runtimeSchemeBuilder":      c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "SchemeBuilder"}),
		"runtimeUtilMust":           c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/util/runtime", Name: "Must"}),
		"schemaGroupVersion":        c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/schema", Name: "GroupVersion"}),
		"metav1AddToGroupVersion":   c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "AddToGroupVersion"}),
	}
	globals := map[string]string{
		"Scheme":         "Scheme",
		"Codecs":         "Codecs",
		"ParameterCodec": "ParameterCodec",
		"Registry":       "Registry",
	}
	for k, v := range globals {
		if g.PrivateScheme {
			m[k] = strings.ToLower(v[0:1]) + v[1:]
		} else {
			m[k] = v
		}
	}

	sw.Do(globalsTemplate, m)

	if g.OutputPath != "" {
		if _, err := os.Stat(filepath.Join(g.OutputPath, strings.ToLower("register_custom.go"))); err == nil {
			m["customRegister"] = true
		}
	}

	if g.CreateRegistry {
		sw.Do(registryRegistration, m)
	} else {
		sw.Do(simpleRegistration, m)
	}

	return sw.Error()
}

var globalsTemplate = `
var $.Scheme$ = $.runtimeNewScheme|raw$()
var $.Codecs$ = $.serializerNewCodecFactory|raw$($.Scheme$)
var $.ParameterCodec$ = $.runtimeNewParameterCodec|raw$($.Scheme$)`

var registryRegistration = `

func init() {
	$.metav1AddToGroupVersion|raw$($.Scheme$, $.schemaGroupVersion|raw${Version: "v1"})
	Install($.Scheme$)
}

// Install registers the API group and adds types to a scheme
func Install(scheme *$.runtimeScheme|raw$) {
	$- range .allInstallGroups$
	$.InstallPackageAlias$.Install(scheme)
	$- end$
	$if .customRegister$
	ExtraInstall(scheme)
	$end -$
}
`

var simpleRegistration = `
var localSchemeBuilder = $.runtimeSchemeBuilder|raw${
	$- range .allGroupVersions$
	$.PackageAlias$.AddToScheme,
	$- end$
	$if .customRegister$
	ExtraAddToScheme,
	$end -$
}

// AddToScheme adds all types of this clientset into the given scheme. This allows composition
// of clientsets, like in:
//
//   import (
//     "k8s.io/client-go/kubernetes"
//     clientsetscheme "k8s.io/client-go/kubernetes/scheme"
//     aggregatorclientsetscheme "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/scheme"
//   )
//
//   kclientset, _ := kubernetes.NewForConfig(c)
//   _ = aggregatorclientsetscheme.AddToScheme(clientsetscheme.Scheme)
//
// After this, RawExtensions in Kubernetes types will serialize kube-aggregator types
// correctly.
var AddToScheme = localSchemeBuilder.AddToScheme

func init() {
	$.metav1AddToGroupVersion|raw$($.Scheme$, $.schemaGroupVersion|raw${Version: "v1"})
	$.runtimeUtilMust|raw$(AddToScheme($.Scheme$))
}
`
