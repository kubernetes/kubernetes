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
	"path"
	"path/filepath"
	"strings"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/code-generator/cmd/lister-gen/args"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems(pluralExceptions map[string]string) namer.NameSystems {
	return namer.NameSystems{
		"public":             namer.NewPublicNamer(0),
		"private":            namer.NewPrivateNamer(0),
		"raw":                namer.NewRawNamer("", nil),
		"publicPlural":       namer.NewPublicPluralNamer(pluralExceptions),
		"allLowercasePlural": namer.NewAllLowercasePluralNamer(pluralExceptions),
		"lowercaseSingular":  &lowercaseSingularNamer{},
	}
}

// lowercaseSingularNamer implements Namer
type lowercaseSingularNamer struct{}

// Name returns t's name in all lowercase.
func (n *lowercaseSingularNamer) Name(t *types.Type) string {
	return strings.ToLower(t.Name.Name)
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

// GetTargets makes the client target definition.
func GetTargets(context *generator.Context, args *args.Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, "", gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	var targetList []generator.Target
	for _, inputPkg := range context.Inputs {
		p := context.Universe.Package(inputPkg)

		objectMeta, internal, err := objectMetaForPackage(p)
		if err != nil {
			klog.Fatal(err)
		}
		if objectMeta == nil {
			// no types in this package had genclient
			continue
		}

		var gv clientgentypes.GroupVersion
		var internalGVPkg string

		if internal {
			lastSlash := strings.LastIndex(p.Path, "/")
			if lastSlash == -1 {
				klog.Fatalf("error constructing internal group version for package %q", p.Path)
			}
			gv.Group = clientgentypes.Group(p.Path[lastSlash+1:])
			internalGVPkg = p.Path
		} else {
			parts := strings.Split(p.Path, "/")
			gv.Group = clientgentypes.Group(parts[len(parts)-2])
			gv.Version = clientgentypes.Version(parts[len(parts)-1])

			internalGVPkg = strings.Join(parts[0:len(parts)-1], "/")
		}
		groupPackageName := strings.ToLower(gv.Group.NonEmpty())

		// If there's a comment of the form "// +groupName=somegroup" or
		// "// +groupName=somegroup.foo.bar.io", use the first field (somegroup) as the name of the
		// group when generating.
		if override := gengo.ExtractCommentTags("+", p.Comments)["groupName"]; override != nil {
			gv.Group = clientgentypes.Group(strings.SplitN(override[0], ".", 2)[0])
		}

		var typesToGenerate []*types.Type
		for _, t := range p.Types {
			tags := util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
			if !tags.GenerateClient || !tags.HasVerb("list") || !tags.HasVerb("get") {
				continue
			}
			typesToGenerate = append(typesToGenerate, t)
		}
		if len(typesToGenerate) == 0 {
			continue
		}
		orderer := namer.Orderer{Namer: namer.NewPrivateNamer(0)}
		typesToGenerate = orderer.OrderTypes(typesToGenerate)

		subdir := []string{groupPackageName, strings.ToLower(gv.Version.NonEmpty())}
		outputDir := filepath.Join(args.OutputDir, filepath.Join(subdir...))
		outputPkg := path.Join(args.OutputPkg, path.Join(subdir...))
		targetList = append(targetList, &generator.SimpleTarget{
			PkgName:       strings.ToLower(gv.Version.NonEmpty()),
			PkgPath:       outputPkg,
			PkgDir:        outputDir,
			HeaderComment: boilerplate,
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				tags := util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
				return tags.GenerateClient && tags.HasVerb("list") && tags.HasVerb("get")
			},
			GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
				generators = append(generators, &expansionGenerator{
					GoGenerator: generator.GoGenerator{
						OutputFilename: "expansion_generated.go",
					},
					outputPath: outputDir,
					types:      typesToGenerate,
				})

				for _, t := range typesToGenerate {
					generators = append(generators, &listerGenerator{
						GoGenerator: generator.GoGenerator{
							OutputFilename: strings.ToLower(t.Name.Name) + ".go",
						},
						outputPackage:  outputPkg,
						groupVersion:   gv,
						internalGVPkg:  internalGVPkg,
						typeToGenerate: t,
						imports:        generator.NewImportTrackerForPackage(outputPkg),
						objectMeta:     objectMeta,
					})
				}
				return generators
			},
		})
	}

	return targetList
}

// objectMetaForPackage returns the type of ObjectMeta used by package p.
func objectMetaForPackage(p *types.Package) (*types.Type, bool, error) {
	generatingForPackage := false
	for _, t := range p.Types {
		// filter out types which don't have genclient.
		if !util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...)).GenerateClient {
			continue
		}
		generatingForPackage = true
		for _, member := range t.Members {
			if member.Name == "ObjectMeta" {
				return member.Type, isInternal(member), nil
			}
		}
	}
	if generatingForPackage {
		return nil, false, fmt.Errorf("unable to find ObjectMeta for any types in package %s", p.Path)
	}
	return nil, false, nil
}

// isInternal returns true if the tags for a member do not contain a json tag
func isInternal(m types.Member) bool {
	return !strings.Contains(m.Tags, "json")
}

// listerGenerator produces a file of listers for a given GroupVersion and
// type.
type listerGenerator struct {
	generator.GoGenerator
	outputPackage  string
	groupVersion   clientgentypes.GroupVersion
	internalGVPkg  string
	typeToGenerate *types.Type
	imports        namer.ImportTracker
	objectMeta     *types.Type
}

var _ generator.Generator = &listerGenerator{}

func (g *listerGenerator) Filter(c *generator.Context, t *types.Type) bool {
	return t == g.typeToGenerate
}

func (g *listerGenerator) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *listerGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	return
}

func (g *listerGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	klog.V(5).Infof("processing type %v", t)
	m := map[string]interface{}{
		"Resource":               c.Universe.Function(types.Name{Package: t.Name.Package, Name: "Resource"}),
		"labelsSelector":         c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/labels", Name: "Selector"}),
		"listersResourceIndexer": c.Universe.Function(types.Name{Package: "k8s.io/client-go/listers", Name: "ResourceIndexer"}),
		"listersNew":             c.Universe.Function(types.Name{Package: "k8s.io/client-go/listers", Name: "New"}),
		"listersNewNamespaced":   c.Universe.Function(types.Name{Package: "k8s.io/client-go/listers", Name: "NewNamespaced"}),
		"cacheIndexer":           c.Universe.Type(types.Name{Package: "k8s.io/client-go/tools/cache", Name: "Indexer"}),
		"type":                   t,
		"objectMeta":             g.objectMeta,
	}

	tags, err := util.ParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
	if err != nil {
		return err
	}

	if tags.NonNamespaced {
		sw.Do(typeListerInterfaceNonNamespaced, m)
	} else {
		sw.Do(typeListerInterface, m)
	}

	sw.Do(typeListerStruct, m)
	sw.Do(typeListerConstructor, m)

	if tags.NonNamespaced {
		return sw.Error()
	}

	sw.Do(typeListerNamespaceLister, m)
	sw.Do(namespaceListerInterface, m)
	sw.Do(namespaceListerStruct, m)

	return sw.Error()
}

var typeListerInterface = `
// $.type|public$Lister helps list $.type|publicPlural$.
// All objects returned here must be treated as read-only.
type $.type|public$Lister interface {
	// List lists all $.type|publicPlural$ in the indexer.
	// Objects returned here must be treated as read-only.
	List(selector $.labelsSelector|raw$) (ret []*$.type|raw$, err error)
	// $.type|publicPlural$ returns an object that can list and get $.type|publicPlural$.
	$.type|publicPlural$(namespace string) $.type|public$NamespaceLister
	$.type|public$ListerExpansion
}
`

var typeListerInterfaceNonNamespaced = `
// $.type|public$Lister helps list $.type|publicPlural$.
// All objects returned here must be treated as read-only.
type $.type|public$Lister interface {
	// List lists all $.type|publicPlural$ in the indexer.
	// Objects returned here must be treated as read-only.
	List(selector $.labelsSelector|raw$) (ret []*$.type|raw$, err error)
	// Get retrieves the $.type|public$ from the index for a given name.
	// Objects returned here must be treated as read-only.
	Get(name string) (*$.type|raw$, error)
	$.type|public$ListerExpansion
}
`

// This embeds a typed resource indexer instead of aliasing, so that the struct
// is available as a receiver for methods specific to the generated type
// (from the corresponding expansion interface).
var typeListerStruct = `
// $.type|private$Lister implements the $.type|public$Lister interface.
type $.type|private$Lister struct {
	$.listersResourceIndexer|raw$[*$.type|raw$]
}
`

var typeListerConstructor = `
// New$.type|public$Lister returns a new $.type|public$Lister.
func New$.type|public$Lister(indexer $.cacheIndexer|raw$) $.type|public$Lister {
	return &$.type|private$Lister{$.listersNew|raw$[*$.type|raw$](indexer, $.Resource|raw$("$.type|lowercaseSingular$"))}
}
`

var typeListerNamespaceLister = `
// $.type|publicPlural$ returns an object that can list and get $.type|publicPlural$.
func (s *$.type|private$Lister) $.type|publicPlural$(namespace string) $.type|public$NamespaceLister {
	return $.type|private$NamespaceLister{$.listersNewNamespaced|raw$[*$.type|raw$](s.ResourceIndexer, namespace)}
}
`

var namespaceListerInterface = `
// $.type|public$NamespaceLister helps list and get $.type|publicPlural$.
// All objects returned here must be treated as read-only.
type $.type|public$NamespaceLister interface {
	// List lists all $.type|publicPlural$ in the indexer for a given namespace.
	// Objects returned here must be treated as read-only.
	List(selector $.labelsSelector|raw$) (ret []*$.type|raw$, err error)
	// Get retrieves the $.type|public$ from the indexer for a given namespace and name.
	// Objects returned here must be treated as read-only.
	Get(name string) (*$.type|raw$, error)
	$.type|public$NamespaceListerExpansion
}
`

// This embeds a typed namespaced resource indexer instead of aliasing, so that the struct
// is available as a receiver for methods specific to the generated type
// (from the corresponding expansion interface).
var namespaceListerStruct = `
// $.type|private$NamespaceLister implements the $.type|public$NamespaceLister
// interface.
type $.type|private$NamespaceLister struct {
	$.listersResourceIndexer|raw$[*$.type|raw$]
}
`
