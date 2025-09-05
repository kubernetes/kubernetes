/*
Copyright 2025 The Kubernetes Authors.

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
	"strconv"
	"strings"

	"k8s.io/code-generator/cmd/prerelease-lifecycle-gen/args"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// This is the comment tag that carries parameters for API status generation.  Because the cadence is fixed, we can predict
// with near certainty when this lifecycle happens as the API is introduced.
const (
	tagEnabledName         = "k8s:openapi-model-gen"
	tagOpenAPIModelPackage = "modelPackageName"
	tagOpenAPIModelName    = "modelName"
)

func extractEnabledTypeTag(comments []string) (bool, error) {
	v, err := singularTag(tagEnabledName, comments)
	if v == nil || err != nil {
		return false, err
	}
	result, err := strconv.ParseBool(v.Value)
	if err != nil {
		return false, fmt.Errorf("invalid %s:%v", tagEnabledName, err)
	}
	return result, nil
}

func extractOpenAPIModelPackage(t *types.Package) (string, error) {
	v, err := singularTag(tagOpenAPIModelPackage, t.Comments)
	if v == nil || err != nil {
		return "", err
	}
	return v.Value, nil
}

func extractOpenAPIModelName(t *types.Type) (string, error) {
	comments := append(append([]string{}, t.SecondClosestCommentLines...), t.CommentLines...)
	v, err := singularTag(tagOpenAPIModelName, comments)
	if v == nil || err != nil {
		return "", err
	}
	return v.Value, nil
}

func singularTag(tagName string, comments []string) (*gengo.Tag, error) {
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{tagName}, comments)
	if err != nil {
		return nil, err
	}
	if len(tags) == 0 {
		return nil, nil
	}
	if len(tags) > 1 {
		return nil, fmt.Errorf("multiple %s tags found", tagName)
	}
	tag := tags[tagName]
	if len(tag) == 0 {
		return nil, nil
	}
	if len(tag) > 1 {
		return nil, fmt.Errorf("multiple %s tags found", tagName)
	}
	value := tag[0]
	return &value, nil
}

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"public": namer.NewPublicNamer(1),
		"raw":    namer.NewRawNamer("", nil),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

// GetTargets makes the target definition.
func GetTargets(context *generator.Context, args *args.Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, gengo.StdBuildTag, gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	targets := []generator.Target{}

	for _, i := range context.Inputs {
		klog.V(5).Infof("Considering pkg %q", i)

		pkg := context.Universe[i]

		pkgEnabled, err := extractEnabledTypeTag(pkg.Comments)
		if err != nil {
			klog.Fatalf("Package %v: invalid %s:%v", i, tagEnabledName, err)
		}
		klog.V(3).Infof("Generating package %q", pkg.Path)

		openAPIModelPackage, err := extractOpenAPIModelPackage(pkg)
		if err != nil {
			klog.Fatalf("Package %v: invalid %s:%v", i, tagOpenAPIModelPackage, err)
		}

		hasNamedModels := false
		hasModels := false
		for _, t := range pkg.Types {
			klog.V(5).Infof("  considering type %q", t.Name.String())
			modelType := isModelType(t)
			if modelType {
				hasModels = true
			}
			typeEnabled, err := extractEnabledTypeTag(t.CommentLines)
			if err != nil {
				klog.Fatalf("Package %v type %v: invalid %s:%v", i, t, tagEnabledName, err)
			}
			if !pkgEnabled && typeEnabled {
				pkgEnabled = true
			}
			modelName, err := extractOpenAPIModelName(t)
			if err != nil {
				klog.Fatalf("Type %v: invalid %s:%v", t.Name.String(), tagEnabledName, err)
			}
			if modelName == "" {
				continue
			}
			if !modelType {
				klog.Fatalf("Type %v requests open api generation but is not an API type", t)
			}
			hasNamedModels = true
		}
		if !pkgEnabled || !hasModels || (!hasNamedModels && openAPIModelPackage == "") {
			klog.V(5).Infof("  skipping package")
			continue
		}

		targets = append(targets,
			&generator.SimpleTarget{
				PkgName:       strings.Split(path.Base(pkg.Path), ".")[0],
				PkgPath:       pkg.Path,
				PkgDir:        pkg.Dir, // output pkg is the same as the input
				HeaderComment: boilerplate,
				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == pkg.Path
				},
				GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
					return []generator.Generator{
						NewOpenAPIModelGen(args.OutputFile, pkg.Path, openAPIModelPackage),
					}
				},
			})
	}
	return targets
}

// genDeepCopy produces a file with autogenerated openapi model functions.
type genOpenAPIModel struct {
	generator.GoGenerator
	targetPackage       string
	imports             namer.ImportTracker
	typesForInit        []*types.Type
	openAPIModelPackage string
}

// NewOpenAPIModelGen creates a generator
func NewOpenAPIModelGen(outputFilename, targetPackage string, openAPIModelPackage string) generator.Generator {
	return &genOpenAPIModel{
		GoGenerator: generator.GoGenerator{
			OutputFilename: outputFilename,
		},
		targetPackage:       targetPackage,
		imports:             generator.NewImportTracker(),
		typesForInit:        make([]*types.Type, 0),
		openAPIModelPackage: openAPIModelPackage,
	}
}

func (g *genOpenAPIModel) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"public":       namer.NewPublicNamer(1),
		"intrapackage": namer.NewPublicNamer(0),
		"raw":          namer.NewRawNamer("", nil),
	}
}

func (g *genOpenAPIModel) Filter(c *generator.Context, t *types.Type) bool {
	// Filter out types not being processed or not copyable within the package.
	if !isModelType(t) {
		klog.V(2).Infof("Type %v is not a valid target for OpenAPI models", t)
		return false
	}
	g.typesForInit = append(g.typesForInit, t)
	return true
}

// isModelType indicates whether or not a type could be used to serve an API.  That means, "does it have TypeMeta".
// This doesn't mean the type is served, but we will handle all TypeMeta types.
func isModelType(t *types.Type) bool {
	// Filter out private types.
	if namer.IsPrivateGoName(t.Name.Name) {
		return false
	}

	for t.Kind == types.Alias {
		t = t.Underlying
	}

	if t.Kind != types.Struct {
		return false
	}
	return true
}

func (g *genOpenAPIModel) isOtherPackage(pkg string) bool {
	if pkg == g.targetPackage {
		return false
	}
	if strings.HasSuffix(pkg, "\""+g.targetPackage+"\"") {
		return false
	}
	return true
}

func (g *genOpenAPIModel) Imports(c *generator.Context) (imports []string) {
	importLines := []string{}
	for _, singleImport := range g.imports.ImportLines() {
		if g.isOtherPackage(singleImport) {
			importLines = append(importLines, singleImport)
		}
	}
	return importLines
}

func (g *genOpenAPIModel) Init(c *generator.Context, w io.Writer) error {
	return nil
}

func (g *genOpenAPIModel) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	klog.V(3).Infof("Generating openapi model names for type %v", t)

	modelName, err := extractOpenAPIModelName(t)
	if err != nil {
		klog.Fatalf("Type %v: invalid %s:%v", t.Name.String(), tagEnabledName, err)
	}
	if modelName == "" {
		if g.openAPIModelPackage == "" {
			klog.V(3).Infof("Skipping openapi model names for type %v", t)
			return nil
		}
		modelName = g.openAPIModelPackage + "." + t.Name.Name
	}

	a := map[string]interface{}{
		"type":      t,
		"modelName": modelName,
	}

	sw := generator.NewSnippetWriter(w, c, "$", "$")

	sw.Do("// OpenAPIModelName returns the OpenAPI model name for this type.\n", a)
	sw.Do("func (in $.type|intrapackage$) OpenAPIModelName() string {\n", a)
	sw.Do("    return \"$.modelName$\"\n", a)
	sw.Do("}\n\n", nil)

	return sw.Error()
}
