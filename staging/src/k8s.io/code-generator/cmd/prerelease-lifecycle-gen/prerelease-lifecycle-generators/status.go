/*
Copyright 2020 The Kubernetes Authors.

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

package prereleaselifecyclegenerators

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
	tagEnabledName    = "k8s:prerelease-lifecycle-gen"
	introducedTagName = tagEnabledName + ":introduced"
	deprecatedTagName = tagEnabledName + ":deprecated"
	removedTagName    = tagEnabledName + ":removed"

	replacementTagName = tagEnabledName + ":replacement"
)

// enabledTagValue holds parameters from a tagName tag.
type tagValue struct {
	value string
}

func extractEnabledTypeTag(t *types.Type) *tagValue {
	comments := append(append([]string{}, t.SecondClosestCommentLines...), t.CommentLines...)
	return extractTag(tagEnabledName, comments)
}

func tagExists(tagName string, t *types.Type) bool {
	comments := append(append([]string{}, t.SecondClosestCommentLines...), t.CommentLines...)
	rawTag := extractTag(tagName, comments)
	return rawTag != nil
}

func extractKubeVersionTag(tagName string, t *types.Type) (*tagValue, int, int, error) {
	comments := append(append([]string{}, t.SecondClosestCommentLines...), t.CommentLines...)
	rawTag := extractTag(tagName, comments)
	if rawTag == nil || len(rawTag.value) == 0 {
		return nil, -1, -1, fmt.Errorf("%v missing %v=Version tag", t, tagName)
	}

	splitValue := strings.Split(rawTag.value, ".")
	if len(splitValue) != 2 || len(splitValue[0]) == 0 || len(splitValue[1]) == 0 {
		return nil, -1, -1, fmt.Errorf("%v format must match %v=xx.yy tag", t, tagName)
	}
	major, err := strconv.ParseInt(splitValue[0], 10, 32)
	if err != nil {
		return nil, -1, -1, fmt.Errorf("%v format must match %v=xx.yy : %w", t, tagName, err)
	}
	minor, err := strconv.ParseInt(splitValue[1], 10, 32)
	if err != nil {
		return nil, -1, -1, fmt.Errorf("%v format must match %v=xx.yy : %w", t, tagName, err)
	}

	return rawTag, int(major), int(minor), nil
}

func extractIntroducedTag(t *types.Type) (*tagValue, int, int, error) {
	return extractKubeVersionTag(introducedTagName, t)
}

func extractDeprecatedTag(t *types.Type) (*tagValue, int, int, error) {
	return extractKubeVersionTag(deprecatedTagName, t)
}

func extractRemovedTag(t *types.Type) (*tagValue, int, int, error) {
	return extractKubeVersionTag(removedTagName, t)
}

func extractReplacementTag(t *types.Type) (group, version, kind string, hasReplacement bool, err error) {
	comments := append(append([]string{}, t.SecondClosestCommentLines...), t.CommentLines...)

	tagVals := gengo.ExtractCommentTags("+", comments)[replacementTagName]
	if len(tagVals) == 0 {
		// No match for the tag.
		return "", "", "", false, nil
	}
	// If there are multiple values, abort.
	if len(tagVals) > 1 {
		return "", "", "", false, fmt.Errorf("found %d %s tags: %q", len(tagVals), replacementTagName, tagVals)
	}
	tagValue := tagVals[0]
	parts := strings.Split(tagValue, ",")
	if len(parts) != 3 {
		return "", "", "", false, fmt.Errorf(`%s value must be "<group>,<version>,<kind>", got %q`, replacementTagName, tagValue)
	}
	group, version, kind = parts[0], parts[1], parts[2]
	if len(version) == 0 || len(kind) == 0 {
		return "", "", "", false, fmt.Errorf(`%s value must be "<group>,<version>,<kind>", got %q`, replacementTagName, tagValue)
	}
	// sanity check the group
	if strings.ToLower(group) != group {
		return "", "", "", false, fmt.Errorf(`replacement group must be all lower-case, got %q`, group)
	}
	// sanity check the version
	if !strings.HasPrefix(version, "v") || strings.ToLower(version) != version {
		return "", "", "", false, fmt.Errorf(`replacement version must start with "v" and be all lower-case, got %q`, version)
	}
	// sanity check the kind
	if strings.ToUpper(kind[:1]) != kind[:1] {
		return "", "", "", false, fmt.Errorf(`replacement kind must start with uppercase-letter, got %q`, kind)
	}
	return group, version, kind, true, nil
}

func extractTag(tagName string, comments []string) *tagValue {
	tagVals := gengo.ExtractCommentTags("+", comments)[tagName]
	if tagVals == nil {
		// No match for the tag.
		return nil
	}
	// If there are multiple values, abort.
	if len(tagVals) > 1 {
		klog.Fatalf("Found %d %s tags: %q", len(tagVals), tagName, tagVals)
	}

	// If we got here we are returning something.
	tag := &tagValue{}

	// Get the primary value.
	parts := strings.Split(tagVals[0], ",")
	if len(parts) >= 1 {
		tag.value = parts[0]
	}

	// Parse extra arguments.
	parts = parts[1:]
	for i := range parts {
		kv := strings.SplitN(parts[i], "=", 2)
		k := kv[0]
		if k != "" {
			klog.Fatalf("Unsupported %s param: %q", tagName, parts[i])
		}
	}
	return tag
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

		ptag := extractTag(tagEnabledName, pkg.Comments)
		pkgNeedsGeneration := false
		if ptag != nil {
			pkgNeedsGeneration, err = strconv.ParseBool(ptag.value)
			if err != nil {
				klog.Fatalf("Package %v: unsupported %s value: %q :%v", i, tagEnabledName, ptag.value, err)
			}
		}
		if !pkgNeedsGeneration {
			klog.V(5).Infof("  skipping package")
			continue
		}
		klog.V(3).Infof("Generating package %q", pkg.Path)

		// If the pkg-scoped tag says to generate, we can skip scanning types.
		if !pkgNeedsGeneration {
			// If the pkg-scoped tag did not exist, scan all types for one that
			// explicitly wants generation.
			for _, t := range pkg.Types {
				klog.V(5).Infof("  considering type %q", t.Name.String())
				ttag := extractEnabledTypeTag(t)
				if ttag != nil && ttag.value == "true" {
					klog.V(5).Infof("    tag=true")
					if !isAPIType(t) {
						klog.Fatalf("Type %v requests prerelease generation but is not an API type", t)
					}
					pkgNeedsGeneration = true
					break
				}
			}
		}

		if pkgNeedsGeneration {
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
							NewPrereleaseLifecycleGen(args.OutputFile, pkg.Path),
						}
					},
				})
		}
	}
	return targets
}

// genDeepCopy produces a file with autogenerated deep-copy functions.
type genPreleaseLifecycle struct {
	generator.GoGenerator
	targetPackage string
	imports       namer.ImportTracker
	typesForInit  []*types.Type
}

// NewPrereleaseLifecycleGen creates a generator for the prerelease-lifecycle-generator
func NewPrereleaseLifecycleGen(outputFilename, targetPackage string) generator.Generator {
	return &genPreleaseLifecycle{
		GoGenerator: generator.GoGenerator{
			OutputFilename: outputFilename,
		},
		targetPackage: targetPackage,
		imports:       generator.NewImportTracker(),
		typesForInit:  make([]*types.Type, 0),
	}
}

func (g *genPreleaseLifecycle) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"public":       namer.NewPublicNamer(1),
		"intrapackage": namer.NewPublicNamer(0),
		"raw":          namer.NewRawNamer("", nil),
	}
}

func (g *genPreleaseLifecycle) Filter(c *generator.Context, t *types.Type) bool {
	// Filter out types not being processed or not copyable within the package.
	if !isAPIType(t) {
		klog.V(2).Infof("Type %v is not a valid target for status", t)
		return false
	}
	g.typesForInit = append(g.typesForInit, t)
	return true
}

// versionMethod returns the signature of an <methodName>() method, nil or an error
// if the type is wrong. Introduced() allows more efficient deep copy
// implementations to be defined by the type's author.  The correct signature
//
//	func (t *T) <methodName>() string
func versionMethod(methodName string, t *types.Type) (*types.Signature, error) {
	f, found := t.Methods[methodName]
	if !found {
		return nil, nil
	}
	if len(f.Signature.Parameters) != 0 {
		return nil, fmt.Errorf("type %v: invalid  %v signature, expected no parameters", t, methodName)
	}
	if len(f.Signature.Results) != 2 {
		return nil, fmt.Errorf("type %v: invalid  %v signature, expected exactly two result types", t, methodName)
	}

	ptrRcvr := f.Signature.Receiver != nil && f.Signature.Receiver.Kind == types.Pointer && f.Signature.Receiver.Elem.Name == t.Name
	nonPtrRcvr := f.Signature.Receiver != nil && f.Signature.Receiver.Name == t.Name

	if !ptrRcvr && !nonPtrRcvr {
		// this should never happen
		return nil, fmt.Errorf("type %v: invalid %v signature, expected a receiver of type %s or *%s", t, methodName, t.Name.Name, t.Name.Name)
	}

	return f.Signature, nil
}

// versionedMethodOrDie returns the signature of a <methodName>() method, nil or calls klog.Fatalf
// if the type is wrong.
func versionedMethodOrDie(methodName string, t *types.Type) *types.Signature {
	ret, err := versionMethod(methodName, t)
	if err != nil {
		klog.Fatal(err)
	}
	return ret
}

// isAPIType indicates whether or not a type could be used to serve an API.  That means, "does it have TypeMeta".
// This doesn't mean the type is served, but we will handle all TypeMeta types.
func isAPIType(t *types.Type) bool {
	// Filter out private types.
	if namer.IsPrivateGoName(t.Name.Name) {
		return false
	}

	if t.Kind != types.Struct {
		return false
	}

	for _, currMember := range t.Members {
		if currMember.Embedded && currMember.Name == "TypeMeta" {
			return true
		}
	}

	if t.Kind == types.Alias {
		return isAPIType(t.Underlying)
	}

	return false
}

func (g *genPreleaseLifecycle) isOtherPackage(pkg string) bool {
	if pkg == g.targetPackage {
		return false
	}
	if strings.HasSuffix(pkg, "\""+g.targetPackage+"\"") {
		return false
	}
	return true
}

func (g *genPreleaseLifecycle) Imports(c *generator.Context) (imports []string) {
	importLines := []string{}
	for _, singleImport := range g.imports.ImportLines() {
		if g.isOtherPackage(singleImport) {
			importLines = append(importLines, singleImport)
		}
	}
	return importLines
}

func (g *genPreleaseLifecycle) argsFromType(c *generator.Context, t *types.Type) (generator.Args, error) {
	a := generator.Args{
		"type": t,
	}
	_, introducedMajor, introducedMinor, err := extractIntroducedTag(t)
	if err != nil {
		return nil, err
	}
	a = a.
		With("introducedMajor", introducedMajor).
		With("introducedMinor", introducedMinor)

	// compute based on our policy
	deprecatedMajor := introducedMajor
	deprecatedMinor := introducedMinor + 3
	// if someone intentionally override the deprecation release
	if tagExists(deprecatedTagName, t) {
		_, deprecatedMajor, deprecatedMinor, err = extractDeprecatedTag(t)
		if err != nil {
			return nil, err
		}
	}
	a = a.
		With("deprecatedMajor", deprecatedMajor).
		With("deprecatedMinor", deprecatedMinor)

	// compute based on our policy
	removedMajor := deprecatedMajor
	removedMinor := deprecatedMinor + 3
	// if someone intentionally override the removed release
	if tagExists(removedTagName, t) {
		_, removedMajor, removedMinor, err = extractRemovedTag(t)
		if err != nil {
			return nil, err
		}
	}
	a = a.
		With("removedMajor", removedMajor).
		With("removedMinor", removedMinor)

	replacementGroup, replacementVersion, replacementKind, hasReplacement, err := extractReplacementTag(t)
	if err != nil {
		return nil, err
	}
	if hasReplacement {
		gvkType := c.Universe.Type(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/schema", Name: "GroupVersionKind"})
		g.imports.AddType(gvkType)
		a = a.
			With("replacementGroup", replacementGroup).
			With("replacementVersion", replacementVersion).
			With("replacementKind", replacementKind).
			With("GroupVersionKind", gvkType)
	}

	return a, nil
}

func (g *genPreleaseLifecycle) Init(c *generator.Context, w io.Writer) error {
	return nil
}

func (g *genPreleaseLifecycle) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	klog.V(3).Infof("Generating prerelease-lifecycle for type %v", t)

	sw := generator.NewSnippetWriter(w, c, "$", "$")
	args, err := g.argsFromType(c, t)
	if err != nil {
		return err
	}

	if versionedMethodOrDie("APILifecycleIntroduced", t) == nil {
		sw.Do("// APILifecycleIntroduced is an autogenerated function, returning the release in which the API struct was introduced as int versions of major and minor for comparison.\n", args)
		sw.Do("// It is controlled by \""+introducedTagName+"\" tags in types.go.\n", args)
		sw.Do("func (in *$.type|intrapackage$) APILifecycleIntroduced() (major, minor int) {\n", args)
		sw.Do("    return $.introducedMajor$, $.introducedMinor$\n", args)
		sw.Do("}\n\n", nil)
	}
	if versionedMethodOrDie("APILifecycleDeprecated", t) == nil {
		sw.Do("// APILifecycleDeprecated is an autogenerated function, returning the release in which the API struct was or will be deprecated as int versions of major and minor for comparison.\n", args)
		sw.Do("// It is controlled by \""+deprecatedTagName+"\" tags in types.go or  \""+introducedTagName+"\" plus three minor.\n", args)
		sw.Do("func (in *$.type|intrapackage$) APILifecycleDeprecated() (major, minor int) {\n", args)
		sw.Do("    return $.deprecatedMajor$, $.deprecatedMinor$\n", args)
		sw.Do("}\n\n", nil)
	}
	if _, hasReplacement := args["replacementKind"]; hasReplacement {
		if versionedMethodOrDie("APILifecycleReplacement", t) == nil {
			sw.Do("// APILifecycleReplacement is an autogenerated function, returning the group, version, and kind that should be used instead of this deprecated type.\n", args)
			sw.Do("// It is controlled by \""+replacementTagName+"=<group>,<version>,<kind>\" tags in types.go.\n", args)
			sw.Do("func (in *$.type|intrapackage$) APILifecycleReplacement() ($.GroupVersionKind|raw$) {\n", args)
			sw.Do("    return $.GroupVersionKind|raw${Group:\"$.replacementGroup$\", Version:\"$.replacementVersion$\", Kind:\"$.replacementKind$\"}\n", args)
			sw.Do("}\n\n", nil)
		}
	}
	if versionedMethodOrDie("APILifecycleRemoved", t) == nil {
		sw.Do("// APILifecycleRemoved is an autogenerated function, returning the release in which the API is no longer served as int versions of major and minor for comparison.\n", args)
		sw.Do("// It is controlled by \""+removedTagName+"\" tags in types.go or  \""+deprecatedTagName+"\" plus three minor.\n", args)
		sw.Do("func (in *$.type|intrapackage$) APILifecycleRemoved() (major, minor int) {\n", args)
		sw.Do("    return $.removedMajor$, $.removedMinor$\n", args)
		sw.Do("}\n\n", nil)
	}

	return sw.Error()
}
