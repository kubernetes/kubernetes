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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"path"
	"reflect"
	"regexp"
	"sort"
	"strings"

	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	openapi "k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"

	"k8s.io/klog/v2"
)

// This is the comment tag that carries parameters for open API generation.
const tagName = "k8s:openapi-gen"
const markerPrefix = "+k8s:validation:"
const tagOptional = "optional"
const tagRequired = "required"
const tagDefault = "default"

// Known values for the tag.
const (
	tagValueTrue  = "true"
	tagValueFalse = "false"
)

// Used for temporary validation of patch struct tags.
// TODO: Remove patch struct tag validation because they we are now consuming OpenAPI on server.
var tempPatchTags = [...]string{
	"patchMergeKey",
	"patchStrategy",
}

func getOpenAPITagValue(comments []string) []string {
	return gengo.ExtractCommentTags("+", comments)[tagName]
}

func getSingleTagsValue(comments []string, tag string) (string, error) {
	tags, ok := gengo.ExtractCommentTags("+", comments)[tag]
	if !ok || len(tags) == 0 {
		return "", nil
	}
	if len(tags) > 1 {
		return "", fmt.Errorf("multiple values are not allowed for tag %s", tag)
	}
	return tags[0], nil
}

func hasOpenAPITagValue(comments []string, value string) bool {
	tagValues := getOpenAPITagValue(comments)
	for _, val := range tagValues {
		if val == value {
			return true
		}
	}
	return false
}

// isOptional returns error if the member has +optional and +required in
// its comments. If +optional is present it returns true. If +required is present
// it returns false. Otherwise, it returns true if `omitempty` JSON tag is present
func isOptional(m *types.Member) (bool, error) {
	hasOptionalCommentTag := gengo.ExtractCommentTags(
		"+", m.CommentLines)[tagOptional] != nil
	hasRequiredCommentTag := gengo.ExtractCommentTags(
		"+", m.CommentLines)[tagRequired] != nil
	if hasOptionalCommentTag && hasRequiredCommentTag {
		return false, fmt.Errorf("member %s cannot be both optional and required", m.Name)
	} else if hasRequiredCommentTag {
		return false, nil
	} else if hasOptionalCommentTag {
		return true, nil
	}

	// If neither +optional nor +required is present in the comments,
	// infer optional from the json tags.
	return strings.Contains(reflect.StructTag(m.Tags).Get("json"), "omitempty"), nil
}

func apiTypeFilterFunc(c *generator.Context, t *types.Type) bool {
	// There is a conflict between this codegen and codecgen, we should avoid types generated for codecgen
	if strings.HasPrefix(t.Name.Name, "codecSelfer") {
		return false
	}
	pkg := c.Universe.Package(t.Name.Package)
	if hasOpenAPITagValue(pkg.Comments, tagValueTrue) {
		return !hasOpenAPITagValue(t.CommentLines, tagValueFalse)
	}
	if hasOpenAPITagValue(t.CommentLines, tagValueTrue) {
		return true
	}
	return false
}

const (
	specPackagePath          = "k8s.io/kube-openapi/pkg/validation/spec"
	openAPICommonPackagePath = "k8s.io/kube-openapi/pkg/common"
)

// openApiGen produces a file with auto-generated OpenAPI functions.
type openAPIGen struct {
	generator.GoGenerator
	// TargetPackage is the package that will get GetOpenAPIDefinitions function returns all open API definitions.
	targetPackage string
	imports       namer.ImportTracker
}

func newOpenAPIGen(outputFilename string, targetPackage string) generator.Generator {
	return &openAPIGen{
		GoGenerator: generator.GoGenerator{
			OutputFilename: outputFilename,
		},
		imports:       generator.NewImportTrackerForPackage(targetPackage),
		targetPackage: targetPackage,
	}
}

const nameTmpl = "schema_$.type|private$"

func (g *openAPIGen) Namers(c *generator.Context) namer.NameSystems {
	// Have the raw namer for this file track what it imports.
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.targetPackage, g.imports),
		"private": &namer.NameStrategy{
			Join: func(pre string, in []string, post string) string {
				return strings.Join(in, "_")
			},
			PrependPackageNames: 4, // enough to fully qualify from k8s.io/api/...
		},
	}
}

func (g *openAPIGen) Imports(c *generator.Context) []string {
	importLines := []string{}
	for _, singleImport := range g.imports.ImportLines() {
		importLines = append(importLines, singleImport)
	}
	return importLines
}

func argsFromType(t *types.Type) generator.Args {
	return generator.Args{
		"type":              t,
		"ReferenceCallback": types.Ref(openAPICommonPackagePath, "ReferenceCallback"),
		"OpenAPIDefinition": types.Ref(openAPICommonPackagePath, "OpenAPIDefinition"),
		"SpecSchemaType":    types.Ref(specPackagePath, "Schema"),
	}
}

func (g *openAPIGen) Init(c *generator.Context, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	sw.Do("func GetOpenAPIDefinitions(ref $.ReferenceCallback|raw$) map[string]$.OpenAPIDefinition|raw$ {\n", argsFromType(nil))
	sw.Do("return map[string]$.OpenAPIDefinition|raw${\n", argsFromType(nil))

	for _, t := range c.Order {
		err := newOpenAPITypeWriter(sw, c).generateCall(t)
		if err != nil {
			return err
		}
	}

	sw.Do("}\n", nil)
	sw.Do("}\n\n", nil)

	return sw.Error()
}

func (g *openAPIGen) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	klog.V(5).Infof("generating for type %v", t)
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	err := newOpenAPITypeWriter(sw, c).generate(t)
	if err != nil {
		return err
	}
	return sw.Error()
}

func getJsonTags(m *types.Member) []string {
	jsonTag := reflect.StructTag(m.Tags).Get("json")
	if jsonTag == "" {
		return []string{}
	}
	return strings.Split(jsonTag, ",")
}

func getReferableName(m *types.Member) string {
	jsonTags := getJsonTags(m)
	if len(jsonTags) > 0 {
		if jsonTags[0] == "-" {
			return ""
		} else {
			return jsonTags[0]
		}
	} else {
		return m.Name
	}
}

func shouldInlineMembers(m *types.Member) bool {
	jsonTags := getJsonTags(m)
	return len(jsonTags) > 1 && jsonTags[1] == "inline"
}

type openAPITypeWriter struct {
	*generator.SnippetWriter
	context                *generator.Context
	refTypes               map[string]*types.Type
	enumContext            *enumContext
	GetDefinitionInterface *types.Type
}

func newOpenAPITypeWriter(sw *generator.SnippetWriter, c *generator.Context) openAPITypeWriter {
	return openAPITypeWriter{
		SnippetWriter: sw,
		context:       c,
		refTypes:      map[string]*types.Type{},
		enumContext:   newEnumContext(c),
	}
}

func methodReturnsValue(mt *types.Type, pkg, name string) bool {
	if len(mt.Signature.Parameters) != 0 || len(mt.Signature.Results) != 1 {
		return false
	}
	r := mt.Signature.Results[0]
	return r.Type.Name.Name == name && r.Type.Name.Package == pkg
}

func hasOpenAPIV3DefinitionMethod(t *types.Type) bool {
	for mn, mt := range t.Methods {
		if mn != "OpenAPIV3Definition" {
			continue
		}
		return methodReturnsValue(mt, openAPICommonPackagePath, "OpenAPIDefinition")
	}
	return false
}

func hasOpenAPIDefinitionMethod(t *types.Type) bool {
	for mn, mt := range t.Methods {
		if mn != "OpenAPIDefinition" {
			continue
		}
		return methodReturnsValue(mt, openAPICommonPackagePath, "OpenAPIDefinition")
	}
	return false
}

func hasOpenAPIDefinitionMethods(t *types.Type) bool {
	var hasSchemaTypeMethod, hasOpenAPISchemaFormat bool
	for mn, mt := range t.Methods {
		switch mn {
		case "OpenAPISchemaType":
			hasSchemaTypeMethod = methodReturnsValue(mt, "", "[]string")
		case "OpenAPISchemaFormat":
			hasOpenAPISchemaFormat = methodReturnsValue(mt, "", "string")
		}
	}
	return hasSchemaTypeMethod && hasOpenAPISchemaFormat
}

func hasOpenAPIV3OneOfMethod(t *types.Type) bool {
	for mn, mt := range t.Methods {
		if mn != "OpenAPIV3OneOfTypes" {
			continue
		}
		return methodReturnsValue(mt, "", "[]string")
	}
	return false
}

// typeShortName returns short package name (e.g. the name x appears in package x definition) dot type name.
func typeShortName(t *types.Type) string {
	// `path` vs. `filepath` because packages use '/'
	return path.Base(t.Name.Package) + "." + t.Name.Name
}

func (g openAPITypeWriter) generateMembers(t *types.Type, required []string) ([]string, error) {
	var err error
	for t.Kind == types.Pointer { // fast-forward to effective type containing members
		t = t.Elem
	}
	for _, m := range t.Members {
		if hasOpenAPITagValue(m.CommentLines, tagValueFalse) {
			continue
		}
		if shouldInlineMembers(&m) {
			required, err = g.generateMembers(m.Type, required)
			if err != nil {
				return required, err
			}
			continue
		}
		name := getReferableName(&m)
		if name == "" {
			continue
		}
		if isOptional, err := isOptional(&m); err != nil {
			klog.Errorf("Error when generating: %v, %v\n", name, m)
			return required, err
		} else if !isOptional {
			required = append(required, name)
		}
		if err = g.generateProperty(&m, t); err != nil {
			klog.Errorf("Error when generating: %v, %v\n", name, m)
			return required, err
		}
	}
	return required, nil
}

func (g openAPITypeWriter) generateCall(t *types.Type) error {
	// Only generate for struct type and ignore the rest
	switch t.Kind {
	case types.Struct:
		args := argsFromType(t)
		g.Do("\"$.$\": ", t.Name)

		hasV2Definition := hasOpenAPIDefinitionMethod(t)
		hasV2DefinitionTypeAndFormat := hasOpenAPIDefinitionMethods(t)
		hasV3Definition := hasOpenAPIV3DefinitionMethod(t)

		switch {
		case hasV2DefinitionTypeAndFormat:
			g.Do(nameTmpl+"(ref),\n", args)
		case hasV2Definition && hasV3Definition:
			g.Do("common.EmbedOpenAPIDefinitionIntoV2Extension($.type|raw${}.OpenAPIV3Definition(), $.type|raw${}.OpenAPIDefinition()),\n", args)
		case hasV2Definition:
			g.Do("$.type|raw${}.OpenAPIDefinition(),\n", args)
		case hasV3Definition:
			g.Do("$.type|raw${}.OpenAPIV3Definition(),\n", args)
		default:
			g.Do(nameTmpl+"(ref),\n", args)
		}
	}
	return g.Error()
}

// Generates Go code to represent an OpenAPI schema. May be refactored in
// the future to take more responsibility as we transition from an on-line
// approach to parsing the comments to spec.Schema
func (g openAPITypeWriter) generateSchema(s *spec.Schema) error {
	if !reflect.DeepEqual(s.SchemaProps, spec.SchemaProps{}) {
		g.Do("SchemaProps: spec.SchemaProps{\n", nil)
		err := g.generateValueValidations(&s.SchemaProps)
		if err != nil {
			return err
		}

		if len(s.Properties) > 0 {
			g.Do("Properties: map[string]spec.Schema{\n", nil)

			// Sort property names to generate deterministic output
			keys := []string{}
			for k := range s.Properties {
				keys = append(keys, k)
			}
			sort.Strings(keys)

			for _, k := range keys {
				v := s.Properties[k]
				g.Do("$.$: {\n", fmt.Sprintf("%#v", k))
				err := g.generateSchema(&v)
				if err != nil {
					return err
				}
				g.Do("},\n", nil)
			}
			g.Do("},\n", nil)
		}

		if s.AdditionalProperties != nil && s.AdditionalProperties.Schema != nil {
			g.Do("AdditionalProperties: &spec.SchemaOrBool{\n", nil)
			g.Do("Allows: true,\n", nil)
			g.Do("Schema: &spec.Schema{\n", nil)
			err := g.generateSchema(s.AdditionalProperties.Schema)
			if err != nil {
				return err
			}
			g.Do("},\n", nil)
			g.Do("},\n", nil)
		}

		if s.Items != nil && s.Items.Schema != nil {
			g.Do("Items: &spec.SchemaOrArray{\n", nil)
			g.Do("Schema: &spec.Schema{\n", nil)
			err := g.generateSchema(s.Items.Schema)
			if err != nil {
				return err
			}
			g.Do("},\n", nil)
			g.Do("},\n", nil)
		}

		g.Do("},\n", nil)
	}

	if len(s.Extensions) > 0 {
		g.Do("VendorExtensible: spec.VendorExtensible{\nExtensions: spec.Extensions{\n", nil)

		// Sort extension keys to generate deterministic output
		keys := []string{}
		for k := range s.Extensions {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		for _, k := range keys {
			v := s.Extensions[k]
			g.Do("$.key$: $.value$,\n", map[string]interface{}{
				"key":   fmt.Sprintf("%#v", k),
				"value": fmt.Sprintf("%#v", v),
			})
		}
		g.Do("},\n},\n", nil)
	}

	return nil
}

func (g openAPITypeWriter) generateValueValidations(vs *spec.SchemaProps) error {

	if vs == nil {
		return nil
	}
	args := generator.Args{
		"ptrTo": &types.Type{
			Name: types.Name{
				Package: "k8s.io/utils/ptr",
				Name:    "To",
			}},
		"spec": vs,
	}
	if vs.Minimum != nil {
		g.Do("Minimum: $.ptrTo|raw$[float64]($.spec.Minimum$),\n", args)
	}
	if vs.Maximum != nil {
		g.Do("Maximum: $.ptrTo|raw$[float64]($.spec.Maximum$),\n", args)
	}
	if vs.ExclusiveMinimum {
		g.Do("ExclusiveMinimum: true,\n", args)
	}
	if vs.ExclusiveMaximum {
		g.Do("ExclusiveMaximum: true,\n", args)
	}
	if vs.MinLength != nil {
		g.Do("MinLength: $.ptrTo|raw$[int64]($.spec.MinLength$),\n", args)
	}
	if vs.MaxLength != nil {
		g.Do("MaxLength: $.ptrTo|raw$[int64]($.spec.MaxLength$),\n", args)
	}

	if vs.MinProperties != nil {
		g.Do("MinProperties: $.ptrTo|raw$[int64]($.spec.MinProperties$),\n", args)
	}
	if vs.MaxProperties != nil {
		g.Do("MaxProperties: $.ptrTo|raw$[int64]($.spec.MaxProperties$),\n", args)
	}
	if len(vs.Pattern) > 0 {
		p, err := json.Marshal(vs.Pattern)
		if err != nil {
			return err
		}
		g.Do("Pattern: $.$,\n", string(p))
	}
	if vs.MultipleOf != nil {
		g.Do("MultipleOf: $.ptrTo|raw$[float64]($.spec.MultipleOf$),\n", args)
	}
	if vs.MinItems != nil {
		g.Do("MinItems: $.ptrTo|raw$[int64]($.spec.MinItems$),\n", args)
	}
	if vs.MaxItems != nil {
		g.Do("MaxItems: $.ptrTo|raw$[int64]($.spec.MaxItems$),\n", args)
	}
	if vs.UniqueItems {
		g.Do("UniqueItems: true,\n", nil)
	}

	if len(vs.AllOf) > 0 {
		g.Do("AllOf: []spec.Schema{\n", nil)
		for _, s := range vs.AllOf {
			g.Do("{\n", nil)
			if err := g.generateSchema(&s); err != nil {
				return err
			}
			g.Do("},\n", nil)
		}
		g.Do("},\n", nil)
	}

	return nil
}

func (g openAPITypeWriter) generate(t *types.Type) error {
	// Only generate for struct type and ignore the rest
	switch t.Kind {
	case types.Struct:
		validationSchema, err := ParseCommentTags(t, t.CommentLines, markerPrefix)
		if err != nil {
			return fmt.Errorf("failed parsing comment tags for %v: %w", t.String(), err)
		}

		hasV2Definition := hasOpenAPIDefinitionMethod(t)
		hasV2DefinitionTypeAndFormat := hasOpenAPIDefinitionMethods(t)
		hasV3OneOfTypes := hasOpenAPIV3OneOfMethod(t)
		hasV3Definition := hasOpenAPIV3DefinitionMethod(t)

		if hasV2Definition || (hasV3Definition && !hasV2DefinitionTypeAndFormat) {
			// already invoked directly
			return nil
		}

		args := argsFromType(t)
		g.Do("func "+nameTmpl+"(ref $.ReferenceCallback|raw$) $.OpenAPIDefinition|raw$ {\n", args)
		switch {
		case hasV2DefinitionTypeAndFormat && hasV3Definition:
			g.Do("return common.EmbedOpenAPIDefinitionIntoV2Extension($.type|raw${}.OpenAPIV3Definition(), $.OpenAPIDefinition|raw${\n"+
				"Schema: spec.Schema{\n"+
				"SchemaProps: spec.SchemaProps{\n", args)
			g.generateDescription(t.CommentLines)
			g.Do("Type:$.type|raw${}.OpenAPISchemaType(),\n"+
				"Format:$.type|raw${}.OpenAPISchemaFormat(),\n", args)
			err = g.generateValueValidations(&validationSchema.SchemaProps)
			if err != nil {
				return err
			}
			g.Do("},\n", nil)
			if err := g.generateStructExtensions(t, validationSchema.Extensions); err != nil {
				return err
			}
			g.Do("},\n", nil)
			g.Do("})\n}\n\n", args)
			return nil
		case hasV2DefinitionTypeAndFormat && hasV3OneOfTypes:
			// generate v3 def.
			g.Do("return common.EmbedOpenAPIDefinitionIntoV2Extension($.OpenAPIDefinition|raw${\n"+
				"Schema: spec.Schema{\n"+
				"SchemaProps: spec.SchemaProps{\n", args)
			g.generateDescription(t.CommentLines)
			g.Do("OneOf:common.GenerateOpenAPIV3OneOfSchema($.type|raw${}.OpenAPIV3OneOfTypes()),\n"+
				"Format:$.type|raw${}.OpenAPISchemaFormat(),\n", args)
			err = g.generateValueValidations(&validationSchema.SchemaProps)
			if err != nil {
				return err
			}
			g.Do("},\n", nil)
			if err := g.generateStructExtensions(t, validationSchema.Extensions); err != nil {
				return err
			}
			g.Do("},\n", nil)
			g.Do("},", args)
			// generate v2 def.
			g.Do("$.OpenAPIDefinition|raw${\n"+
				"Schema: spec.Schema{\n"+
				"SchemaProps: spec.SchemaProps{\n", args)
			g.generateDescription(t.CommentLines)
			g.Do("Type:$.type|raw${}.OpenAPISchemaType(),\n"+
				"Format:$.type|raw${}.OpenAPISchemaFormat(),\n", args)
			err = g.generateValueValidations(&validationSchema.SchemaProps)
			if err != nil {
				return err
			}
			g.Do("},\n", nil)
			if err := g.generateStructExtensions(t, validationSchema.Extensions); err != nil {
				return err
			}
			g.Do("},\n", nil)
			g.Do("})\n}\n\n", args)
			return nil
		case hasV2DefinitionTypeAndFormat:
			g.Do("return $.OpenAPIDefinition|raw${\n"+
				"Schema: spec.Schema{\n"+
				"SchemaProps: spec.SchemaProps{\n", args)
			g.generateDescription(t.CommentLines)
			g.Do("Type:$.type|raw${}.OpenAPISchemaType(),\n"+
				"Format:$.type|raw${}.OpenAPISchemaFormat(),\n", args)
			err = g.generateValueValidations(&validationSchema.SchemaProps)
			if err != nil {
				return err
			}
			g.Do("},\n", nil)
			if err := g.generateStructExtensions(t, validationSchema.Extensions); err != nil {
				return err
			}
			g.Do("},\n", nil)
			g.Do("}\n}\n\n", args)
			return nil
		case hasV3OneOfTypes:
			// having v3 oneOf types without custom v2 type or format does not make sense.
			return fmt.Errorf("type %q has v3 one of types but not v2 type or format", t.Name)
		}

		g.Do("return $.OpenAPIDefinition|raw${\nSchema: spec.Schema{\nSchemaProps: spec.SchemaProps{\n", args)
		g.generateDescription(t.CommentLines)
		g.Do("Type: []string{\"object\"},\n", nil)
		err = g.generateValueValidations(&validationSchema.SchemaProps)
		if err != nil {
			return err
		}

		// write members into a temporary buffer, in order to postpone writing out the Properties field. We only do
		// that if it is not empty.
		propertiesBuf := bytes.Buffer{}
		bsw := g
		bsw.SnippetWriter = generator.NewSnippetWriter(&propertiesBuf, g.context, "$", "$")
		required, err := bsw.generateMembers(t, []string{})
		if err != nil {
			return err
		}
		if propertiesBuf.Len() > 0 {
			g.Do("Properties: map[string]$.SpecSchemaType|raw${\n", args)
			g.Do(strings.Replace(propertiesBuf.String(), "$", "$\"$\"$", -1), nil) // escape $ (used as delimiter of the templates)
			g.Do("},\n", nil)
		}

		if len(required) > 0 {
			g.Do("Required: []string{\"$.$\"},\n", strings.Join(required, "\",\""))
		}
		g.Do("},\n", nil)
		if err := g.generateStructExtensions(t, validationSchema.Extensions); err != nil {
			return err
		}
		g.Do("},\n", nil)

		// Map order is undefined, sort them or we may get a different file generated each time.
		keys := []string{}
		for k := range g.refTypes {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		deps := []string{}
		for _, k := range keys {
			v := g.refTypes[k]
			if t, _ := openapi.OpenAPITypeFormat(v.String()); t != "" {
				// This is a known type, we do not need a reference to it
				// Will eliminate special case of time.Time
				continue
			}
			deps = append(deps, k)
		}
		if len(deps) > 0 {
			g.Do("Dependencies: []string{\n", args)
			for _, k := range deps {
				g.Do("\"$.$\",", k)
			}
			g.Do("},\n", nil)
		}
		g.Do("}\n}\n\n", nil)
	}
	return nil
}

func (g openAPITypeWriter) generateStructExtensions(t *types.Type, otherExtensions map[string]interface{}) error {
	extensions, errors := parseExtensions(t.CommentLines)
	// Initially, we will only log struct extension errors.
	if len(errors) > 0 {
		for _, e := range errors {
			klog.Errorf("[%s]: %s\n", t.String(), e)
		}
	}
	unions, errors := parseUnions(t)
	if len(errors) > 0 {
		for _, e := range errors {
			klog.Errorf("[%s]: %s\n", t.String(), e)
		}
	}

	// TODO(seans3): Validate struct extensions here.
	g.emitExtensions(extensions, unions, otherExtensions)
	return nil
}

func (g openAPITypeWriter) generateMemberExtensions(m *types.Member, parent *types.Type, otherExtensions map[string]interface{}) error {
	extensions, parseErrors := parseExtensions(m.CommentLines)
	validationErrors := validateMemberExtensions(extensions, m)
	errors := append(parseErrors, validationErrors...)
	// Initially, we will only log member extension errors.
	if len(errors) > 0 {
		errorPrefix := fmt.Sprintf("[%s] %s:", parent.String(), m.String())
		for _, e := range errors {
			klog.V(2).Infof("%s %s\n", errorPrefix, e)
		}
	}
	g.emitExtensions(extensions, nil, otherExtensions)
	return nil
}

func (g openAPITypeWriter) emitExtensions(extensions []extension, unions []union, otherExtensions map[string]interface{}) {
	// If any extensions exist, then emit code to create them.
	if len(extensions) == 0 && len(unions) == 0 && len(otherExtensions) == 0 {
		return
	}
	g.Do("VendorExtensible: spec.VendorExtensible{\nExtensions: spec.Extensions{\n", nil)
	for _, extension := range extensions {
		g.Do("\"$.$\": ", extension.xName)
		if extension.hasMultipleValues() || extension.isAlwaysArrayFormat() {
			g.Do("[]interface{}{\n", nil)
		}
		for _, value := range extension.values {
			g.Do("\"$.$\",\n", value)
		}
		if extension.hasMultipleValues() || extension.isAlwaysArrayFormat() {
			g.Do("},\n", nil)
		}
	}
	if len(unions) > 0 {
		g.Do("\"x-kubernetes-unions\": []interface{}{\n", nil)
		for _, u := range unions {
			u.emit(g)
		}
		g.Do("},\n", nil)
	}

	if len(otherExtensions) > 0 {
		// Sort extension keys to generate deterministic output
		keys := []string{}
		for k := range otherExtensions {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		for _, k := range keys {
			v := otherExtensions[k]
			g.Do("$.key$: $.value$,\n", map[string]interface{}{
				"key":   fmt.Sprintf("%#v", k),
				"value": fmt.Sprintf("%#v", v),
			})
		}
	}

	g.Do("},\n},\n", nil)
}

// TODO(#44005): Move this validation outside of this generator (probably to policy verifier)
func (g openAPITypeWriter) validatePatchTags(m *types.Member, parent *types.Type) error {
	// TODO: Remove patch struct tag validation because they we are now consuming OpenAPI on server.
	for _, tagKey := range tempPatchTags {
		structTagValue := reflect.StructTag(m.Tags).Get(tagKey)
		commentTagValue, err := getSingleTagsValue(m.CommentLines, tagKey)
		if err != nil {
			return err
		}
		if structTagValue != commentTagValue {
			return fmt.Errorf("Tags in comment and struct should match for member (%s) of (%s)",
				m.Name, parent.Name.String())
		}
	}
	return nil
}

func defaultFromComments(comments []string, commentPath string, t *types.Type) (interface{}, *types.Name, error) {
	var tag string

	for {
		var err error
		tag, err = getSingleTagsValue(comments, tagDefault)
		if err != nil {
			return nil, nil, err
		}

		if t == nil || len(tag) > 0 {
			break
		}

		comments = t.CommentLines
		commentPath = t.Name.Package
		switch t.Kind {
		case types.Pointer:
			t = t.Elem
		case types.Alias:
			t = t.Underlying
		default:
			t = nil
		}
	}

	if tag == "" {
		return nil, nil, nil
	}

	var i interface{}
	if id, ok := parseSymbolReference(tag, commentPath); ok {
		klog.V(5).Infof("%v, %v", id, commentPath)
		return nil, &id, nil
	} else if err := json.Unmarshal([]byte(tag), &i); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal default: %v", err)
	}
	return i, nil, nil
}

var refRE = regexp.MustCompile(`^ref\((?P<reference>[^"]+)\)$`)
var refREIdentIndex = refRE.SubexpIndex("reference")

// parseSymbolReference looks for strings that match one of the following:
//   - ref(Ident)
//   - ref(pkgpath.Ident)
//     If the input string matches either of these, it will return the (optional)
//     pkgpath, the Ident, and true.  Otherwise it will return empty strings and
//     false.
//
// This is borrowed from k8s.io/code-generator.
func parseSymbolReference(s, sourcePackage string) (types.Name, bool) {
	matches := refRE.FindStringSubmatch(s)
	if len(matches) < refREIdentIndex || matches[refREIdentIndex] == "" {
		return types.Name{}, false
	}

	contents := matches[refREIdentIndex]
	name := types.ParseFullyQualifiedName(contents)
	if len(name.Package) == 0 {
		name.Package = sourcePackage
	}
	return name, true
}

func implementsCustomUnmarshalling(t *types.Type) bool {
	switch t.Kind {
	case types.Pointer:
		unmarshaller, isUnmarshaller := t.Elem.Methods["UnmarshalJSON"]
		return isUnmarshaller && unmarshaller.Signature.Receiver.Kind == types.Pointer
	case types.Struct:
		_, isUnmarshaller := t.Methods["UnmarshalJSON"]
		return isUnmarshaller
	default:
		return false
	}
}

func mustEnforceDefault(t *types.Type, omitEmpty bool) (interface{}, error) {
	// Treat types with custom unmarshalling as a value
	// (Can be alias, struct, or pointer)
	if implementsCustomUnmarshalling(t) {
		// Since Go JSON deserializer always feeds `null` when present
		// to structs with custom UnmarshalJSON, the zero value for
		// these structs is also null.
		//
		// In general, Kubernetes API types with custom marshalling should
		// marshal their empty values to `null`.
		return nil, nil
	}

	switch t.Kind {
	case types.Alias:
		return mustEnforceDefault(t.Underlying, omitEmpty)
	case types.Pointer, types.Map, types.Slice, types.Array, types.Interface:
		return nil, nil
	case types.Struct:
		if len(t.Members) == 1 && t.Members[0].Embedded {
			// Treat a struct with a single embedded member the same as an alias
			return mustEnforceDefault(t.Members[0].Type, omitEmpty)
		}

		return map[string]interface{}{}, nil
	case types.Builtin:
		if !omitEmpty {
			if zero, ok := openapi.OpenAPIZeroValue(t.String()); ok {
				return zero, nil
			} else {
				return nil, fmt.Errorf("please add type %v to getOpenAPITypeFormat function", t)
			}
		}
		return nil, nil
	default:
		return nil, fmt.Errorf("not sure how to enforce default for %v", t.Kind)
	}
}

func (g openAPITypeWriter) generateDefault(comments []string, t *types.Type, omitEmpty bool, commentOwningType *types.Type) error {
	def, ref, err := defaultFromComments(comments, commentOwningType.Name.Package, t)
	if err != nil {
		return err
	}
	if enforced, err := mustEnforceDefault(t, omitEmpty); err != nil {
		return err
	} else if enforced != nil {
		if def == nil {
			def = enforced
		} else if !reflect.DeepEqual(def, enforced) {
			enforcedJson, _ := json.Marshal(enforced)
			return fmt.Errorf("invalid default value (%#v) for non-pointer/non-omitempty. If specified, must be: %v", def, string(enforcedJson))
		}
	}
	if def != nil {
		g.Do("Default: $.$,\n", fmt.Sprintf("%#v", def))
	} else if ref != nil {
		g.Do("Default: $.|raw$,\n", &types.Type{Name: *ref})
	}
	return nil
}

func (g openAPITypeWriter) generateDescription(CommentLines []string) {
	var buffer bytes.Buffer
	delPrevChar := func() {
		if buffer.Len() > 0 {
			buffer.Truncate(buffer.Len() - 1) // Delete the last " " or "\n"
		}
	}

	for _, line := range CommentLines {
		// Ignore all lines after ---
		if line == "---" {
			break
		}
		line = strings.TrimRight(line, " ")
		leading := strings.TrimLeft(line, " ")
		switch {
		case len(line) == 0: // Keep paragraphs
			delPrevChar()
			buffer.WriteString("\n\n")
		case strings.HasPrefix(leading, "TODO"): // Ignore one line TODOs
		case strings.HasPrefix(leading, "+"): // Ignore instructions to go2idl
		default:
			if strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t") {
				delPrevChar()
				line = "\n" + line + "\n" // Replace it with newline. This is useful when we have a line with: "Example:\n\tJSON-something..."
			} else {
				line += " "
			}
			buffer.WriteString(line)
		}
	}

	postDoc := strings.TrimSpace(buffer.String())
	if len(postDoc) > 0 {
		g.Do("Description: $.$,\n", fmt.Sprintf("%#v", postDoc))
	}
}

func (g openAPITypeWriter) generateProperty(m *types.Member, parent *types.Type) error {
	name := getReferableName(m)
	if name == "" {
		return nil
	}
	validationSchema, err := ParseCommentTags(m.Type, m.CommentLines, markerPrefix)
	if err != nil {
		return err
	}
	if err := g.validatePatchTags(m, parent); err != nil {
		return err
	}
	g.Do("\"$.$\": {\n", name)
	if err := g.generateMemberExtensions(m, parent, validationSchema.Extensions); err != nil {
		return err
	}
	g.Do("SchemaProps: spec.SchemaProps{\n", nil)
	var extraComments []string
	if enumType, isEnum := g.enumContext.EnumType(m.Type); isEnum {
		extraComments = enumType.DescriptionLines()
	}
	g.generateDescription(append(m.CommentLines, extraComments...))
	jsonTags := getJsonTags(m)
	if len(jsonTags) > 1 && jsonTags[1] == "string" {
		g.generateSimpleProperty("string", "")
		g.Do("},\n},\n", nil)
		return nil
	}
	omitEmpty := strings.Contains(reflect.StructTag(m.Tags).Get("json"), "omitempty")
	if err := g.generateDefault(m.CommentLines, m.Type, omitEmpty, parent); err != nil {
		return fmt.Errorf("failed to generate default in %v: %v: %v", parent, m.Name, err)
	}
	err = g.generateValueValidations(&validationSchema.SchemaProps)
	if err != nil {
		return err
	}
	t := resolveAliasAndPtrType(m.Type)
	// If we can get a openAPI type and format for this type, we consider it to be simple property
	typeString, format := openapi.OpenAPITypeFormat(t.String())
	if typeString != "" {
		g.generateSimpleProperty(typeString, format)
		if enumType, isEnum := g.enumContext.EnumType(m.Type); isEnum {
			// original type is an enum, add "Enum: " and the values
			g.Do("Enum: []interface{}{$.$},\n", strings.Join(enumType.ValueStrings(), ", "))
		}
		g.Do("},\n},\n", nil)
		return nil
	}
	switch t.Kind {
	case types.Builtin:
		return fmt.Errorf("please add type %v to getOpenAPITypeFormat function", t)
	case types.Map:
		if err := g.generateMapProperty(t); err != nil {
			return fmt.Errorf("failed to generate map property in %v: %v: %v", parent, m.Name, err)
		}
	case types.Slice, types.Array:
		if err := g.generateSliceProperty(t); err != nil {
			return fmt.Errorf("failed to generate slice property in %v: %v: %v", parent, m.Name, err)
		}
	case types.Struct, types.Interface:
		g.generateReferenceProperty(t)
	default:
		return fmt.Errorf("cannot generate spec for type %v", t)
	}
	g.Do("},\n},\n", nil)
	return g.Error()
}

func (g openAPITypeWriter) generateSimpleProperty(typeString, format string) {
	g.Do("Type: []string{\"$.$\"},\n", typeString)
	g.Do("Format: \"$.$\",\n", format)
}

func (g openAPITypeWriter) generateReferenceProperty(t *types.Type) {
	g.refTypes[t.Name.String()] = t
	g.Do("Ref: ref(\"$.$\"),\n", t.Name.String())
}

func resolvePtrType(t *types.Type) *types.Type {
	var prev *types.Type
	for prev != t {
		prev = t
		if t.Kind == types.Pointer {
			t = t.Elem
		}
	}
	return t
}

func resolveAliasAndPtrType(t *types.Type) *types.Type {
	var prev *types.Type
	for prev != t {
		prev = t
		if t.Kind == types.Alias {
			t = t.Underlying
		}
		if t.Kind == types.Pointer {
			t = t.Elem
		}
	}
	return t
}

func (g openAPITypeWriter) generateMapProperty(t *types.Type) error {
	keyType := resolveAliasAndPtrType(t.Key)
	elemType := resolveAliasAndPtrType(t.Elem)

	// According to OpenAPI examples, only map from string is supported
	if keyType.Name.Name != "string" {
		return fmt.Errorf("map with non-string keys are not supported by OpenAPI in %v", t)
	}

	g.Do("Type: []string{\"object\"},\n", nil)
	g.Do("AdditionalProperties: &spec.SchemaOrBool{\nAllows: true,\nSchema: &spec.Schema{\nSchemaProps: spec.SchemaProps{\n", nil)
	if err := g.generateDefault(t.Elem.CommentLines, t.Elem, false, t.Elem); err != nil {
		return err
	}
	typeString, format := openapi.OpenAPITypeFormat(elemType.String())
	if typeString != "" {
		g.generateSimpleProperty(typeString, format)
		if enumType, isEnum := g.enumContext.EnumType(t.Elem); isEnum {
			// original type is an enum, add "Enum: " and the values
			g.Do("Enum: []interface{}{$.$},\n", strings.Join(enumType.ValueStrings(), ", "))
		}
		g.Do("},\n},\n},\n", nil)
		return nil
	}
	switch elemType.Kind {
	case types.Builtin:
		return fmt.Errorf("please add type %v to getOpenAPITypeFormat function", elemType)
	case types.Struct:
		g.generateReferenceProperty(elemType)
	case types.Slice, types.Array:
		if err := g.generateSliceProperty(elemType); err != nil {
			return err
		}
	case types.Map:
		if err := g.generateMapProperty(elemType); err != nil {
			return err
		}
	default:
		return fmt.Errorf("map Element kind %v is not supported in %v", elemType.Kind, t.Name)
	}
	g.Do("},\n},\n},\n", nil)
	return nil
}

func (g openAPITypeWriter) generateSliceProperty(t *types.Type) error {
	elemType := resolveAliasAndPtrType(t.Elem)
	g.Do("Type: []string{\"array\"},\n", nil)
	g.Do("Items: &spec.SchemaOrArray{\nSchema: &spec.Schema{\nSchemaProps: spec.SchemaProps{\n", nil)
	if err := g.generateDefault(t.Elem.CommentLines, t.Elem, false, t.Elem); err != nil {
		return err
	}
	typeString, format := openapi.OpenAPITypeFormat(elemType.String())
	if typeString != "" {
		g.generateSimpleProperty(typeString, format)
		if enumType, isEnum := g.enumContext.EnumType(t.Elem); isEnum {
			// original type is an enum, add "Enum: " and the values
			g.Do("Enum: []interface{}{$.$},\n", strings.Join(enumType.ValueStrings(), ", "))
		}
		g.Do("},\n},\n},\n", nil)
		return nil
	}
	switch elemType.Kind {
	case types.Builtin:
		return fmt.Errorf("please add type %v to getOpenAPITypeFormat function", elemType)
	case types.Struct:
		g.generateReferenceProperty(elemType)
	case types.Slice, types.Array:
		if err := g.generateSliceProperty(elemType); err != nil {
			return err
		}
	case types.Map:
		if err := g.generateMapProperty(elemType); err != nil {
			return err
		}
	default:
		return fmt.Errorf("slice Element kind %v is not supported in %v", elemType.Kind, t)
	}
	g.Do("},\n},\n},\n", nil)
	return nil
}
