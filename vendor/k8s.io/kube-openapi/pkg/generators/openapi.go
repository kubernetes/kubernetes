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
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	openapi "k8s.io/kube-openapi/pkg/common"

	"k8s.io/klog/v2"
)

// This is the comment tag that carries parameters for open API generation.
const tagName = "k8s:openapi-gen"
const tagOptional = "optional"
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
	return types.ExtractCommentTags("+", comments)[tagName]
}

func getSingleTagsValue(comments []string, tag string) (string, error) {
	tags, ok := types.ExtractCommentTags("+", comments)[tag]
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

// hasOptionalTag returns true if the member has +optional in its comments or
// omitempty in its json tags.
func hasOptionalTag(m *types.Member) bool {
	hasOptionalCommentTag := types.ExtractCommentTags(
		"+", m.CommentLines)[tagOptional] != nil
	hasOptionalJsonTag := strings.Contains(
		reflect.StructTag(m.Tags).Get("json"), "omitempty")
	return hasOptionalCommentTag || hasOptionalJsonTag
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
	generator.DefaultGen
	// TargetPackage is the package that will get GetOpenAPIDefinitions function returns all open API definitions.
	targetPackage string
	imports       namer.ImportTracker
}

func newOpenAPIGen(sanitizedName string, targetPackage string) generator.Generator {
	return &openAPIGen{
		DefaultGen: generator.DefaultGen{
			OptionalName: sanitizedName,
		},
		imports:       generator.NewImportTracker(),
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

func (g *openAPIGen) isOtherPackage(pkg string) bool {
	if pkg == g.targetPackage {
		return false
	}
	if strings.HasSuffix(pkg, "\""+g.targetPackage+"\"") {
		return false
	}
	return true
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
	GetDefinitionInterface *types.Type
}

func newOpenAPITypeWriter(sw *generator.SnippetWriter, c *generator.Context) openAPITypeWriter {
	return openAPITypeWriter{
		SnippetWriter: sw,
		context:       c,
		refTypes:      map[string]*types.Type{},
	}
}

func methodReturnsValue(mt *types.Type, pkg, name string) bool {
	if len(mt.Signature.Parameters) != 0 || len(mt.Signature.Results) != 1 {
		return false
	}
	r := mt.Signature.Results[0]
	return r.Name.Name == name && r.Name.Package == pkg
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

// typeShortName returns short package name (e.g. the name x appears in package x definition) dot type name.
func typeShortName(t *types.Type) string {
	return filepath.Base(t.Name.Package) + "." + t.Name.Name
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
		if !hasOptionalTag(&m) {
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

func (g openAPITypeWriter) generate(t *types.Type) error {
	// Only generate for struct type and ignore the rest
	switch t.Kind {
	case types.Struct:
		hasV2Definition := hasOpenAPIDefinitionMethod(t)
		hasV2DefinitionTypeAndFormat := hasOpenAPIDefinitionMethods(t)
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
				"Format:$.type|raw${}.OpenAPISchemaFormat(),\n"+
				"},\n"+
				"},\n"+
				"})\n}\n\n", args)
			return nil
		case hasV2DefinitionTypeAndFormat:
			g.Do("return $.OpenAPIDefinition|raw${\n"+
				"Schema: spec.Schema{\n"+
				"SchemaProps: spec.SchemaProps{\n", args)
			g.generateDescription(t.CommentLines)
			g.Do("Type:$.type|raw${}.OpenAPISchemaType(),\n"+
				"Format:$.type|raw${}.OpenAPISchemaFormat(),\n"+
				"},\n"+
				"},\n"+
				"}\n}\n\n", args)
			return nil
		}
		g.Do("return $.OpenAPIDefinition|raw${\nSchema: spec.Schema{\nSchemaProps: spec.SchemaProps{\n", args)
		g.generateDescription(t.CommentLines)
		g.Do("Type: []string{\"object\"},\n", nil)

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
		if err := g.generateStructExtensions(t); err != nil {
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

func (g openAPITypeWriter) generateStructExtensions(t *types.Type) error {
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
	g.emitExtensions(extensions, unions)
	return nil
}

func (g openAPITypeWriter) generateMemberExtensions(m *types.Member, parent *types.Type) error {
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
	g.emitExtensions(extensions, nil)
	return nil
}

func (g openAPITypeWriter) emitExtensions(extensions []extension, unions []union) {
	// If any extensions exist, then emit code to create them.
	if len(extensions) == 0 && len(unions) == 0 {
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

func defaultFromComments(comments []string) (interface{}, error) {
	tag, err := getSingleTagsValue(comments, tagDefault)
	if tag == "" {
		return nil, err
	}
	var i interface{}
	if err := json.Unmarshal([]byte(tag), &i); err != nil {
		return nil, fmt.Errorf("failed to unmarshal default: %v", err)
	}
	return i, nil
}

func mustEnforceDefault(t *types.Type, omitEmpty bool) (interface{}, error) {
	switch t.Kind {
	case types.Pointer, types.Map, types.Slice, types.Array, types.Interface:
		return nil, nil
	case types.Struct:
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

func (g openAPITypeWriter) generateDefault(comments []string, t *types.Type, omitEmpty bool) error {
	t = resolveAliasAndEmbeddedType(t)
	def, err := defaultFromComments(comments)
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

	postDoc := strings.TrimRight(buffer.String(), "\n")
	postDoc = strings.Replace(postDoc, "\\\"", "\"", -1) // replace user's \" to "
	postDoc = strings.Replace(postDoc, "\"", "\\\"", -1) // Escape "
	postDoc = strings.Replace(postDoc, "\n", "\\n", -1)
	postDoc = strings.Replace(postDoc, "\t", "\\t", -1)
	postDoc = strings.Trim(postDoc, " ")
	if postDoc != "" {
		g.Do("Description: \"$.$\",\n", postDoc)
	}
}

func (g openAPITypeWriter) generateProperty(m *types.Member, parent *types.Type) error {
	name := getReferableName(m)
	if name == "" {
		return nil
	}
	if err := g.validatePatchTags(m, parent); err != nil {
		return err
	}
	g.Do("\"$.$\": {\n", name)
	if err := g.generateMemberExtensions(m, parent); err != nil {
		return err
	}
	g.Do("SchemaProps: spec.SchemaProps{\n", nil)
	g.generateDescription(m.CommentLines)
	jsonTags := getJsonTags(m)
	if len(jsonTags) > 1 && jsonTags[1] == "string" {
		g.generateSimpleProperty("string", "")
		g.Do("},\n},\n", nil)
		return nil
	}
	omitEmpty := strings.Contains(reflect.StructTag(m.Tags).Get("json"), "omitempty")
	if err := g.generateDefault(m.CommentLines, m.Type, omitEmpty); err != nil {
		return fmt.Errorf("failed to generate default in %v: %v: %v", parent, m.Name, err)
	}
	t := resolveAliasAndPtrType(m.Type)
	// If we can get a openAPI type and format for this type, we consider it to be simple property
	typeString, format := openapi.OpenAPITypeFormat(t.String())
	if typeString != "" {
		g.generateSimpleProperty(typeString, format)
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

func resolveAliasAndEmbeddedType(t *types.Type) *types.Type {
	var prev *types.Type
	for prev != t {
		prev = t
		if t.Kind == types.Alias {
			t = t.Underlying
		}
		if t.Kind == types.Struct {
			if len(t.Members) == 1 && t.Members[0].Embedded {
				t = t.Members[0].Type
			}
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
	if err := g.generateDefault(t.Elem.CommentLines, t.Elem, false); err != nil {
		return err
	}
	typeString, format := openapi.OpenAPITypeFormat(elemType.String())
	if typeString != "" {
		g.generateSimpleProperty(typeString, format)
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
	if err := g.generateDefault(t.Elem.CommentLines, t.Elem, false); err != nil {
		return err
	}
	typeString, format := openapi.OpenAPITypeFormat(elemType.String())
	if typeString != "" {
		g.generateSimpleProperty(typeString, format)
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
