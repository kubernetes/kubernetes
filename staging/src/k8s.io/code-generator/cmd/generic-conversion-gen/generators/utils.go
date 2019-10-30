/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"text/template"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	"k8s.io/klog"
)

// ConversionPair represents a conversion pair from inType to outType
type ConversionPair struct {
	InType  *types.Type
	OutType *types.Type
}

// A NamedVariable represents a named variable to be rendered in snippets.
type NamedVariable struct {
	Name string
	Type *types.Type
}

// NewNamedVariable builds a new NamedVariable.
func NewNamedVariable(name string, t *types.Type) NamedVariable {
	return NamedVariable{
		Name: name,
		Type: t,
	}
}

const (
	conversionFunctionPrefix = "Convert_"
	snippetDelimiter         = "$"
)

func conversionFunctionNameTemplate(namer string) string {
	return fmt.Sprintf("%s%s.inType|%s%s_To_%s.outType|%s%s",
		conversionFunctionPrefix, snippetDelimiter, namer, snippetDelimiter, snippetDelimiter, namer, snippetDelimiter)
}

func argsFromType(inType, outType *types.Type) generator.Args {
	return generator.Args{
		"inType":  inType,
		"outType": outType,
	}
}

// ConversionNamer returns a namer for conversion function names.
// It is a good namer to use as a default namer when using conversion generators,
// so as to limit the number of changes the generator makes.
func ConversionNamer() *namer.NameStrategy {
	return &namer.NameStrategy{
		Join: func(pre string, in []string, post string) string {
			return strings.Join(in, "_")
		},
		PrependPackageNames: 1,
	}
}

// unwrapAlias recurses down aliased types to find the bedrock type.
func unwrapAlias(in *types.Type) *types.Type {
	for in.Kind == types.Alias {
		in = in.Underlying
	}
	return in
}

func functionHasTag(function *types.Type, functionTagName, tagValue string) bool {
	if functionTagName == "" {
		return false
	}
	values := types.ExtractCommentTags("+", function.CommentLines)[functionTagName]
	return len(values) == 1 && values[0] == tagValue
}

func isCopyOnlyFunction(function *types.Type, functionTagName string) bool {
	return functionHasTag(function, functionTagName, "copy-only")
}
func conversionFunctionName(in, out *types.Type, conversionNamer *namer.NameStrategy, buffer *bytes.Buffer) string {
	namerName := "conversion"
	tmpl, err := template.New(fmt.Sprintf("conversion function name from %s to %s", in.Name, out.Name)).
		Delims(snippetDelimiter, snippetDelimiter).
		Funcs(map[string]interface{}{namerName: conversionNamer.Name}).
		Parse(conversionFunctionNameTemplate(namerName))
	if err != nil {
		// this really shouldn't error out
		klog.Fatalf("error when generating conversion function name: %v", err)
	}
	buffer.Reset()
	err = tmpl.Execute(buffer, argsFromType(in, out))
	if err != nil {
		// this really shouldn't error out
		klog.Fatalf("error when generating conversion function name: %v", err)
	}
	return buffer.String()
}
