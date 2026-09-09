/*
Copyright 2024 The Kubernetes Authors.

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

package library

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"strings"
)

// JSONPatch provides a CEL function library extension of JSONPatch functions.
//
// jsonpatch.escapeKey
//
// Escapes a string for use as a JSONPatch path key.
//
//	jsonpatch.escapeKey(<string>) <string>
//
// Examples:
//
//	"/metadata/labels/" + jsonpatch.escapeKey('k8s.io/my~label') // returns "/metadata/labels/k8s.io~1my~0label"
func JSONPatch() cel.EnvOption {
	return cel.Lib(jsonPatchLib)
}

var jsonPatchLib = &jsonPatch{}

type jsonPatch struct{}

func (*jsonPatch) LibraryName() string {
	return "kubernetes.jsonpatch"
}

func (*jsonPatch) declarations() map[string][]cel.FunctionOpt {
	return jsonPatchLibraryDecls
}

func (*jsonPatch) Types() []*cel.Type {
	return []*cel.Type{}
}

var jsonPatchLibraryDecls = map[string][]cel.FunctionOpt{
	"jsonpatch.escapeKey": {
		cel.Overload("string_jsonpatch_escapeKey_string", []*cel.Type{cel.StringType}, cel.StringType,
			cel.UnaryBinding(escape)),
	},
}

func (*jsonPatch) CompileOptions() []cel.EnvOption {
	var options []cel.EnvOption
	for name, overloads := range jsonPatchLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	return options
}

func (*jsonPatch) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

var jsonPatchReplacer = strings.NewReplacer("/", "~1", "~", "~0")

func escapeKey(k string) string {
	return jsonPatchReplacer.Replace(k)
}

func escape(arg ref.Val) ref.Val {
	s, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	escaped := escapeKey(s)
	return types.String(escaped)
}
