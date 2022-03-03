/*
Copyright 2022 The Kubernetes Authors.

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
	"regexp"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter/functions"
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Regex provides a CEL function library extension of regex utility functions.
//
// find / findAll
//
// Returns substrings that match the provided regular expression. find returns the first match. findAll may optionally
// be provided a limit. If the limit is set and >= 0, no more than the limit number of matches are returned.
//
//     <string>.find(<string>) <string>
//     <string>.findAll(<string>) <list <string>>
//     <string>.findAll(<string>, <int>) <list <string>>
//
// Examples:
//
//     "abc 123".find('[0-9]*') // returns '123'
//     "abc 123".find('xyz') // returns ''
//     "123 abc 456".findAll('[0-9]*') // returns ['123', '456']
//     "123 abc 456".findAll('[0-9]*', 1) // returns ['123']
//     "123 abc 456".findAll('xyz') // returns []
//
func Regex() cel.EnvOption {
	return cel.Lib(regexLib)
}

var regexLib = &regex{}

type regex struct{}

var regexLibraryDecls = []*exprpb.Decl{

	decls.NewFunction("find",
		decls.NewInstanceOverload("string_find_string",
			[]*exprpb.Type{decls.String, decls.String},
			decls.String),
	),
	decls.NewFunction("findAll",
		decls.NewInstanceOverload("string_find_all_string",
			[]*exprpb.Type{decls.String, decls.String},
			decls.NewListType(decls.String)),
		decls.NewInstanceOverload("string_find_all_string_int",
			[]*exprpb.Type{decls.String, decls.String, decls.Int},
			decls.NewListType(decls.String)),
	),
}

func (*regex) CompileOptions() []cel.EnvOption {
	return []cel.EnvOption{
		cel.Declarations(regexLibraryDecls...),
	}
}

func (*regex) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{
		cel.Functions(
			&functions.Overload{
				Operator: "find",
				Binary:   find,
			},
			&functions.Overload{
				Operator: "string_find_string",
				Binary:   find,
			},
			&functions.Overload{
				Operator: "findAll",
				Binary: func(str, regex ref.Val) ref.Val {
					return findAll(str, regex, types.Int(-1))
				},
				Function: findAll,
			},
			&functions.Overload{
				Operator: "string_find_all_string",
				Binary: func(str, regex ref.Val) ref.Val {
					return findAll(str, regex, types.Int(-1))
				},
			},
			&functions.Overload{
				Operator: "string_find_all_string_int",
				Function: findAll,
			},
		),
	}
}

func find(strVal ref.Val, regexVal ref.Val) ref.Val {
	str, ok := strVal.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(strVal)
	}
	regex, ok := regexVal.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(regexVal)
	}
	re, err := regexp.Compile(regex)
	if err != nil {
		return types.NewErr("Illegal regex: %v", err.Error())
	}
	result := re.FindString(str)
	return types.String(result)
}

func findAll(args ...ref.Val) ref.Val {
	argn := len(args)
	if argn < 2 || argn > 3 {
		return types.NoSuchOverloadErr()
	}
	str, ok := args[0].Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(args[0])
	}
	regex, ok := args[1].Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(args[1])
	}
	n := int64(-1)
	if argn == 3 {
		n, ok = args[2].Value().(int64)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[2])
		}
	}

	re, err := regexp.Compile(regex)
	if err != nil {
		return types.NewErr("Illegal regex: %v", err.Error())
	}

	result := re.FindAllString(str, int(n))

	return types.NewStringList(types.DefaultTypeAdapter, result)
}
