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
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
)

// Regex provides a CEL function library extension of regex utility functions.
//
// find / findAll
//
// Returns substrings that match the provided regular expression. find returns the first match. findAll may optionally
// be provided a limit. If the limit is set and >= 0, no more than the limit number of matches are returned.
//
//	<string>.find(<string>) <string>
//	<string>.findAll(<string>) <list <string>>
//	<string>.findAll(<string>, <int>) <list <string>>
//
// Examples:
//
//	"abc 123".find('[0-9]*') // returns '123'
//	"abc 123".find('xyz') // returns ''
//	"123 abc 456".findAll('[0-9]*') // returns ['123', '456']
//	"123 abc 456".findAll('[0-9]*', 1) // returns ['123']
//	"123 abc 456".findAll('xyz') // returns []
func Regex() cel.EnvOption {
	return cel.Lib(regexLib)
}

var regexLib = &regex{}

type regex struct{}

var regexLibraryDecls = map[string][]cel.FunctionOpt{
	"find": {
		cel.MemberOverload("string_find_string", []*cel.Type{cel.StringType, cel.StringType}, cel.StringType,
			cel.BinaryBinding(find))},
	"findAll": {
		cel.MemberOverload("string_find_all_string", []*cel.Type{cel.StringType, cel.StringType},
			cel.ListType(cel.StringType),
			cel.BinaryBinding(func(str, regex ref.Val) ref.Val {
				return findAll(str, regex, types.Int(-1))
			})),
		cel.MemberOverload("string_find_all_string_int",
			[]*cel.Type{cel.StringType, cel.StringType, cel.IntType},
			cel.ListType(cel.StringType),
			cel.FunctionBinding(findAll)),
	},
}

func (*regex) CompileOptions() []cel.EnvOption {
	options := []cel.EnvOption{}
	for name, overloads := range regexLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	return options
}

func (*regex) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{
		cel.OptimizeRegex(FindRegexOptimization, FindAllRegexOptimization),
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

// FindRegexOptimization optimizes the 'find' function by compiling the regex pattern and
// reporting any compilation errors at program creation time, and using the compiled regex pattern for all function
// call invocations.
var FindRegexOptimization = &interpreter.RegexOptimization{
	Function:   "find",
	RegexIndex: 1,
	Factory: func(call interpreter.InterpretableCall, regexPattern string) (interpreter.InterpretableCall, error) {
		compiledRegex, err := regexp.Compile(regexPattern)
		if err != nil {
			return nil, err
		}
		return interpreter.NewCall(call.ID(), call.Function(), call.OverloadID(), call.Args(), func(args ...ref.Val) ref.Val {
			if len(args) != 2 {
				return types.NoSuchOverloadErr()
			}
			in, ok := args[0].Value().(string)
			if !ok {
				return types.MaybeNoSuchOverloadErr(args[0])
			}
			return types.String(compiledRegex.FindString(in))
		}), nil
	},
}

// FindAllRegexOptimization optimizes the 'findAll' function by compiling the regex pattern and
// reporting any compilation errors at program creation time, and using the compiled regex pattern for all function
// call invocations.
var FindAllRegexOptimization = &interpreter.RegexOptimization{
	Function:   "findAll",
	RegexIndex: 1,
	Factory: func(call interpreter.InterpretableCall, regexPattern string) (interpreter.InterpretableCall, error) {
		compiledRegex, err := regexp.Compile(regexPattern)
		if err != nil {
			return nil, err
		}
		return interpreter.NewCall(call.ID(), call.Function(), call.OverloadID(), call.Args(), func(args ...ref.Val) ref.Val {
			argn := len(args)
			if argn < 2 || argn > 3 {
				return types.NoSuchOverloadErr()
			}
			str, ok := args[0].Value().(string)
			if !ok {
				return types.MaybeNoSuchOverloadErr(args[0])
			}
			n := int64(-1)
			if argn == 3 {
				n, ok = args[2].Value().(int64)
				if !ok {
					return types.MaybeNoSuchOverloadErr(args[2])
				}
			}

			result := compiledRegex.FindAllString(str, int(n))
			return types.NewStringList(types.DefaultTypeAdapter, result)
		}), nil
	},
}
