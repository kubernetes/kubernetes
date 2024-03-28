/*
Copyright 2023 The Kubernetes Authors.

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
	"github.com/blang/semver/v4"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	apiservercel "k8s.io/apiserver/pkg/cel"
)

// SemVer provides a CEL function library extension for [semver.Version].
//
// semver
//
// Converts a string to a SemVer or results in an error if the string is not a valid SemVer. Refer
// to semver.SemVer documentation for information on accepted patterns.
//
//	semver(<string>) <SemVer>
//
// Examples:
//
//	semver('1.0.0') // returns a SemVer
//	semver('0.1.0-alpha.1') // returns a SemVer
//	semver('200K') // error
//	semver('Three') // error
//	semver('Mi') // error
//
// isSemVer
//
// Returns true if a string is a valid SemVer. isSemVer returns true if and
// only if semver does not result in error.
//
//	isSemVer( <string>) <bool>
//
// Examples:
//
//	isSemVer('1.0.0') // returns true
//	isSemVer('v1.0') // returns true (tolerant parsing)
//	isSemVer('hello') // returns false
//
// Conversion to Scalars:
//
//   - major/minor/patch: return the major version number as int64.
//
//     <SemVer>.major() <int>
//
// Examples:
//
// semver("1.2.3").major() // returns 1
//
// Comparisons
//
//   - isGreaterThan: Returns true if and only if the receiver is greater than the operand
//
//   - isLessThan: Returns true if and only if the receiver is less than the operand
//
//   - compareTo: Compares receiver to operand and returns 0 if they are equal, 1 if the receiver is greater, or -1 if the receiver is less than the operand
//
//
//     <SemVer>.isLessThan(<semver>) <bool>
//     <SemVer>.isGreaterThan(<semver>) <bool>
//     <SemVer>.compareTo(<semver>) <int>
//
// Examples:
//
// semver("1.2.3").compareTo(semver("1.2.3")) // returns 0
// semver("1.2.3").compareTo(semver("2.0.0")) // returns -1
// semver("1.2.3").compareTo(semver("0.1.2")) // returns 1

func SemVer() cel.EnvOption {
	return cel.Lib(semverLib)
}

var semverLib = &semverLibType{}

type semverLibType struct{}

func (*semverLibType) LibraryName() string {
	return "k8s.semver"
}

var semverLibraryDecls = map[string][]cel.FunctionOpt{
	"semver": {
		cel.Overload("string_to_semver", []*cel.Type{cel.StringType}, apiservercel.SemVerType, cel.UnaryBinding((stringToSemVer))),
	},
	"isSemVer": {
		cel.Overload("is_semver_string", []*cel.Type{cel.StringType}, cel.BoolType, cel.UnaryBinding(isSemVer)),
	},
	"isGreaterThan": {
		cel.MemberOverload("semver_is_greater_than", []*cel.Type{apiservercel.SemVerType, apiservercel.SemVerType}, cel.BoolType, cel.BinaryBinding(semverIsGreaterThan)),
	},
	"isLessThan": {
		cel.MemberOverload("semver_is_less_than", []*cel.Type{apiservercel.SemVerType, apiservercel.SemVerType}, cel.BoolType, cel.BinaryBinding(semverIsLessThan)),
	},
	"compareTo": {
		cel.MemberOverload("semver_compare_to", []*cel.Type{apiservercel.SemVerType, apiservercel.SemVerType}, cel.IntType, cel.BinaryBinding(semverCompareTo)),
	},
	"major": {
		cel.MemberOverload("semver_major", []*cel.Type{apiservercel.SemVerType}, cel.IntType, cel.UnaryBinding(semverMajor)),
	},
	"minor": {
		cel.MemberOverload("semver_minor", []*cel.Type{apiservercel.SemVerType}, cel.IntType, cel.UnaryBinding(semverMinor)),
	},
	"patch": {
		cel.MemberOverload("semver_patch", []*cel.Type{apiservercel.SemVerType}, cel.IntType, cel.UnaryBinding(semverPatch)),
	},
}

func (*semverLibType) CompileOptions() []cel.EnvOption {
	options := make([]cel.EnvOption, 0, len(semverLibraryDecls))
	for name, overloads := range semverLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	return options
}

func (*semverLibType) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func isSemVer(arg ref.Val) ref.Val {
	str, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	_, err := semver.Parse(str)
	if err != nil {
		return types.Bool(false)
	}

	return types.Bool(true)
}

func stringToSemVer(arg ref.Val) ref.Val {
	str, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	v, err := semver.Parse(str)
	if err != nil {
		return types.WrapErr(err)
	}

	return apiservercel.SemVer{Version: v}
}

func semverMajor(arg ref.Val) ref.Val {
	v, ok := arg.Value().(semver.Version)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.Int(v.Major)
}

func semverMinor(arg ref.Val) ref.Val {
	v, ok := arg.Value().(semver.Version)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.Int(v.Minor)
}

func semverPatch(arg ref.Val) ref.Val {
	v, ok := arg.Value().(semver.Version)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.Int(v.Patch)
}

func semverIsGreaterThan(arg ref.Val, other ref.Val) ref.Val {
	v, ok := arg.Value().(semver.Version)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	v2, ok := other.Value().(semver.Version)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	return types.Bool(v.Compare(v2) == 1)
}

func semverIsLessThan(arg ref.Val, other ref.Val) ref.Val {
	v, ok := arg.Value().(semver.Version)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	v2, ok := other.Value().(semver.Version)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	return types.Bool(v.Compare(v2) == -1)
}

func semverCompareTo(arg ref.Val, other ref.Val) ref.Val {
	v, ok := arg.Value().(semver.Version)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	v2, ok := other.Value().(semver.Version)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	return types.Int(v.Compare(v2))
}
