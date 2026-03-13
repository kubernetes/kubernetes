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
	"errors"
	"math"
	"strings"

	"github.com/blang/semver/v4"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	apiservercel "k8s.io/apiserver/pkg/cel"
)

// Semver provides a CEL function library extension for [semver.Version].
//
// semver
//
// Converts a string to a semantic version or results in an error if the string is not a valid semantic version. Refer
// to semver.org documentation for information on accepted patterns.
// An optional "normalize" argument can be passed to enable normalization. Normalization removes any "v" prefix, adds a
// 0 minor and patch numbers to versions with only major or major.minor components specified, and removes any leading 0s.
//
//	semver(<string>) <Semver>
//	semver(<string>, <bool>) <Semver>
//
// Examples:
//
//	semver('1.0.0') // returns a Semver
//	semver('0.1.0-alpha.1') // returns a Semver
//	semver('200K') // error
//	semver('Three') // error
//	semver('Mi') // error
//	semver('v1.0.0', true) // Applies normalization to remove the leading "v". Returns a Semver of "1.0.0".
//	semver('1.0', true) // Applies normalization to add the missing patch version. Returns a Semver of "1.0.0"
//	semver('01.01.01', true) // Applies normalization to remove leading zeros. Returns a Semver of "1.1.1"
//
// isSemver
//
// Returns true if a string is a valid Semver. isSemver returns true if and
// only if semver does not result in error.
// An optional "normalize" argument can be passed to enable normalization. Normalization removes any "v" prefix, adds a
// 0 minor and patch numbers to versions with only major or major.minor components specified, and removes any leading 0s.
//
//	isSemver( <string>) <bool>
//	isSemver( <string>, <bool>) <bool>
//
// Examples:
//
//	isSemver('1.0.0') // returns true
//	isSemver('hello') // returns false
//	isSemver('v1.0')  // returns false (leading "v" is not allowed unless normalization is enabled)
//	isSemver('v1.0', true) // Applies normalization to remove leading "v". returns true
//	semver('1.0', true) // Applies normalization to add the missing patch version. Returns true
//	semver('01.01.01', true) // Applies normalization to remove leading zeros. Returns true
//
// Conversion to Scalars:
//
//   - major/minor/patch: return the major version number as int64.
//
//     <Semver>.major() <int>
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
//     <Semver>.isLessThan(<semver>) <bool>
//     <Semver>.isGreaterThan(<semver>) <bool>
//     <Semver>.compareTo(<semver>) <int>
//
// Examples:
//
// semver("1.2.3").compareTo(semver("1.2.3")) // returns 0
// semver("1.2.3").compareTo(semver("2.0.0")) // returns -1
// semver("1.2.3").compareTo(semver("0.1.2")) // returns 1
func SemverLib(options ...SemverOption) cel.EnvOption {
	semverLib := &semverLibType{}
	for _, o := range options {
		semverLib = o(semverLib)
	}
	return cel.Lib(semverLib)
}

var semverLib = &semverLibType{version: math.MaxUint32} // include all versions

type semverLibType struct {
	version uint32
}

// StringsOption is a functional interface for configuring the strings library.
type SemverOption func(*semverLibType) *semverLibType

func SemverVersion(version uint32) SemverOption {
	return func(lib *semverLibType) *semverLibType {
		lib.version = version
		return lib
	}
}

func (*semverLibType) LibraryName() string {
	return "kubernetes.Semver"
}

func (*semverLibType) Types() []*cel.Type {
	return []*cel.Type{apiservercel.SemverType}
}

func (lib *semverLibType) declarations() map[string][]cel.FunctionOpt {
	fnOpts := map[string][]cel.FunctionOpt{
		"semver": {
			cel.Overload("string_to_semver", []*cel.Type{cel.StringType}, apiservercel.SemverType, cel.UnaryBinding((stringToSemver))),
		},
		"isSemver": {
			cel.Overload("is_semver_string", []*cel.Type{cel.StringType}, cel.BoolType, cel.UnaryBinding(isSemver)),
		},
		"isGreaterThan": {
			cel.MemberOverload("semver_is_greater_than", []*cel.Type{apiservercel.SemverType, apiservercel.SemverType}, cel.BoolType, cel.BinaryBinding(semverIsGreaterThan)),
		},
		"isLessThan": {
			cel.MemberOverload("semver_is_less_than", []*cel.Type{apiservercel.SemverType, apiservercel.SemverType}, cel.BoolType, cel.BinaryBinding(semverIsLessThan)),
		},
		"compareTo": {
			cel.MemberOverload("semver_compare_to", []*cel.Type{apiservercel.SemverType, apiservercel.SemverType}, cel.IntType, cel.BinaryBinding(semverCompareTo)),
		},
		"major": {
			cel.MemberOverload("semver_major", []*cel.Type{apiservercel.SemverType}, cel.IntType, cel.UnaryBinding(semverMajor)),
		},
		"minor": {
			cel.MemberOverload("semver_minor", []*cel.Type{apiservercel.SemverType}, cel.IntType, cel.UnaryBinding(semverMinor)),
		},
		"patch": {
			cel.MemberOverload("semver_patch", []*cel.Type{apiservercel.SemverType}, cel.IntType, cel.UnaryBinding(semverPatch)),
		},
	}
	if lib.version >= 1 {
		fnOpts["semver"] = append(fnOpts["semver"], cel.Overload("string_bool_to_semver", []*cel.Type{cel.StringType, cel.BoolType}, apiservercel.SemverType, cel.BinaryBinding((stringToSemverNormalize))))
		fnOpts["isSemver"] = append(fnOpts["isSemver"], cel.Overload("is_semver_string_bool", []*cel.Type{cel.StringType, cel.BoolType}, cel.BoolType, cel.BinaryBinding(isSemverNormalize)))
	}
	return fnOpts
}

func (s *semverLibType) CompileOptions() []cel.EnvOption {
	// Defined in this function to avoid an initialization order problem.
	semverLibraryDecls := s.declarations()
	options := make([]cel.EnvOption, 0, len(semverLibraryDecls))
	for name, overloads := range semverLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	return options
}

func (*semverLibType) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func isSemver(arg ref.Val) ref.Val {
	return isSemverNormalize(arg, types.Bool(false))
}
func isSemverNormalize(arg ref.Val, normalizeArg ref.Val) ref.Val {
	str, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	normalize, ok := normalizeArg.Value().(bool)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	// Using semver/v4 here is okay because this function isn't
	// used to validate the Kubernetes API. In the CEL base library
	// we would have to use the regular expression from
	// pkg/apis/resource/structured/namedresources/validation/validation.go.
	var err error
	if normalize {
		_, err = normalizeAndParse(str)
	} else {
		_, err = semver.Parse(str)
	}
	if err != nil {
		return types.Bool(false)
	}

	return types.Bool(true)
}

func stringToSemver(arg ref.Val) ref.Val {
	return stringToSemverNormalize(arg, types.Bool(false))
}
func stringToSemverNormalize(arg ref.Val, normalizeArg ref.Val) ref.Val {
	str, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	normalize, ok := normalizeArg.Value().(bool)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	// Using semver/v4 here is okay because this function isn't
	// used to validate the Kubernetes API. In the CEL base library
	// we would have to use the regular expression from
	// pkg/apis/resource/structured/namedresources/validation/validation.go
	// first before parsing.
	var err error
	var v semver.Version
	if normalize {
		v, err = normalizeAndParse(str)
	} else {
		v, err = semver.Parse(str)
	}
	if err != nil {
		return types.WrapErr(err)
	}

	return apiservercel.Semver{Version: v}
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

// normalizeAndParse removes any "v" prefix,  adds a 0 minor and patch numbers to versions with
// only major or major.minor components specified, and removes any leading 0s.
// normalizeAndParse is based on semver.ParseTolerant but does not trim extra whitespace and is
// guaranteed to not change behavior in the future.
func normalizeAndParse(s string) (semver.Version, error) {
	s = strings.TrimPrefix(s, "v")

	// Split into major.minor.(patch+pr+meta)
	parts := strings.SplitN(s, ".", 3)
	// Remove leading zeros.
	for i, p := range parts {
		if len(p) > 1 {
			p = strings.TrimLeft(p, "0")
			if len(p) == 0 || !strings.ContainsAny(p[0:1], "0123456789") {
				p = "0" + p
			}
			parts[i] = p
		}
	}

	// Fill up shortened versions.
	if len(parts) < 3 {
		if strings.ContainsAny(parts[len(parts)-1], "+-") {
			return semver.Version{}, errors.New("short version cannot contain PreRelease/Build meta data")
		}
		for len(parts) < 3 {
			parts = append(parts, "0")
		}
	}
	s = strings.Join(parts, ".")

	return semver.Parse(s)
}
