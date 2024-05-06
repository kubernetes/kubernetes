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
	"math"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// Test provides a test() function that returns true.
func Test(options ...TestOption) cel.EnvOption {
	t := &testLib{version: math.MaxUint32}
	for _, o := range options {
		t = o(t)
	}
	return cel.Lib(t)
}

type testLib struct {
	version uint32
}

func (*testLib) LibraryName() string {
	return "k8s.test"
}

type TestOption func(*testLib) *testLib

func TestVersion(version uint32) func(lib *testLib) *testLib {
	return func(sl *testLib) *testLib {
		sl.version = version
		return sl
	}
}

func (t *testLib) CompileOptions() []cel.EnvOption {
	var options []cel.EnvOption

	if t.version == 0 {
		options = append(options, cel.Function("test",
			cel.Overload("test", []*cel.Type{}, cel.BoolType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					return types.True
				}))))
	}

	if t.version >= 1 {
		options = append(options, cel.Function("test",
			cel.Overload("test", []*cel.Type{}, cel.BoolType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					// Return false here so tests can observe which version of the function is registered
					// Actual function libraries must not break backward compatibility
					return types.False
				}))))
		options = append(options, cel.Function("testV1",
			cel.Overload("testV1", []*cel.Type{}, cel.BoolType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					return types.True
				}))))
	}
	return options
}

func (*testLib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}
