// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package cel defines the top-level interface for the Common Expression Language (CEL).
//
// CEL is a non-Turing complete expression language designed to parse, check, and evaluate
// expressions against user-defined environments.
package cel

// Compile is a convenience function that constructs a new Env using the provided EnvOption values,
// compiles the expression string, and plans an executable Program.
//
// Warning: Creating a new environment for every compilation is expensive. Environment setup should be done once
// and shared across expression compilations when the options remain the same.
func Compile(expression string, opts ...EnvOption) (Program, error) {
	env, err := NewEnv(opts...)
	if err != nil {
		return nil, err
	}
	ast, iss := env.Compile(expression)
	if iss.Err() != nil {
		return nil, iss.Err()
	}
	prg, err := env.Program(ast, EvalOptions(OptOptimize))
	if err != nil {
		return nil, err
	}
	return prg, nil
}
