// Copyright 2020 Google LLC
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

package cel

import (
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/interpreter/functions"
	"github.com/google/cel-go/parser"
)

// Library provides a collection of EnvOption and ProgramOption values used to confiugre a CEL
// environment for a particular use case or with a related set of functionality.
//
// Note, the ProgramOption values provided by a library are expected to be static and not vary
// between calls to Env.Program(). If there is a need for such dynamic configuration, prefer to
// configure these options outside the Library and within the Env.Program() call directly.
type Library interface {
	// CompileOptions returns a collection of funcitional options for configuring the Parse / Check
	// environment.
	CompileOptions() []EnvOption

	// ProgramOptions returns a collection of functional options which should be included in every
	// Program generated from the Env.Program() call.
	ProgramOptions() []ProgramOption
}

// Lib creates an EnvOption out of a Library, allowing libraries to be provided as functional args,
// and to be linked to each other.
func Lib(l Library) EnvOption {
	return func(e *Env) (*Env, error) {
		var err error
		for _, opt := range l.CompileOptions() {
			e, err = opt(e)
			if err != nil {
				return nil, err
			}
		}
		e.progOpts = append(e.progOpts, l.ProgramOptions()...)
		return e, nil
	}
}

// StdLib returns an EnvOption for the standard library of CEL functions and macros.
func StdLib() EnvOption {
	return Lib(stdLibrary{})
}

// stdLibrary implements the Library interface and provides functional options for the core CEL
// features documented in the specification.
type stdLibrary struct{}

// EnvOptions returns options for the standard CEL function declarations and macros.
func (stdLibrary) CompileOptions() []EnvOption {
	return []EnvOption{
		Declarations(checker.StandardDeclarations()...),
		Macros(parser.AllMacros...),
	}
}

// ProgramOptions returns function implementations for the standard CEL functions.
func (stdLibrary) ProgramOptions() []ProgramOption {
	return []ProgramOption{
		Functions(functions.StandardOverloads()...),
	}
}
