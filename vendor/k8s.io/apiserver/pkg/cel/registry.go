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

package cel

import (
	"sync"

	"github.com/google/cel-go/cel"
)

// Resolver declares methods to find policy templates and related configuration objects.
type Resolver interface {
	// FindType returns a DeclType instance corresponding to the given fully-qualified name, if
	// present.
	FindType(name string) (*DeclType, bool)
}

// NewRegistry create a registry for keeping track of environments and types
// from a base cel.Env expression environment.
func NewRegistry(stdExprEnv *cel.Env) *Registry {
	return &Registry{
		exprEnvs: map[string]*cel.Env{"": stdExprEnv},
		types: map[string]*DeclType{
			BoolType.TypeName():      BoolType,
			BytesType.TypeName():     BytesType,
			DoubleType.TypeName():    DoubleType,
			DurationType.TypeName():  DurationType,
			IntType.TypeName():       IntType,
			NullType.TypeName():      NullType,
			StringType.TypeName():    StringType,
			TimestampType.TypeName(): TimestampType,
			UintType.TypeName():      UintType,
			ListType.TypeName():      ListType,
			MapType.TypeName():       MapType,
		},
	}
}

// Registry defines a repository of environment, schema, template, and type definitions.
//
// Registry instances are concurrency-safe.
type Registry struct {
	rwMux    sync.RWMutex
	exprEnvs map[string]*cel.Env
	types    map[string]*DeclType
}

// FindType implements the Resolver interface method.
func (r *Registry) FindType(name string) (*DeclType, bool) {
	r.rwMux.RLock()
	defer r.rwMux.RUnlock()
	typ, found := r.types[name]
	if found {
		return typ, true
	}
	return typ, found
}

// SetType registers a DeclType descriptor by its fully qualified name.
func (r *Registry) SetType(name string, declType *DeclType) error {
	r.rwMux.Lock()
	defer r.rwMux.Unlock()
	r.types[name] = declType
	return nil
}
