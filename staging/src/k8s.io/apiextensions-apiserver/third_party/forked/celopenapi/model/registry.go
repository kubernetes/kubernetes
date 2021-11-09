// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"fmt"
	"sync"

	"github.com/google/cel-go/cel"
)

// Resolver declares methods to find policy templates and related configuration objects.
type Resolver interface {
	// FindEnv returns an Env object by its fully-qualified name, if present.
	FindEnv(name string) (*Env, bool)

	// FindExprEnv returns a CEL expression environment by its fully-qualified name, if present.
	//
	// Note, the CEL expression environment name corresponds with the model Environment name;
	// however, the expression environment may inherit configuration via the CEL env.Extend method.
	FindExprEnv(name string) (*cel.Env, bool)

	// FindSchema returns an Open API Schema instance by name, if present.
	//
	// Schema names start with a `#` sign as this method is only used to resolve references to
	// relative schema elements within `$ref` schema nodes.
	FindSchema(name string) (*OpenAPISchema, bool)

	// FindTemplate returns a Template by its fully-qualified name, if present.
	FindTemplate(name string) (*Template, bool)

	// FindType returns a DeclType instance corresponding to the given fully-qualified name, if
	// present.
	FindType(name string) (*DeclType, bool)
}

// NewRegistry create a registry for keeping track of environments, schemas, templates, and more
// from a base cel.Env expression environment.
func NewRegistry(stdExprEnv *cel.Env) *Registry {
	return &Registry{
		envs:     map[string]*Env{},
		exprEnvs: map[string]*cel.Env{"": stdExprEnv},
		schemas: map[string]*OpenAPISchema{
			"#anySchema":      AnySchema,
			"#envSchema":      envSchema,
			"#instanceSchema": instanceSchema,
			"#openAPISchema":  schemaDef,
			"#templateSchema": templateSchema,
		},
		templates: map[string]*Template{},
		types: map[string]*DeclType{
			AnyType.TypeName():       AnyType,
			BoolType.TypeName():      BoolType,
			BytesType.TypeName():     BytesType,
			DoubleType.TypeName():    DoubleType,
			DurationType.TypeName():  DurationType,
			IntType.TypeName():       IntType,
			NullType.TypeName():      NullType,
			PlainTextType.TypeName(): PlainTextType,
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
	rwMux     sync.RWMutex
	envs      map[string]*Env
	exprEnvs  map[string]*cel.Env
	schemas   map[string]*OpenAPISchema
	templates map[string]*Template
	types     map[string]*DeclType
}

// FindEnv implements the Resolver interface method.
func (r *Registry) FindEnv(name string) (*Env, bool) {
	r.rwMux.RLock()
	defer r.rwMux.RUnlock()
	env, found := r.envs[name]
	return env, found
}

// FindExprEnv implements the Resolver interface method.
func (r *Registry) FindExprEnv(name string) (*cel.Env, bool) {
	r.rwMux.RLock()
	defer r.rwMux.RUnlock()
	exprEnv, found := r.exprEnvs[name]
	return exprEnv, found
}

// FindSchema implements the Resolver interface method.
func (r *Registry) FindSchema(name string) (*OpenAPISchema, bool) {
	r.rwMux.RLock()
	defer r.rwMux.RUnlock()
	schema, found := r.schemas[name]
	return schema, found
}

// FindTemplate implements the Resolver interface method.
func (r *Registry) FindTemplate(name string) (*Template, bool) {
	r.rwMux.RLock()
	defer r.rwMux.RUnlock()
	tmpl, found := r.templates[name]
	return tmpl, found
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

// SetEnv registers an environment description by fully qualified name.
func (r *Registry) SetEnv(name string, env *Env) error {
	r.rwMux.Lock()
	defer r.rwMux.Unlock()
	// Cleanup environment related artifacts when the env is reset.
	priorEnv, found := r.envs[name]
	if found {
		for typeName := range priorEnv.Types {
			delete(r.types, typeName)
		}
	}
	// Configure the new environment.
	baseExprEnv, found := r.exprEnvs[""]
	if !found {
		return fmt.Errorf("missing default expression environment")
	}
	exprEnv, err := baseExprEnv.Extend(env.ExprEnvOptions()...)
	if err != nil {
		return err
	}
	r.exprEnvs[name] = exprEnv
	r.envs[name] = env
	for typeName, typ := range env.Types {
		r.types[typeName] = typ
	}
	return nil
}

// SetSchema registers an OpenAPISchema fragment by its relative name so that it may be referenced
// as a reusable schema unit within other OpenAPISchema instances.
//
// Name format: '#<simpleName>'.
func (r *Registry) SetSchema(name string, schema *OpenAPISchema) error {
	r.rwMux.Lock()
	defer r.rwMux.Unlock()
	r.schemas[name] = schema
	return nil
}

// SetTemplate registers a template by its fully qualified name.
func (r *Registry) SetTemplate(name string, tmpl *Template) error {
	r.rwMux.Lock()
	defer r.rwMux.Unlock()
	r.templates[name] = tmpl
	return nil
}

// SetType registers a DeclType descriptor by its fully qualified name.
func (r *Registry) SetType(name string, declType *DeclType) error {
	r.rwMux.Lock()
	defer r.rwMux.Unlock()
	r.types[name] = declType
	return nil
}
