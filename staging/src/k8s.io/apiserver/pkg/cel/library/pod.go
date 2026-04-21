/*
Copyright 2025 The Kubernetes Authors.

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
	"github.com/google/cel-go/common/types/traits"
)

// Pod provides a CEL function library extension with pod-related helper functions.
//
// allContainers
//
// Returns a single list containing all containers from a pod spec: initContainers,
// containers, and ephemeralContainers (in that order). Absent and null fields are
// treated as empty. A field that is present but not a list causes a runtime error.
//
//	allContainers(<podSpec>) <list>
//
// Examples:
//
//	// enforce image prefix across every container type
//	allContainers(object.spec).all(c, c.image.startsWith('myregistry.io/'))
//
//	// works for pod template specs too
//	allContainers(object.spec.template.spec).exists(c, c.name == 'sidecar')
func Pod() cel.EnvOption {
	return cel.Lib(podLib)
}

var podLib = &pod{}

type pod struct{}

func (*pod) LibraryName() string {
	return "kubernetes.pod"
}

func (*pod) Types() []*cel.Type {
	return []*cel.Type{}
}

func (*pod) declarations() map[string][]cel.FunctionOpt {
	return podLibraryDecls
}

var podLibraryDecls = map[string][]cel.FunctionOpt{
	"allContainers": {
		cel.Overload("pod_spec_all_containers",
			[]*cel.Type{cel.DynType},
			cel.ListType(cel.DynType),
			cel.UnaryBinding(allContainersImpl),
		),
	},
}

func (*pod) CompileOptions() []cel.EnvOption {
	options := []cel.EnvOption{}
	for name, overloads := range podLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	return options
}

func (*pod) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

// allContainersImpl concatenates initContainers, containers, and ephemeralContainers
// from a pod spec into a single list. Absent or null fields are treated as empty; a
// field that is present but not a list returns a runtime error.
func allContainersImpl(spec ref.Val) ref.Val {
	indexer, ok := spec.(traits.Indexer)
	if !ok {
		return types.MaybeNoSuchOverloadErr(spec)
	}

	var items []ref.Val
	for _, field := range []string{"initContainers", "containers", "ephemeralContainers"} {
		val := indexer.Get(types.String(field))
		if types.IsError(val) {
			// field absent; skip
			continue
		}
		if val == types.NullValue {
			continue
		}
		lister, ok := val.(traits.Lister)
		if !ok {
			return types.NewErr("allContainers: %s is present but is not a list (got %s)", field, val.Type().TypeName())
		}
		for it := lister.Iterator(); it.HasNext() == types.True; {
			items = append(items, it.Next())
		}
	}
	return types.NewRefValList(types.DefaultTypeAdapter, items)
}
