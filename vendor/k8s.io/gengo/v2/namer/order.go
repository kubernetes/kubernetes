/*
Copyright 2015 The Kubernetes Authors.

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

package namer

import (
	"sort"

	"k8s.io/gengo/v2/types"
)

// Orderer produces an ordering of types given a Namer.
type Orderer struct {
	Namer
}

// OrderUniverse assigns a name to every type in the Universe, including Types,
// Functions and Variables, and returns a list sorted by those names.
func (o *Orderer) OrderUniverse(u types.Universe) []*types.Type {
	list := tList{
		namer: o.Namer,
	}
	for _, p := range u {
		for _, t := range p.Types {
			list.types = append(list.types, t)
		}
		for _, f := range p.Functions {
			list.types = append(list.types, f)
		}
		for _, v := range p.Variables {
			list.types = append(list.types, v)
		}
		for _, v := range p.Constants {
			list.types = append(list.types, v)
		}
	}
	sort.Sort(list)
	return list.types
}

// OrderTypes assigns a name to every type, and returns a list sorted by those
// names.
func (o *Orderer) OrderTypes(typeList []*types.Type) []*types.Type {
	list := tList{
		namer: o.Namer,
		types: typeList,
	}
	sort.Sort(list)
	return list.types
}

type tList struct {
	namer Namer
	types []*types.Type
}

func (t tList) Len() int           { return len(t.types) }
func (t tList) Less(i, j int) bool { return t.namer.Name(t.types[i]) < t.namer.Name(t.types[j]) }
func (t tList) Swap(i, j int)      { t.types[i], t.types[j] = t.types[j], t.types[i] }
