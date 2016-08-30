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

package types

// FlattenMembers recursively takes any embedded members and puts them in the
// top level, correctly hiding them if the top level hides them. There must not
// be a cycle-- that implies infinite members.
//
// This is useful for e.g. computing all the valid keys in a json struct,
// properly considering any configuration of embedded structs.
func FlattenMembers(m []Member) []Member {
	embedded := []Member{}
	normal := []Member{}
	type nameInfo struct {
		top bool
		i   int
	}
	names := map[string]nameInfo{}
	for i := range m {
		if m[i].Embedded && m[i].Type.Kind == Struct {
			embedded = append(embedded, m[i])
		} else {
			normal = append(normal, m[i])
			names[m[i].Name] = nameInfo{true, len(normal) - 1}
		}
	}
	for i := range embedded {
		for _, e := range FlattenMembers(embedded[i].Type.Members) {
			if info, found := names[e.Name]; found {
				if info.top {
					continue
				}
				if n := normal[info.i]; n.Name == e.Name && n.Type == e.Type {
					continue
				}
				panic("conflicting members")
			}
			normal = append(normal, e)
			names[e.Name] = nameInfo{false, len(normal) - 1}
		}
	}
	return normal
}
