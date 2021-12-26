/*
Copyright 2019 The Kubernetes Authors.

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

package generator

import "sort"

type edge struct {
	from string
	to   string
}

func transitiveClosure(in map[string][]string) map[string][]string {
	adj := make(map[edge]bool)
	imports := make(map[string]struct{})
	for from, tos := range in {
		for _, to := range tos {
			adj[edge{from, to}] = true
			imports[to] = struct{}{}
		}
	}

	// Warshal's algorithm
	for k := range in {
		for i := range in {
			if !adj[edge{i, k}] {
				continue
			}
			for j := range imports {
				if adj[edge{i, j}] {
					continue
				}
				if adj[edge{k, j}] {
					adj[edge{i, j}] = true
				}
			}
		}
	}

	out := make(map[string][]string, len(in))
	for i := range in {
		for j := range imports {
			if adj[edge{i, j}] {
				out[i] = append(out[i], j)
			}
		}

		sort.Strings(out[i])
	}

	return out
}
