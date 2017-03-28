/*
Copyright 2016 The Kubernetes Authors.

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

package apiregistration

import (
	"sort"
	"strings"
)

func SortedByGroup(servers []*APIService) [][]*APIService {
	serversByPriority := ByPriority(servers)
	sort.Sort(serversByPriority)

	ret := [][]*APIService{}
	for _, curr := range serversByPriority {
		// check to see if we already have an entry for this group
		existingIndex := -1
		for j, groupInReturn := range ret {
			if groupInReturn[0].Spec.Group == curr.Spec.Group {
				existingIndex = j
				break
			}
		}

		if existingIndex >= 0 {
			ret[existingIndex] = append(ret[existingIndex], curr)
			continue
		}

		ret = append(ret, []*APIService{curr})
	}

	return ret
}

type ByPriority []*APIService

func (s ByPriority) Len() int      { return len(s) }
func (s ByPriority) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s ByPriority) Less(i, j int) bool {
	if s[i].Spec.Priority == s[j].Spec.Priority {
		return strings.Compare(s[i].Name, s[j].Name) < 0
	}
	return s[i].Spec.Priority < s[j].Spec.Priority
}
