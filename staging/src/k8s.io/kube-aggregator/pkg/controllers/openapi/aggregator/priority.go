/*
Copyright 2017 The Kubernetes Authors.

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

package aggregator

import (
	"sort"
)

// byPriority can be used in sort.Sort to sort specs with their priorities.
// TODO: rework this to not only look at the first apiService. This is just temporary until we split specs by APIServices.
type byPriority struct {
	specs           []specInfo
	groupPriorities map[string]int32
}

func (a byPriority) Len() int      { return len(a.specs) }
func (a byPriority) Swap(i, j int) { a.specs[i], a.specs[j] = a.specs[j], a.specs[i] }
func (a byPriority) Less(i, j int) bool {
	// All local specs will come first
	if a.specs[i].apiServices[0].Spec.Service == nil && a.specs[j].apiServices[0].Spec.Service != nil {
		return true
	}
	if a.specs[i].apiServices[0].Spec.Service != nil && a.specs[j].apiServices[0].Spec.Service == nil {
		return false
	}
	// WARNING: This will result in not following priorities for local APIServices.
	if a.specs[i].apiServices[0].Spec.Service == nil {
		// Sort local specs with their name. This is the order in the delegation chain (aggregator first).
		return a.specs[i].apiServices[0].Name < a.specs[j].apiServices[0].Name
	}
	var iPriority, jPriority int32
	if a.specs[i].apiServices[0].Spec.Group == a.specs[j].apiServices[0].Spec.Group {
		iPriority = a.specs[i].apiServices[0].Spec.VersionPriority
		jPriority = a.specs[i].apiServices[0].Spec.VersionPriority
	} else {
		iPriority = a.groupPriorities[a.specs[i].apiServices[0].Spec.Group]
		jPriority = a.groupPriorities[a.specs[j].apiServices[0].Spec.Group]
	}
	if iPriority != jPriority {
		// Sort by priority, higher first
		return iPriority > jPriority
	}
	// Sort by service name.
	return a.specs[i].apiServices[0].Name < a.specs[j].apiServices[0].Name
}

func sortByPriority(specs []specInfo) {
	b := byPriority{
		specs:           specs,
		groupPriorities: map[string]int32{},
	}
	for _, spec := range specs {
		// TODO: split spec and merge in the right order
		// Note: using [0] here is not worse than before when we were merging n-times, but overriding old definitions
		//       which did not belong to the APIService at hand (specs can be for more than one APIService!)
		if spec.apiServices[0].Spec.Service == nil {
			continue
		}
		if pr, found := b.groupPriorities[spec.apiServices[0].Spec.Group]; !found || spec.apiServices[0].Spec.GroupPriorityMinimum > pr {
			b.groupPriorities[spec.apiServices[0].Spec.Group] = spec.apiServices[0].Spec.GroupPriorityMinimum
		}
	}
	sort.Sort(b)
}
