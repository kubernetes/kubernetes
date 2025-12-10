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

	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
)

// byPriority can be used in sort.Sort to sort specs with their priorities.
type byPriority struct {
	apiServices     []*apiregistrationv1.APIService
	groupPriorities map[string]int32
}

func (a byPriority) Len() int { return len(a.apiServices) }
func (a byPriority) Swap(i, j int) {
	a.apiServices[i], a.apiServices[j] = a.apiServices[j], a.apiServices[i]
}
func (a byPriority) Less(i, j int) bool {
	// All local specs will come first
	if a.apiServices[i].Spec.Service == nil && a.apiServices[j].Spec.Service != nil {
		return true
	}
	if a.apiServices[i].Spec.Service != nil && a.apiServices[j].Spec.Service == nil {
		return false
	}
	// WARNING: This will result in not following priorities for local APIServices.
	if a.apiServices[i].Spec.Service == nil {
		// Sort local specs with their name. This is the order in the delegation chain (aggregator first).
		return a.apiServices[i].Name < a.apiServices[j].Name
	}
	var iPriority, jPriority int32
	if a.apiServices[i].Spec.Group == a.apiServices[j].Spec.Group {
		iPriority = a.apiServices[i].Spec.VersionPriority
		jPriority = a.apiServices[i].Spec.VersionPriority
	} else {
		iPriority = a.groupPriorities[a.apiServices[i].Spec.Group]
		jPriority = a.groupPriorities[a.apiServices[j].Spec.Group]
	}
	if iPriority != jPriority {
		// Sort by priority, higher first
		return iPriority > jPriority
	}
	// Sort by service name.
	return a.apiServices[i].Name < a.apiServices[j].Name
}

func sortByPriority(apiServices []*apiregistrationv1.APIService) {
	b := byPriority{
		apiServices:     apiServices,
		groupPriorities: map[string]int32{},
	}
	for _, apiService := range apiServices {
		if apiService.Spec.Service == nil {
			continue
		}
		if pr, found := b.groupPriorities[apiService.Spec.Group]; !found || apiService.Spec.GroupPriorityMinimum > pr {
			b.groupPriorities[apiService.Spec.Group] = apiService.Spec.GroupPriorityMinimum
		}
	}
	sort.Sort(b)
}
