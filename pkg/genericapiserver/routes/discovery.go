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

package routes

import (
	"net/http"
	"sort"

	"github.com/emicklei/go-restful"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/runtime"
)

func ApisDiscovery(serializer runtime.NegotiatedSerializer, apiPrefix string, groupsForDiscovery map[string]unversioned.APIGroup, getServerAddressByClientCIDRs func(req *http.Request) []unversioned.ServerAddressByClientCIDR) *restful.WebService {
	// sort to have a deterministic order
	sortedGroups := []unversioned.APIGroup{}
	groupNames := make([]string, 0, len(groupsForDiscovery))
	for groupName := range groupsForDiscovery {
		groupNames = append(groupNames, groupName)
	}
	sort.Strings(groupNames)
	for _, groupName := range groupNames {
		sortedGroups = append(sortedGroups, groupsForDiscovery[groupName])
	}

	return apiserver.NewApisWebService(serializer, apiPrefix, func(req *restful.Request) []unversioned.APIGroup {
		serverCIDR := getServerAddressByClientCIDRs(req.Request)
		groups := make([]unversioned.APIGroup, len(sortedGroups))
		for i := range sortedGroups {
			groups[i] = sortedGroups[i]
			groups[i].ServerAddressByClientCIDRs = serverCIDR
		}
		return groups
	})
}
