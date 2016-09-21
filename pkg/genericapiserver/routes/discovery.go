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
