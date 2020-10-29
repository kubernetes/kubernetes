package apiserver

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

// alwaysLocalDelegatePrefixes specify a list of API paths that we want to delegate to Kubernetes API server
// instead of handling with OpenShift API server.
var alwaysLocalDelegatePathPrefixes = sets.NewString()

// AddAlwaysLocalDelegateForPrefix will cause the given URL prefix always be served by local API server (kube apiserver).
// This allows to move some resources from aggregated API server into CRD.
func AddAlwaysLocalDelegateForPrefix(prefix string) {
	if alwaysLocalDelegatePathPrefixes.Has(prefix) {
		return
	}
	alwaysLocalDelegatePathPrefixes.Insert(prefix)
}

var overlappingGroupVersion = map[schema.GroupVersion]bool{}

// AddOverlappingGroupVersion will stop the CRD registration controller from trying to manage an APIService.
func AddOverlappingGroupVersion(groupVersion schema.GroupVersion) {
	overlappingGroupVersion[groupVersion] = true
}

var alwaysLocalDelegateGroupResource = map[schema.GroupResource]bool{}

func AddAlwaysLocalDelegateGroupResource(groupResource schema.GroupResource) {
	alwaysLocalDelegateGroupResource[groupResource] = true
}

func APIServiceAlreadyExists(groupVersion schema.GroupVersion) bool {
	if overlappingGroupVersion[groupVersion] {
		return true
	}

	testPrefix := fmt.Sprintf("/apis/%s/%s/", groupVersion.Group, groupVersion.Version)
	for _, prefix := range alwaysLocalDelegatePathPrefixes.List() {
		if strings.HasPrefix(prefix, testPrefix) {
			return true
		}
	}
	return false
}
