package garbagecollector

import (
	"strings"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

// TODO: put this function to the discovery package.
func ServerPreferredGroupVersionResources(discoveryClient discovery.DiscoveryInterface) ([]unversioned.GroupVersionResource, error) {
	results := []unversioned.GroupVersionResource{}
	serverGroupList, err := discoveryClient.ServerGroups()
	if err != nil {
		return results, err
	}

	allErrs := []error{}
	for _, apiGroup := range serverGroupList.Groups {
		preferredVersion := apiGroup.PreferredVersion
		apiResourceList, err := discoveryClient.ServerResourcesForGroupVersion(preferredVersion.GroupVersion)
		if err != nil {
			allErrs = append(allErrs, err)
			continue
		}
		groupVersion := unversioned.GroupVersion{Group: apiGroup.Name, Version: preferredVersion.Version}
		// TODO: we need to skip bindings, extensions/v1beta1/replicationcontrollers (the dummy). Discovery API doesn't tell what operations are allowed.
		for _, apiResource := range apiResourceList.APIResources {
			if strings.Contains(apiResource.Name, "/") {
				continue
			}
			results = append(results, groupVersion.WithResource(apiResource.Name))
		}
	}
	return results, utilerrors.NewAggregate(allErrs)
}
