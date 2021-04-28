package crdinstall

import (
	"context"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
)

func CRDReady(ctx context.Context, config *rest.Config) (bool, error) {
	dc, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return false, err
	}
	apiresources, err := restmapper.GetAPIGroupResources(dc)
	if err != nil {
		return false, err
	}
	for _, apiGroupResource := range apiresources {
		if apiGroupResource.Group.Name == "apiextensions.k8s.io" {
			return true, nil
		}
	}
	return false, nil
}
