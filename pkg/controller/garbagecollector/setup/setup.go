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

package setup

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/discovery"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/memcachediscovery"
)

func GetDeletableResources(discoveryClient discovery.DiscoveryInterface) (map[schema.GroupVersionResource]struct{}, error) {
	preferredResources, err := discoveryClient.ServerPreferredResources()
	if err != nil {
		return nil, fmt.Errorf("failed to get supported resources from server: %v", err)
	}
	deletableResources := discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"delete"}}, preferredResources)
	deletableGroupVersionResources, err := discovery.GroupVersionResources(deletableResources)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse resources from server: %v", err)
	}
	return deletableGroupVersionResources, nil
}

func RESTMapper(discoveryClient discovery.DiscoveryInterface, pollPeriod time.Duration) (restMapper *discovery.DeferredDiscoveryRESTMapper, syncThread func(gc *garbagecollector.GarbageCollector, stopCh <-chan struct{})) {
	restMapper = discovery.NewDeferredDiscoveryRESTMapper(
		memcachediscovery.NewClient(discoveryClient),
		meta.InterfacesForUnstructured,
	)
	restMapper.Reset()
	return restMapper, func(gc *garbagecollector.GarbageCollector, stopCh <-chan struct{}) {
		t := time.NewTicker(pollPeriod)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				restMapper.Reset()
			case <-stopCh:
				return
			}
			deletableResources, err := GetDeletableResources(discoveryClient)
			if err != nil {
				utilruntime.HandleError(err)
				continue
			}
			if err := gc.SyncResourceMonitors(deletableResources); err != nil {
				utilruntime.HandleError(err)
			}
		}
	}
}
