/*
Copyright 2022 The Kubernetes Authors.

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

package dra

import (
	"sync"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	dra "k8s.io/kubernetes/pkg/kubelet/cm/dra/plugin"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// resource contains resource attributes required
// to prepare and unprepare the resource
type resource struct {
	// name of the DRA driver
	driverName string

	// name is an unique resource name
	name string

	// claimUID is an UID of the resource claim
	claimUID types.UID

	// resourcePluginClient is an instance of the GRPC DRA client
	// that's used to query node resource plugin
	resourcePluginClient dra.DRAClient

	// cdiDevice is a list of CDI devices returned by the
	// GRPC API call NodePrepareResource
	cdiDevice []string

	// annotations is a list of container annotations associated with
	// a prepared resource
	annotations []kubecontainer.Annotation
}

// podResources is a map by pod UID of resources used by pod
type podResources struct {
	sync.RWMutex
	resources map[types.UID]map[string]map[string]*resource
}

// newPodResources is a function that returns object of podResources
func newPodResources() *podResources {
	return &podResources{
		resources: make(map[types.UID]map[string]map[string]*resource),
	}
}

func (pres *podResources) insert(podUID types.UID, contName, resName string, res *resource) {
	pres.Lock()
	defer pres.Unlock()

	if _, podExists := pres.resources[podUID]; !podExists {
		pres.resources[podUID] = make(map[string]map[string]*resource)
	}
	if _, contExists := pres.resources[podUID][contName]; !contExists {
		pres.resources[podUID][contName] = make(map[string]*resource)
	}
	pres.resources[podUID][contName][resName] = res
}

func (pres *podResources) pods() sets.String {
	pres.RLock()
	defer pres.RUnlock()

	ret := sets.NewString()
	for podUID := range pres.resources {
		ret.Insert(string(podUID))
	}
	return ret
}

func (pres *podResources) delete(podUIDs []string) {
	pres.Lock()
	defer pres.Unlock()
	for _, podUID := range podUIDs {
		delete(pres.resources, types.UID(podUID))
	}
}

func (pres *podResources) prepared(podUID types.UID, contName, resName string) bool {
	res := pres.get(podUID, contName, resName)
	return res != nil
}

func (pres *podResources) getPodResources(podUID types.UID) []*resource {
	pres.Lock()
	defer pres.Unlock()

	resources := []*resource{}
	for contName := range pres.resources[podUID] {
		for _, resource := range pres.resources[podUID][contName] {
			resources = append(resources, resource)
		}
	}

	return resources
}

func (pres *podResources) get(podUID types.UID, contName, resName string) *resource {
	pres.Lock()
	defer pres.Unlock()

	if _, podExists := pres.resources[podUID]; podExists {
		if _, contExists := pres.resources[podUID][contName]; contExists {
			if res, resExists := pres.resources[podUID][contName][resName]; resExists {
				return res
			}
		}
	}
	return nil
}
