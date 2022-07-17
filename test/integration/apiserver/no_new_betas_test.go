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

package apiserver

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestNoNewBetaAPIsByDefault(t *testing.T) {
	// yes, this *is* a copy/paste from somewhere else.  We really do mean it when we say you shouldn't be modifying
	// this list and this test was created to make it more painful.
	// legacyBetaEnabledByDefaultResources is the list of beta resources we enable.  You may not add to this list
	legacyBetaEnabledByDefaultResources := map[schema.GroupVersionResource]bool{
		gvr("autoscaling", "v2beta1", "horizontalpodautoscalers"):                     true, // remove in 1.25
		gvr("autoscaling", "v2beta2", "horizontalpodautoscalers"):                     true, // remove in 1.26
		gvr("batch", "v1beta1", "jobtemplates"):                                       true, // remove in 1.25
		gvr("batch", "v1beta1", "cronjobs"):                                           true, // remove in 1.25
		gvr("discovery.k8s.io", "v1beta1", "endpointslices"):                          true, // remove in 1.25
		gvr("events.k8s.io", "v1beta1", "events"):                                     true, // remove in 1.25
		gvr("node.k8s.io", "v1beta1", "runtimeclasses"):                               true, // remove in 1.25
		gvr("policy", "v1beta1", "poddisruptionbudgets"):                              true, // remove in 1.25
		gvr("policy", "v1beta1", "podsecuritypolicies"):                               true, // remove in 1.25
		gvr("storage.k8s.io", "v1beta1", "csinodes"):                                  true, // remove in 1.25
		gvr("storage.k8s.io", "v1beta1", "csistoragecapacities"):                      true, // remove in 1.27
		gvr("flowcontrol.apiserver.k8s.io", "v1beta1", "flowschemas"):                 true, // remove in 1.26
		gvr("flowcontrol.apiserver.k8s.io", "v1beta1", "prioritylevelconfigurations"): true, // remove in 1.26
		gvr("flowcontrol.apiserver.k8s.io", "v1beta2", "flowschemas"):                 true, // remove in 1.29
		gvr("flowcontrol.apiserver.k8s.io", "v1beta2", "prioritylevelconfigurations"): true, // remove in 1.29
	}

	// legacyBetaResourcesWithoutStableEquivalents contains those groupresources that were enabled by default as beta
	// before we changed that policy and do not have stable versions. These resources are allowed to have additional
	// beta versions enabled by default.  Nothing new should be added here.  There are no future exceptions because there
	// are no more beta resources enabled by default.
	legacyBetaResourcesWithoutStableEquivalents := map[schema.GroupResource]bool{
		gvr("storage.k8s.io", "v1beta1", "csistoragecapacities").GroupResource():                      true,
		gvr("flowcontrol.apiserver.k8s.io", "v1beta1", "flowschemas").GroupResource():                 true,
		gvr("flowcontrol.apiserver.k8s.io", "v1beta1", "prioritylevelconfigurations").GroupResource(): true,
	}

	// if you found this because you want to create an integration test for your new beta API, the method you're looking for
	// is this setupWithResources method and you need to pass the resource you want to enable into it.
	kubeClient, _, tearDownFn := setupWithResources(t,
		[]schema.GroupVersion{},
		[]schema.GroupVersionResource{},
	)
	defer tearDownFn()

	_, allResourceLists, err := kubeClient.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Error(err)
	}

	for _, currResourceList := range allResourceLists {
		for _, currResource := range currResourceList.APIResources {
			if !strings.Contains(currResource.Version, "beta") {
				continue // skip non-beta apis
			}
			if strings.Contains(currResource.Name, "/") {
				continue // skip subresources
			}
			enabledGVR := schema.GroupVersionResource{
				Group:    currResource.Group,
				Version:  currResource.Version,
				Resource: currResource.Name,
			}
			if legacyBetaEnabledByDefaultResources[enabledGVR] {
				continue
			}
			if legacyBetaResourcesWithoutStableEquivalents[enabledGVR.GroupResource()] {
				continue
			}

			t.Errorf("%v is a new beta API.  New beta APIs may not be enabled by default.  "+
				"See https://github.com/kubernetes/enhancements/blob/0ad0fc8269165ca300d05ca51c7ce190a79976a5/keps/sig-architecture/3136-beta-apis-off-by-default/README.md "+
				"for more details.", enabledGVR)
		}
	}
}
