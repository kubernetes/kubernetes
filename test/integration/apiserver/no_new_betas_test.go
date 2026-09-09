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
	// if you found this because you want to create an integration test for your new beta API, the method you're looking for
	// is this setupWithResources method and you need to pass the resource you want to enable into it.
	_, kubeClient, _, tearDownFn := setupWithResources(t,
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

			t.Errorf("%v is a new beta API.  New beta APIs may not be enabled by default.  "+
				"See https://github.com/kubernetes/enhancements/blob/0ad0fc8269165ca300d05ca51c7ce190a79976a5/keps/sig-architecture/3136-beta-apis-off-by-default/README.md "+
				"for more details.", enabledGVR)
		}
	}
}
