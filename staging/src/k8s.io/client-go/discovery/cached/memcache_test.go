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

package cached

import (
	"testing"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakediscovery "k8s.io/client-go/discovery/fake"
	"k8s.io/client-go/kubernetes/fake"
)

func TestClient(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	fakeDiscovery := fakeClient.Discovery().(*fakediscovery.FakeDiscovery)
	fakeDiscovery.Resources = append(fakeDiscovery.Resources, &metav1.APIResourceList{
		GroupVersion: "astronomy/v8beta1",
		APIResources: []metav1.APIResource{{
			Name:         "dwarfplanets",
			SingularName: "dwarfplanet",
			Namespaced:   true,
			Kind:         "DwarfPlanet",
			ShortNames:   []string{"dp"},
		}},
	})
	c := NewMemCacheClient(fakeDiscovery)
	g, err := c.ServerGroups()
	assert.Error(t, err, "unexpectedly get cached server groups")
	assert.False(t, c.Fresh(), "expect not fresh")

	c.Invalidate()

	assert.True(t, c.Fresh(), "expect not fresh")

	g, err = c.ServerGroups()
	assert.NoError(t, err, "fail to get group list")
	groupList, err := fakeDiscovery.ServerGroups()
	assert.NoError(t, err, "fail to get group list")
	assert.Equal(t, g, groupList, "server group mismatched")
	r, err := c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	assert.NoError(t, err, "fail to get server resource")
	groupResources, err := fakeDiscovery.ServerResourcesForGroupVersion("astronomy/v8beta1")
	assert.NoError(t, err, "fail to get server resource")
	assert.Equal(t, groupResources, r, "server resources mismatched")

	fakeClient.Fake.Lock()
	fakeClient.Fake.Resources = append(fakeClient.Fake.Resources, &metav1.APIResourceList{
		GroupVersion: "astronomy/v8beta1",
		APIResources: []metav1.APIResource{{
			Name:         "stars",
			SingularName: "star",
			Namespaced:   true,
			Kind:         "Star",
			ShortNames:   []string{"s"},
		}},
	})
	fakeClient.Fake.Unlock()

	c.Invalidate()
	groupResources, err = fakeDiscovery.ServerResourcesForGroupVersion("astronomy/v8beta1")
	assert.NoError(t, err, "fail to get server resource")
	r, err = c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	assert.NoError(t, err, "fail to server resources after invalidate")
	assert.Equal(t, groupResources, r, "server resources mismatched after invalidate")
}
