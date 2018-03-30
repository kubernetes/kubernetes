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
	"errors"
	"reflect"
	"sync"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/discovery/fake"
)

type fakeDiscovery struct {
	*fake.FakeDiscovery

	lock                         sync.Mutex
	groupList                    *metav1.APIGroupList
	resourceMap                  map[string]*metav1.APIResourceList
	preferredResources           []*metav1.APIResourceList
	preferredNamespacedResources []*metav1.APIResourceList
}

func (c *fakeDiscovery) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if rl, ok := c.resourceMap[groupVersion]; ok {
		return rl, nil
	}
	return nil, errors.New("doesn't exist")
}

func (c *fakeDiscovery) ServerGroups() (*metav1.APIGroupList, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if c.groupList == nil {
		return nil, errors.New("doesn't exist")
	}
	return c.groupList, nil
}

func (c *fakeDiscovery) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if c.preferredResources == nil {
		return nil, errors.New("doesn't exist")
	}
	return c.preferredResources, nil
}

func (c *fakeDiscovery) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if c.preferredNamespacedResources == nil {
		return nil, errors.New("doesn't exist")
	}
	return c.preferredNamespacedResources, nil
}

func TestClient(t *testing.T) {
	fake := &fakeDiscovery{
		groupList: &metav1.APIGroupList{
			Groups: []metav1.APIGroup{
				{
					Name: "astronomy",
					Versions: []metav1.GroupVersionForDiscovery{{
						GroupVersion: "astronomy/v8beta1",
						Version:      "v8beta1",
					}},
				},
				{
					Name: "astronomy",
					Versions: []metav1.GroupVersionForDiscovery{{
						GroupVersion: "astronomy/v8",
						Version:      "v8",
					}},
				},
			},
		},
		resourceMap: map[string]*metav1.APIResourceList{
			"astronomy/v8beta1": {
				GroupVersion: "astronomy/v8beta1",
				APIResources: []metav1.APIResource{{
					Name:         "dwarfplanets",
					SingularName: "dwarfplanet",
					Namespaced:   true,
					Kind:         "DwarfPlanet",
					ShortNames:   []string{"dp"},
				}},
			},
			"astronomy/v8": {
				GroupVersion: "astronomy/v8",
				APIResources: []metav1.APIResource{{
					Name:         "dwarfplanets",
					SingularName: "dwarfplanet",
					Namespaced:   true,
					Kind:         "DwarfPlanet",
					ShortNames:   []string{"dp"},
				}},
			},
		},
		preferredResources: []*metav1.APIResourceList{
			{
				GroupVersion: "astronomy/v8",
				APIResources: []metav1.APIResource{{
					Name:         "dwarfplanets",
					SingularName: "dwarfplanet",
					Namespaced:   true,
					Kind:         "DwarfPlanet",
					ShortNames:   []string{"dp"},
				}},
			},
		},
		preferredNamespacedResources: []*metav1.APIResourceList{
			{
				GroupVersion: "astronomy/v8",
				APIResources: []metav1.APIResource{{
					Name:         "dwarfplanets",
					SingularName: "dwarfplanet",
					Namespaced:   true,
					Kind:         "DwarfPlanet",
					ShortNames:   []string{"dp"},
				}},
			},
		},
	}

	c := NewMemCacheClient(fake)
	g, err := c.ServerGroups()
	if err == nil {
		t.Errorf("Unexpected non-error.")
	}
	if c.Fresh() {
		t.Errorf("Expected not fresh.")
	}

	c.Invalidate()
	if !c.Fresh() {
		t.Errorf("Expected fresh.")
	}

	g, err = c.ServerGroups()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.groupList, g; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
	r, err := c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8beta1"], r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	r, err = c.ServerResourcesForGroupVersion("astronomy/v8")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8"], r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	pr, err := c.ServerPreferredResources()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8"], pr[0]; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	pnr, err := c.ServerPreferredNamespacedResources()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8"], pnr[0]; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	fake.lock.Lock()
	fake.resourceMap = map[string]*metav1.APIResourceList{
		"astronomy/v8beta1": {
			GroupVersion: "astronomy/v8beta1",
			APIResources: []metav1.APIResource{{
				Name:         "stars",
				SingularName: "star",
				Namespaced:   true,
				Kind:         "Star",
				ShortNames:   []string{"s"},
			}},
		},
		"astronomy/v8": {
			GroupVersion: "astronomy/v8",
			APIResources: []metav1.APIResource{{
				Name:         "stars",
				SingularName: "star",
				Namespaced:   true,
				Kind:         "Star",
				ShortNames:   []string{"s"},
			}},
		},
	}
	fake.preferredResources = []*metav1.APIResourceList{
		{
			GroupVersion: "astronomy/v8",
			APIResources: []metav1.APIResource{{
				Name:         "stars",
				SingularName: "star",
				Namespaced:   true,
				Kind:         "Star",
				ShortNames:   []string{"s"},
			}},
		},
	}
	fake.preferredNamespacedResources = []*metav1.APIResourceList{
		{
			GroupVersion: "astronomy/v8",
			APIResources: []metav1.APIResource{{
				Name:         "stars",
				SingularName: "star",
				Namespaced:   true,
				Kind:         "Star",
				ShortNames:   []string{"s"},
			}},
		},
	}
	fake.lock.Unlock()

	c.Invalidate()
	r, err = c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8beta1"], r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	r, err = c.ServerResourcesForGroupVersion("astronomy/v8")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8"], r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	pr, err = c.ServerPreferredResources()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8"], pr[0]; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	pnr, err = c.ServerPreferredNamespacedResources()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8"], pnr[0]; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}
