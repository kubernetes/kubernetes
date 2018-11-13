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
	"net/http"
	"reflect"
	"sync"
	"testing"

	errorsutil "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/discovery/fake"
)

type resourceMapEntry struct {
	list *metav1.APIResourceList
	err  error
}

type fakeDiscovery struct {
	*fake.FakeDiscovery

	lock         sync.Mutex
	groupList    *metav1.APIGroupList
	groupListErr error
	resourceMap  map[string]*resourceMapEntry
}

func (c *fakeDiscovery) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if rl, ok := c.resourceMap[groupVersion]; ok {
		return rl.list, rl.err
	}
	return nil, errors.New("doesn't exist")
}

func (c *fakeDiscovery) ServerGroups() (*metav1.APIGroupList, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if c.groupList == nil {
		return nil, errors.New("doesn't exist")
	}
	return c.groupList, c.groupListErr
}

func TestClient(t *testing.T) {
	fake := &fakeDiscovery{
		groupList: &metav1.APIGroupList{
			Groups: []metav1.APIGroup{{
				Name: "astronomy",
				Versions: []metav1.GroupVersionForDiscovery{{
					GroupVersion: "astronomy/v8beta1",
					Version:      "v8beta1",
				}},
			}},
		},
		resourceMap: map[string]*resourceMapEntry{
			"astronomy/v8beta1": {
				list: &metav1.APIResourceList{
					GroupVersion: "astronomy/v8beta1",
					APIResources: []metav1.APIResource{{
						Name:         "dwarfplanets",
						SingularName: "dwarfplanet",
						Namespaced:   true,
						Kind:         "DwarfPlanet",
						ShortNames:   []string{"dp"},
					}},
				},
			},
		},
	}

	c := NewMemCacheClient(fake)
	if c.Fresh() {
		t.Errorf("Expected not fresh.")
	}
	g, err := c.ServerGroups()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !c.Fresh() {
		t.Errorf("Expected fresh.")
	}
	c.Invalidate()
	if c.Fresh() {
		t.Errorf("Expected not fresh.")
	}

	g, err = c.ServerGroups()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.groupList, g; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
	if !c.Fresh() {
		t.Errorf("Expected fresh.")
	}
	r, err := c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8beta1"].list, r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	fake.lock.Lock()
	fake.resourceMap = map[string]*resourceMapEntry{
		"astronomy/v8beta1": {
			list: &metav1.APIResourceList{
				GroupVersion: "astronomy/v8beta1",
				APIResources: []metav1.APIResource{{
					Name:         "stars",
					SingularName: "star",
					Namespaced:   true,
					Kind:         "Star",
					ShortNames:   []string{"s"},
				}},
			},
		},
	}
	fake.lock.Unlock()

	c.Invalidate()
	r, err = c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8beta1"].list, r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestServerGroupsFails(t *testing.T) {
	fake := &fakeDiscovery{
		groupList: &metav1.APIGroupList{
			Groups: []metav1.APIGroup{{
				Name: "astronomy",
				Versions: []metav1.GroupVersionForDiscovery{{
					GroupVersion: "astronomy/v8beta1",
					Version:      "v8beta1",
				}},
			}},
		},
		groupListErr: errors.New("some error"),
		resourceMap: map[string]*resourceMapEntry{
			"astronomy/v8beta1": {
				list: &metav1.APIResourceList{
					GroupVersion: "astronomy/v8beta1",
					APIResources: []metav1.APIResource{{
						Name:         "dwarfplanets",
						SingularName: "dwarfplanet",
						Namespaced:   true,
						Kind:         "DwarfPlanet",
						ShortNames:   []string{"dp"},
					}},
				},
			},
		},
	}

	c := NewMemCacheClient(fake)
	if c.Fresh() {
		t.Errorf("Expected not fresh.")
	}
	_, err := c.ServerGroups()
	if err == nil {
		t.Errorf("Expected error")
	}
	if c.Fresh() {
		t.Errorf("Expected not fresh.")
	}
	fake.lock.Lock()
	fake.groupListErr = nil
	fake.lock.Unlock()
	r, err := c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8beta1"].list, r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
	if !c.Fresh() {
		t.Errorf("Expected not fresh.")
	}
}

func TestPartialPermanentFailure(t *testing.T) {
	fake := &fakeDiscovery{
		groupList: &metav1.APIGroupList{
			Groups: []metav1.APIGroup{{
				Name: "astronomy",
				Versions: []metav1.GroupVersionForDiscovery{{
					GroupVersion: "astronomy/v8beta1",
					Version:      "v8beta1",
				}, {
					GroupVersion: "astronomy2/v8beta1",
					Version:      "v8beta1",
				}},
			}},
		},
		resourceMap: map[string]*resourceMapEntry{
			"astronomy/v8beta1": {
				err: errors.New("some permanent error"),
			},
			"astronomy2/v8beta1": {
				list: &metav1.APIResourceList{
					GroupVersion: "astronomy2/v8beta1",
					APIResources: []metav1.APIResource{{
						Name:         "dwarfplanets",
						SingularName: "dwarfplanet",
						Namespaced:   true,
						Kind:         "DwarfPlanet",
						ShortNames:   []string{"dp"},
					}},
				},
			},
		},
	}

	c := NewMemCacheClient(fake)
	if c.Fresh() {
		t.Errorf("Expected not fresh.")
	}
	r, err := c.ServerResourcesForGroupVersion("astronomy2/v8beta1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy2/v8beta1"].list, r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
	_, err = c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err == nil {
		t.Errorf("Expected error, got nil")
	}

	fake.lock.Lock()
	fake.resourceMap["astronomy/v8beta1"] = &resourceMapEntry{
		list: &metav1.APIResourceList{
			GroupVersion: "astronomy/v8beta1",
			APIResources: []metav1.APIResource{{
				Name:         "dwarfplanets",
				SingularName: "dwarfplanet",
				Namespaced:   true,
				Kind:         "DwarfPlanet",
				ShortNames:   []string{"dp"},
			}},
		},
		err: nil,
	}
	fake.lock.Unlock()
	// We don't retry permanent errors, so it should fail.
	_, err = c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err == nil {
		t.Errorf("Expected error, got nil")
	}
	c.Invalidate()

	// After Invalidate, we should retry.
	r, err = c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8beta1"].list, r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestPartialRetryableFailure(t *testing.T) {
	fake := &fakeDiscovery{
		groupList: &metav1.APIGroupList{
			Groups: []metav1.APIGroup{{
				Name: "astronomy",
				Versions: []metav1.GroupVersionForDiscovery{{
					GroupVersion: "astronomy/v8beta1",
					Version:      "v8beta1",
				}, {
					GroupVersion: "astronomy2/v8beta1",
					Version:      "v8beta1",
				}},
			}},
		},
		resourceMap: map[string]*resourceMapEntry{
			"astronomy/v8beta1": {
				err: &errorsutil.StatusError{
					ErrStatus: metav1.Status{
						Message: "Some retryable error",
						Code:    int32(http.StatusServiceUnavailable),
						Reason:  metav1.StatusReasonServiceUnavailable,
					},
				},
			},
			"astronomy2/v8beta1": {
				list: &metav1.APIResourceList{
					GroupVersion: "astronomy2/v8beta1",
					APIResources: []metav1.APIResource{{
						Name:         "dwarfplanets",
						SingularName: "dwarfplanet",
						Namespaced:   true,
						Kind:         "DwarfPlanet",
						ShortNames:   []string{"dp"},
					}},
				},
			},
		},
	}

	c := NewMemCacheClient(fake)
	if c.Fresh() {
		t.Errorf("Expected not fresh.")
	}
	r, err := c.ServerResourcesForGroupVersion("astronomy2/v8beta1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := fake.resourceMap["astronomy2/v8beta1"].list, r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
	_, err = c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err == nil {
		t.Errorf("Expected error, got nil")
	}

	fake.lock.Lock()
	fake.resourceMap["astronomy/v8beta1"] = &resourceMapEntry{
		list: &metav1.APIResourceList{
			GroupVersion: "astronomy/v8beta1",
			APIResources: []metav1.APIResource{{
				Name:         "dwarfplanets",
				SingularName: "dwarfplanet",
				Namespaced:   true,
				Kind:         "DwarfPlanet",
				ShortNames:   []string{"dp"},
			}},
		},
		err: nil,
	}
	fake.lock.Unlock()
	// We should retry retryable error even without Invalidate() being called,
	// so no error is expected.
	r, err = c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8beta1"].list, r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	// Check that the last result was cached and we don't retry further.
	fake.lock.Lock()
	fake.resourceMap["astronomy/v8beta1"].err = errors.New("some permanent error")
	fake.lock.Unlock()
	r, err = c.ServerResourcesForGroupVersion("astronomy/v8beta1")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if e, a := fake.resourceMap["astronomy/v8beta1"].list, r; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}
