/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	expapi "k8s.io/kubernetes/pkg/apis/extensions"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/sets"
)

type FakeAPIInterface struct {
	removed   []string
	installed []*expapi.ThirdPartyResource
	t         *testing.T
	gvrs      []metav1.GroupVersionResource
}

func (f *FakeAPIInterface) RemoveThirdPartyResource(gvr metav1.GroupVersionResource) error {
	path := MakeThirdPartyPath(gvr.Group) + "/" + gvr.Version
	f.removed = append(f.removed, path)
	return nil
}

func (f *FakeAPIInterface) InstallThirdPartyResource(rsrc *expapi.ThirdPartyResource) error {
	f.installed = append(f.installed, rsrc)
	kind, group, _ := thirdpartyresourcedata.ExtractApiGroupAndKind(rsrc)
	for _, version := range rsrc.Versions {
		plural, _ := meta.KindToResource(schema.GroupVersionKind{
			Group:   group,
			Version: version.Name,
			Kind:    kind,
		})
		gvr := metav1.GroupVersionResource{
			Group:    group,
			Version:  version.Name,
			Resource: plural.Resource,
		}

		found := false
		for _, installedGvr := range f.gvrs {
			if reflect.DeepEqual(installedGvr, gvr) {
				found = true
				break
			}
		}
		if !found {
			f.gvrs = append(f.gvrs, gvr)
		}
	}
	return nil
}

func (f *FakeAPIInterface) HasThirdPartyResource(rsrc *expapi.ThirdPartyResource) (bool, error) {
	if f.gvrs == nil {
		return false, nil
	}
	kind, group, _ := thirdpartyresourcedata.ExtractApiGroupAndKind(rsrc)
	for _, version := range rsrc.Versions {
		plural, _ := meta.KindToResource(schema.GroupVersionKind{
			Group:   group,
			Version: version.Name,
			Kind:    kind,
		})
		gvr := metav1.GroupVersionResource{
			Group:    group,
			Version:  version.Name,
			Resource: plural.Resource,
		}

		for _, installedGvr := range f.gvrs {
			if !reflect.DeepEqual(installedGvr, gvr) {
				return false, nil
			}
		}
	}
	return true, nil
}

func (f *FakeAPIInterface) ListThirdPartyResources() []metav1.GroupVersionResource {
	return f.gvrs
}

func TestSyncAPIs(t *testing.T) {
	resourceNamed := func(name string, versions ...string) expapi.ThirdPartyResource {
		apiVersions := []expapi.APIVersion{}
		for _, version := range versions {
			apiVersions = append(apiVersions, expapi.APIVersion{
				Name: version,
			})
		}
		return expapi.ThirdPartyResource{
			ObjectMeta: api.ObjectMeta{Name: name},
			Versions:   apiVersions,
		}
	}

	tests := []struct {
		list              *expapi.ThirdPartyResourceList
		gvrs              []metav1.GroupVersionResource
		expectedInstalled []string
		expectedRemoved   []string
		name              string
	}{
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					resourceNamed("foo.example.com", "v1"),
				},
			},
			expectedInstalled: []string{"foo.example.com"},
			name:              "simple add",
		},
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					resourceNamed("foo.example.com", "v1"),
				},
			},
			gvrs: []metav1.GroupVersionResource{
				{Group: "example.com", Version: "v1", Resource: "foos"},
			},
			name: "does nothing",
		},
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					resourceNamed("foo.example.com", "v1", "v2"),
				},
			},
			gvrs: []metav1.GroupVersionResource{
				{Group: "example.com", Version: "v1", Resource: "foos"},
			},
			name:              "add new version",
			expectedInstalled: []string{"foo.example.com"},
		},
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					resourceNamed("foo.example.com", "v1"),
					resourceNamed("bar.example.com", "v2"),
				},
			},
			gvrs: []metav1.GroupVersionResource{
				{Group: "example.com", Version: "v1", Resource: "foos"},
			},
			expectedInstalled: []string{"bar.example.com"},
			name:              "adds new kind with existing group",
		},
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					resourceNamed("foo.example.com", "v1"),
				},
			},
			gvrs: []metav1.GroupVersionResource{
				{Group: "company.com", Version: "v1", Resource: "foos"},
			},
			expectedInstalled: []string{"foo.example.com"},
			expectedRemoved:   []string{"/apis/company.com/v1"},
			name:              "removes with existing",
		},
	}

	for _, test := range tests {
		fake := FakeAPIInterface{
			gvrs: test.gvrs,
			t:    t,
		}

		cntrl := ThirdPartyController{master: &fake}

		if err := cntrl.syncResourceList(test.list); err != nil {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
		}
		if len(test.expectedInstalled) != len(fake.installed) {
			t.Errorf("[%s] unexpected installed APIs: %d, expected %d (%#v)", test.name, len(fake.installed), len(test.expectedInstalled), fake.installed[0])
			continue
		} else {
			names := sets.String{}
			for ix := range fake.installed {
				names.Insert(fake.installed[ix].Name)
			}
			for _, name := range test.expectedInstalled {
				if !names.Has(name) {
					t.Errorf("[%s] missing installed API: %s", test.name, name)
				}
			}
		}
		if len(test.expectedRemoved) != len(fake.removed) {
			t.Errorf("[%s] unexpected installed APIs: %d, expected %d", test.name, len(fake.removed), len(test.expectedRemoved))
			continue
		} else {
			names := sets.String{}
			names.Insert(fake.removed...)
			for _, name := range test.expectedRemoved {
				if !names.Has(name) {
					t.Errorf("[%s] missing removed API: %s (%s)", test.name, name, names)
				}
			}
		}
	}
}
