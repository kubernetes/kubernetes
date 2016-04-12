/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package master

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	expapi "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/thirdpartyresourcedata"
	"k8s.io/kubernetes/pkg/util/sets"
)

type FakeAPIInterface struct {
	removed              []string
	installed            []*expapi.ThirdPartyResource
	services             []string
	removedWebservices   []string
	installedWebservices []string
	apis                 []string
	t                    *testing.T
}

func (f *FakeAPIInterface) RemoveThirdPartyResource(path string) error {
	f.removed = append(f.removed, path)
	group, kind := getThirdPartyGroupKind(path)
	for _, webservice := range f.installedWebservices {
		found := false
		for _, installedResourcePath := range f.apis {
			installedGroup, installedKind := getThirdPartyGroupKind(installedResourcePath)
			if kind != installedKind && group != installedGroup && strings.HasPrefix(webservice, makeThirdPartyPath(installedGroup)) {
				found = true
			}
		}
		if !found {
			f.removedWebservices = append(f.removedWebservices, webservice)
		}
	}
	return nil
}

func (f *FakeAPIInterface) InstallThirdPartyResource(rsrc *expapi.ThirdPartyResource) error {
	f.installed = append(f.installed, rsrc)
	kind, group, _ := thirdpartyresourcedata.ExtractApiGroupAndKind(rsrc)
	f.apis = append(f.apis, makeThirdPartyPath(group)+"/"+strings.ToLower(kind)+"s")
	return nil
}

func (f *FakeAPIInterface) HasWebservice(group string) (bool, error) {
	for _, webservice := range f.installedWebservices {
		if strings.HasPrefix(webservice, makeThirdPartyPath(group)) {
			return true, nil
		}
	}
	return false, nil
}

func (f *FakeAPIInterface) HasThirdPartyResource(rsrc *expapi.ThirdPartyResource) (bool, error) {
	if f.apis == nil {
		return false, nil
	}
	kind, group, _ := thirdpartyresourcedata.ExtractApiGroupAndKind(rsrc)
	path := makeThirdPartyPath(group)
	for _, api := range f.apis {
		if api == path+"/"+strings.ToLower(kind)+"s" {
			return true, nil
		}
	}
	return false, nil
}

func (f *FakeAPIInterface) ListThirdPartyResources() []string {
	return f.apis
}

func TestSyncAPIs(t *testing.T) {
	tests := []struct {
		list              *expapi.ThirdPartyResourceList
		apis              []string
		expectedInstalled []string
		expectedRemoved   []string
		name              string
	}{
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo.example.com",
						},
					},
				},
			},
			expectedInstalled: []string{"foo.example.com"},
			name:              "simple add",
		},
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo.example.com",
						},
					},
				},
			},
			apis: []string{
				"/apis/example.com/foos",
				"/apis/example.com/v1/foos",
			},
			name: "does nothing",
		},
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo.example.com",
						},
					},
				},
			},
			apis: []string{
				"/apis/example.com/foos",
				"/apis/example.com/v1/foos",
				"/apis/example.co/foo",
				"/apis/example.co/v1/foo",
			},
			name: "deletes substring API",
			expectedRemoved: []string{
				"/apis/example.co/foo",
				"/apis/example.co/v1/foo",
			},
		},
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo.example.com",
						},
					},
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo.company.com",
						},
					},
				},
			},
			apis: []string{
				"/apis/company.com/foos",
				"/apis/company.com/v1/foos",
			},
			expectedInstalled: []string{"foo.example.com"},
			name:              "adds with existing",
		},
		{
			list: &expapi.ThirdPartyResourceList{
				Items: []expapi.ThirdPartyResource{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo.example.com",
						},
					},
				},
			},
			apis: []string{
				"/apis/company.com/foos",
				"/apis/company.com/v1/foos",
			},
			expectedInstalled: []string{"foo.example.com"},
			expectedRemoved:   []string{"/apis/company.com/foos", "/apis/company.com/v1/foos"},
			name:              "removes with existing",
		},
	}

	for _, test := range tests {
		fake := FakeAPIInterface{
			apis: test.apis,
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
			t.Errorf("[%s] unexpected uninstalled APIs: %d, expected %d", test.name, len(fake.removed), len(test.expectedRemoved))
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
