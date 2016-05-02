/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"encoding/json"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var data = `{
  "kind": "Foo",
  "apiVersion": "company.com/v1",
  "metadata": {
    "name": "bazz"
  },
  "someField": "hello world",
  "otherField": 1
}`

type Foo struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty" description:"standard object metadata"`

	SomeField  string `json:"someField"`
	OtherField int    `json:"otherField"`
}

type FooList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata"`

	Items []Foo `json:"items"`
}

var _ = Describe("ThirdParty resources", func() {

	f := framework.NewDefaultFramework("thirdparty")

	rsrc := &extensions.ThirdPartyResource{
		ObjectMeta: api.ObjectMeta{
			Name: "foo.company.com",
		},
		Versions: []extensions.APIVersion{
			{Name: "v1"},
		},
	}

	Context("Simple Third Party", func() {
		It("creating/deleting thirdparty objects works [Conformance]", func() {
			defer func() {
				if err := f.Client.ThirdPartyResources("").Delete(rsrc.Name); err != nil {
					framework.Failf("failed to delete third party resource: %v", err)
				}
			}()
			if _, err := f.Client.ThirdPartyResources("").Create(rsrc); err != nil {
				framework.Failf("failed to create third party resource: %v", err)
			}

			data, err := f.Client.RESTClient.Get().AbsPath("/apis/company.com/v1/foos").DoRaw()
			if err != nil {
				framework.Failf("failed to list with no objects: %v", err)
			}
			list := FooList{}
			if err := json.Unmarshal(data, &list); err != nil {
				framework.Failf("failed to decode: %v", err)
			}
			if len(list.Items) != 0 {
				framework.Failf("unexpected object: %v", list)
			}
			foo := &Foo{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				SomeField:  "bar",
				OtherField: 10,
			}
			if _, err := f.Client.RESTClient.Post().AbsPath("/apis/company.com/v1/foos").Body(foo).DoRaw(); err != nil {
				framework.Failf("failed to create: %v", err)
			}

			data, err = f.Client.RESTClient.Get().AbsPath("/apis/company.com/v1/foos/foo").DoRaw()
			if err != nil {
				framework.Failf("failed to get object: %v", err)
			}
			out := Foo{}
			if err := json.Unmarshal(data, &out); err != nil {
				framework.Failf("failed to decode: %v", err)
			}
			if out.Name != foo.Name || out.SomeField != foo.SomeField || out.OtherField != foo.OtherField {
				framework.Failf("expected: %#v, saw: %#v", foo, &out)
			}

			data, err = f.Client.RESTClient.Get().AbsPath("/apis/company.com/v1/foos").DoRaw()
			if err != nil {
				framework.Failf("failed to list with no objects: %v", err)
			}
			if err := json.Unmarshal(data, &list); err != nil {
				framework.Failf("failed to decode: %v", err)
			}
			if len(list.Items) != 1 {
				framework.Failf("unexpected object: %v", list)
			}
			if list.Items[0].Name != foo.Name || list.Items[0].SomeField != foo.SomeField || list.Items[0].OtherField != foo.OtherField {
				framework.Failf("expected: %#v, saw in list: %#v", foo, list.Items[0])
			}

			if _, err := f.Client.RESTClient.Delete().AbsPath("/apis/company.com/v1/foos").Body(foo).DoRaw(); err != nil {
				framework.Failf("failed to delete: %v", err)
			}

			data, err = f.Client.RESTClient.Get().AbsPath("/apis/company.com/v1/foos").DoRaw()
			if err != nil {
				framework.Failf("failed to list with no objects: %v", err)
			}
			if err := json.Unmarshal(data, &list); err != nil {
				framework.Failf("failed to decode: %v", err)
			}
			if len(list.Items) != 0 {
				framework.Failf("unexpected object: %v", list)
			}
		})
	})
})
