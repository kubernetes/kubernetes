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

package filters

import (
	"net/http"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestGetAttribs(t *testing.T) {
	r := &requestAttributeGetter{api.NewRequestContextMapper(), &apiserver.RequestInfoResolver{APIPrefixes: sets.NewString("api", "apis"), GrouplessAPIPrefixes: sets.NewString("api")}}

	testcases := map[string]struct {
		Verb               string
		Path               string
		ExpectedAttributes *authorizer.AttributesRecord
	}{
		"non-resource root": {
			Verb: "POST",
			Path: "/",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb: "post",
				Path: "/",
			},
		},
		"non-resource api prefix": {
			Verb: "GET",
			Path: "/api/",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb: "get",
				Path: "/api/",
			},
		},
		"non-resource group api prefix": {
			Verb: "GET",
			Path: "/apis/extensions/",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb: "get",
				Path: "/apis/extensions/",
			},
		},

		"resource": {
			Verb: "POST",
			Path: "/api/v1/nodes/mynode",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "create",
				Path:            "/api/v1/nodes/mynode",
				ResourceRequest: true,
				Resource:        "nodes",
				APIVersion:      "v1",
				Name:            "mynode",
			},
		},
		"namespaced resource": {
			Verb: "PUT",
			Path: "/api/v1/namespaces/myns/pods/mypod",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "update",
				Path:            "/api/v1/namespaces/myns/pods/mypod",
				ResourceRequest: true,
				Namespace:       "myns",
				Resource:        "pods",
				APIVersion:      "v1",
				Name:            "mypod",
			},
		},
		"API group resource": {
			Verb: "GET",
			Path: "/apis/extensions/v1beta1/namespaces/myns/jobs",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "list",
				Path:            "/apis/extensions/v1beta1/namespaces/myns/jobs",
				ResourceRequest: true,
				APIGroup:        extensions.GroupName,
				APIVersion:      "v1beta1",
				Namespace:       "myns",
				Resource:        "jobs",
			},
		},
	}

	for k, tc := range testcases {
		req, _ := http.NewRequest(tc.Verb, tc.Path, nil)
		attribs := r.GetAttribs(req)
		if !reflect.DeepEqual(attribs, tc.ExpectedAttributes) {
			t.Errorf("%s: expected\n\t%#v\ngot\n\t%#v", k, tc.ExpectedAttributes, attribs)
		}
	}
}
