/*
Copyright The Kubernetes Authors.

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

package openapi

import (
	"testing"

	restful "github.com/emicklei/go-restful/v3"

	"k8s.io/kube-openapi/pkg/common/restfuladapter"
)

func TestGetOperationIDAndTagsFromRoute(t *testing.T) {
	for _, tc := range []struct {
		op, path, wantID string
		wantTags         []string
		wantErr          bool
	}{
		{op: "listNamespacedPod", path: "/api/v1/namespaces/{namespace}/pods", wantID: "listCoreV1NamespacedPod", wantTags: []string{"core_v1"}},
		{op: "createDeployment", path: "/apis/apps/v1/deployments", wantID: "createAppsV1Deployment", wantTags: []string{"apps_v1"}},
		{op: "getFoo", path: "/healthz", wantID: "getFoo", wantTags: []string{"healthz"}},
		{op: "noverb", path: "/api/v1/pods", wantID: "noverb", wantErr: true},
	} {
		r := &restfuladapter.RouteAdapter{Route: &restful.Route{Operation: tc.op, Path: tc.path}}
		id, tags, err := GetOperationIDAndTagsFromRoute(r)
		if (err != nil) != tc.wantErr {
			t.Errorf("%s %s: err = %v, wantErr %v", tc.op, tc.path, err, tc.wantErr)
		}
		if id != tc.wantID {
			t.Errorf("%s %s: id = %q, want %q", tc.op, tc.path, id, tc.wantID)
		}
		if !tc.wantErr && (len(tags) != len(tc.wantTags) || (len(tags) > 0 && tags[0] != tc.wantTags[0])) {
			t.Errorf("%s %s: tags = %v, want %v", tc.op, tc.path, tags, tc.wantTags)
		}
	}
}
