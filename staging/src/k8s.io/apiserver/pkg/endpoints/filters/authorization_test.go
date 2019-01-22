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
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	batch "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestGetAuthorizerAttributes(t *testing.T) {
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
			Path: "/apis/batch/v1/namespaces/myns/jobs",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "list",
				Path:            "/apis/batch/v1/namespaces/myns/jobs",
				ResourceRequest: true,
				APIGroup:        batch.GroupName,
				APIVersion:      "v1",
				Namespace:       "myns",
				Resource:        "jobs",
			},
		},
	}

	for k, tc := range testcases {
		req, _ := http.NewRequest(tc.Verb, tc.Path, nil)
		req.RemoteAddr = "127.0.0.1"

		var attribs authorizer.Attributes
		var err error
		var handler http.Handler = http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			ctx := req.Context()
			attribs, err = GetAuthorizerAttributes(ctx)
		})
		handler = WithRequestInfo(handler, newTestRequestInfoResolver())
		handler.ServeHTTP(httptest.NewRecorder(), req)

		if err != nil {
			t.Errorf("%s: unexpected error: %v", k, err)
		} else if !reflect.DeepEqual(attribs, tc.ExpectedAttributes) {
			t.Errorf("%s: expected\n\t%#v\ngot\n\t%#v", k, tc.ExpectedAttributes, attribs)
		}
	}
}

type fakeAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
}

func (f fakeAuthorizer) Authorize(a authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.decision, f.reason, f.err
}

func TestAuditAnnotation(t *testing.T) {
	testcases := map[string]struct {
		authorizer         fakeAuthorizer
		decisionAnnotation string
		reasonAnnotation   string
	}{
		"decision allow": {
			fakeAuthorizer{
				authorizer.DecisionAllow,
				"RBAC: allowed to patch pod",
				nil,
			},
			"allow",
			"RBAC: allowed to patch pod",
		},
		"decision forbid": {
			fakeAuthorizer{
				authorizer.DecisionDeny,
				"RBAC: not allowed to patch pod",
				nil,
			},
			"forbid",
			"RBAC: not allowed to patch pod",
		},
		"error": {
			fakeAuthorizer{
				authorizer.DecisionNoOpinion,
				"",
				errors.New("can't parse user info"),
			},
			"",
			reasonError,
		},
	}

	scheme := runtime.NewScheme()
	negotiatedSerializer := serializer.DirectCodecFactory{CodecFactory: serializer.NewCodecFactory(scheme)}
	for k, tc := range testcases {
		audit := &auditinternal.Event{Level: auditinternal.LevelMetadata}
		handler := WithAuthorization(&fakeHTTPHandler{}, tc.authorizer, negotiatedSerializer)
		// TODO: fake audit injector

		req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
		req = withTestContext(req, nil, audit)
		req.RemoteAddr = "127.0.0.1"
		handler.ServeHTTP(httptest.NewRecorder(), req)
		assert.Equal(t, tc.decisionAnnotation, audit.Annotations[decisionAnnotationKey], k+": unexpected decision annotation")
		assert.Equal(t, tc.reasonAnnotation, audit.Annotations[reasonAnnotationKey], k+": unexpected reason annotation")
	}

}
