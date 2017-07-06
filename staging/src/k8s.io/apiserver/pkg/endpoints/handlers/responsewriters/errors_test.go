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

package responsewriters

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestErrors(t *testing.T) {
	internalError := errors.New("ARGH")
	fns := map[string]func(http.ResponseWriter, *http.Request){
		"BadGatewayError": BadGatewayError,
		"NotFound":        NotFound,
		"InternalError": func(w http.ResponseWriter, req *http.Request) {
			InternalError(w, req, internalError)
		},
	}
	cases := []struct {
		fn       string
		uri      string
		expected string
	}{
		{"BadGatewayError", "/get", `Bad Gateway: "/get"`},
		{"BadGatewayError", "/<script>", `Bad Gateway: "/&lt;script&gt;"`},
		{"NotFound", "/get", `Not Found: "/get"`},
		{"NotFound", "/<script&>", `Not Found: "/&lt;script&amp;&gt;"`},
		{"InternalError", "/get", `Internal Server Error: "/get": ARGH`},
		{"InternalError", "/<script>", `Internal Server Error: "/&lt;script&gt;": ARGH`},
	}
	for _, test := range cases {
		observer := httptest.NewRecorder()
		fns[test.fn](observer, &http.Request{RequestURI: test.uri})
		result := string(observer.Body.Bytes())
		if result != test.expected {
			t.Errorf("%s(..., %q) != %q, got %q", test.fn, test.uri, test.expected, result)
		}
	}
}

func TestForbidden(t *testing.T) {
	u := &user.DefaultInfo{Name: "NAME"}
	cases := []struct {
		expected    string
		attributes  authorizer.Attributes
		reason      string
		contentType string
	}{
		{`{"metadata":{},"status":"Failure","message":" \"\" is forbidden: User \"NAME\" cannot GET path \"/whatever\".","reason":"Forbidden","details":{},"code":403}
`, authorizer.AttributesRecord{User: u, Verb: "GET", Path: "/whatever"}, "", "application/json"},
		{`{"metadata":{},"status":"Failure","message":" \"\" is forbidden: User \"NAME\" cannot GET path \"/\u0026lt;script\u0026gt;\".","reason":"Forbidden","details":{},"code":403}
`, authorizer.AttributesRecord{User: u, Verb: "GET", Path: "/<script>"}, "", "application/json"},
		{`{"metadata":{},"status":"Failure","message":"pod \"\" is forbidden: User \"NAME\" cannot GET pod at the cluster scope.","reason":"Forbidden","details":{"kind":"pod"},"code":403}
`, authorizer.AttributesRecord{User: u, Verb: "GET", Resource: "pod", ResourceRequest: true}, "", "application/json"},
		{`{"metadata":{},"status":"Failure","message":"pod.v2 \"\" is forbidden: User \"NAME\" cannot GET pod.v2/quota in the namespace \"test\".","reason":"Forbidden","details":{"group":"v2","kind":"pod"},"code":403}
`, authorizer.AttributesRecord{User: u, Verb: "GET", Namespace: "test", APIGroup: "v2", Resource: "pod", Subresource: "quota", ResourceRequest: true}, "", "application/json"},
	}
	for _, test := range cases {
		observer := httptest.NewRecorder()
		scheme := runtime.NewScheme()
		negotiatedSerializer := serializer.DirectCodecFactory{CodecFactory: serializer.NewCodecFactory(scheme)}
		Forbidden(request.NewDefaultContext(), test.attributes, observer, &http.Request{}, test.reason, negotiatedSerializer)
		result := string(observer.Body.Bytes())
		if result != test.expected {
			t.Errorf("Forbidden response body(%#v...) != %#v, got %#v", test.attributes, test.expected, result)
		}
		resultType := observer.HeaderMap.Get("Content-Type")
		if resultType != test.contentType {
			t.Errorf("Forbidden content type(%#v...) != %#v, got %#v", test.attributes, test.expected, result)
		}
	}
}
