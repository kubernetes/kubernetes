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

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
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
		expected   string
		attributes authorizer.Attributes
		reason     string
	}{
		{`User "NAME" cannot GET path "/whatever".`,
			authorizer.AttributesRecord{User: u, Verb: "GET", Path: "/whatever"}, ""},
		{`User "NAME" cannot GET path "/&lt;script&gt;".`,
			authorizer.AttributesRecord{User: u, Verb: "GET", Path: "/<script>"}, ""},
		{`User "NAME" cannot GET pod at the cluster scope.`,
			authorizer.AttributesRecord{User: u, Verb: "GET", Resource: "pod", ResourceRequest: true}, ""},
		{`User "NAME" cannot GET pod.v2/quota in the namespace "test".`,
			authorizer.AttributesRecord{User: u, Verb: "GET", Namespace: "test", APIGroup: "v2", Resource: "pod", Subresource: "quota", ResourceRequest: true}, ""},
	}
	for _, test := range cases {
		observer := httptest.NewRecorder()
		Forbidden(test.attributes, observer, &http.Request{}, test.reason)
		result := string(observer.Body.Bytes())
		if result != test.expected {
			t.Errorf("Forbidden(%#v...) != %#v, got %#v", test.attributes, test.expected, result)
		}
	}
}
