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

package unversioned

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestIsConfigTransportTLS(t *testing.T) {
	testCases := []struct {
		Config       *Config
		TransportTLS bool
	}{
		{
			Config:       &Config{},
			TransportTLS: false,
		},
		{
			Config: &Config{
				Host: "https://localhost",
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host: "localhost",
				TLSClientConfig: TLSClientConfig{
					CertFile: "foo",
				},
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host: "///:://localhost",
				TLSClientConfig: TLSClientConfig{
					CertFile: "foo",
				},
			},
			TransportTLS: false,
		},
		{
			Config: &Config{
				Host:     "1.2.3.4:567",
				Insecure: true,
			},
			TransportTLS: true,
		},
	}
	for _, testCase := range testCases {
		if err := SetKubernetesDefaults(testCase.Config); err != nil {
			t.Errorf("setting defaults failed for %#v: %v", testCase.Config, err)
			continue
		}
		useTLS := IsConfigTransportTLS(*testCase.Config)
		if testCase.TransportTLS != useTLS {
			t.Errorf("expected %v for %#v", testCase.TransportTLS, testCase.Config)
		}
	}
}

func TestSetKubernetesDefaults(t *testing.T) {
	testCases := []struct {
		Config Config
		After  Config
		Err    bool
	}{
		{
			Config{},
			Config{
				APIPath:      "/api",
				GroupVersion: testapi.Default.GroupVersion(),
				Codec:        testapi.Default.Codec(),
				QPS:          5,
				Burst:        10,
			},
			false,
		},
		// Add this test back when we fixed config and SetKubernetesDefaults
		// {
		// 	Config{
		// 		GroupVersion: &unversioned.GroupVersion{Group: "not.a.group", Version: "not_an_api"},
		// 	},
		// 	Config{},
		// 	true,
		// },
	}
	for _, testCase := range testCases {
		val := &testCase.Config
		err := SetKubernetesDefaults(val)
		val.UserAgent = ""
		switch {
		case err == nil && testCase.Err:
			t.Errorf("expected error but was nil")
			continue
		case err != nil && !testCase.Err:
			t.Errorf("unexpected error %v", err)
			continue
		case err != nil:
			continue
		}
		if !reflect.DeepEqual(*val, testCase.After) {
			t.Errorf("unexpected result object: %#v", val)
		}
	}
}

func TestSetKubernetesDefaultsUserAgent(t *testing.T) {
	config := &Config{}
	if err := SetKubernetesDefaults(config); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(config.UserAgent, "kubernetes/") {
		t.Errorf("no user agent set: %#v", config)
	}
}

func TestHelperGetServerAPIVersions(t *testing.T) {
	expect := []string{"v1", "v2", "v3"}
	APIVersions := unversioned.APIVersions{Versions: expect}
	expect = append(expect, "group1/v1", "group1/v2", "group2/v1", "group2/v2")
	APIGroupList := unversioned.APIGroupList{
		Groups: []unversioned.APIGroup{
			{
				Versions: []unversioned.GroupVersionForDiscovery{
					{
						GroupVersion: "group1/v1",
					},
					{
						GroupVersion: "group1/v2",
					},
				},
			},
			{
				Versions: []unversioned.GroupVersionForDiscovery{
					{
						GroupVersion: "group2/v1",
					},
					{
						GroupVersion: "group2/v2",
					},
				},
			},
		},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var output []byte
		var err error
		switch req.URL.Path {
		case "/api":
			output, err = json.Marshal(APIVersions)

		case "/apis":
			output, err = json.Marshal(APIGroupList)
		}
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	// TODO: Uncomment when fix #19254
	// defer server.Close()
	got, err := ServerAPIVersions(&Config{Host: server.URL, GroupVersion: &unversioned.GroupVersion{Group: "invalid version", Version: "one"}, Codec: testapi.Default.Codec()})
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestSetsCodec(t *testing.T) {
	testCases := map[string]struct {
		Err    bool
		Prefix string
		Codec  runtime.Codec
	}{
		testapi.Default.GroupVersion().Version: {false, "/api/" + testapi.Default.GroupVersion().Version, testapi.Default.Codec()},
		// Add this test back when we fixed config and SetKubernetesDefaults
		// "invalidVersion":                       {true, "", nil},
	}
	for version, expected := range testCases {
		client, err := New(&Config{Host: "127.0.0.1", GroupVersion: &unversioned.GroupVersion{Version: version}})
		switch {
		case err == nil && expected.Err:
			t.Errorf("expected error but was nil")
			continue
		case err != nil && !expected.Err:
			t.Errorf("unexpected error %v", err)
			continue
		case err != nil:
			continue
		}
		if e, a := expected.Prefix, client.RESTClient.versionedAPIPath; e != a {
			t.Errorf("expected %#v, got %#v", e, a)
		}
		if e, a := expected.Codec, client.RESTClient.Codec; e != a {
			t.Errorf("expected %#v, got %#v", e, a)
		}
	}
}

func TestRESTClientRequires(t *testing.T) {
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", Codec: testapi.Default.Codec()}); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", GroupVersion: testapi.Default.GroupVersion()}); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", GroupVersion: testapi.Default.GroupVersion(), Codec: testapi.Default.Codec()}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidatesHostParameter(t *testing.T) {
	testCases := []struct {
		Host    string
		APIPath string

		URL string
		Err bool
	}{
		{"127.0.0.1", "", "http://127.0.0.1/" + testapi.Default.GroupVersion().Version, false},
		{"127.0.0.1:8080", "", "http://127.0.0.1:8080/" + testapi.Default.GroupVersion().Version, false},
		{"foo.bar.com", "", "http://foo.bar.com/" + testapi.Default.GroupVersion().Version, false},
		{"http://host/prefix", "", "http://host/prefix/" + testapi.Default.GroupVersion().Version, false},
		{"http://host", "", "http://host/" + testapi.Default.GroupVersion().Version, false},
		{"http://host", "/", "http://host/" + testapi.Default.GroupVersion().Version, false},
		{"http://host", "/other", "http://host/other/" + testapi.Default.GroupVersion().Version, false},
		{"host/server", "", "", true},
	}
	for i, testCase := range testCases {
		u, versionedAPIPath, err := DefaultServerURL(testCase.Host, testCase.APIPath, *testapi.Default.GroupVersion(), false)
		switch {
		case err == nil && testCase.Err:
			t.Errorf("expected error but was nil")
			continue
		case err != nil && !testCase.Err:
			t.Errorf("unexpected error %v", err)
			continue
		case err != nil:
			continue
		}
		u.Path = path.Join(u.Path, versionedAPIPath)
		if e, a := testCase.URL, u.String(); e != a {
			t.Errorf("%d: expected host %s, got %s", i, e, a)
			continue
		}
	}
}
