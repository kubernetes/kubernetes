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
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestSetKubernetesDefaults(t *testing.T) {
	testCases := []struct {
		Config restclient.Config
		After  restclient.Config
		Err    bool
	}{
		{
			restclient.Config{},
			restclient.Config{
				APIPath: "/api",
				ContentConfig: restclient.ContentConfig{
					GroupVersion:         testapi.Default.GroupVersion(),
					Codec:                testapi.Default.Codec(),
					NegotiatedSerializer: testapi.Default.NegotiatedSerializer(),
				},
				QPS:   5,
				Burst: 10,
			},
			false,
		},
		// Add this test back when we fixed config and SetKubernetesDefaults
		// {
		// 	restclient.Config{
		// 		GroupVersion: &unversioned.GroupVersion{Group: "not.a.group", Version: "not_an_api"},
		// 	},
		// 	restclient.Config{},
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
	defer server.Close()
	got, err := restclient.ServerAPIVersions(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Group: "invalid version", Version: "one"}, NegotiatedSerializer: testapi.Default.NegotiatedSerializer()}})
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
		conf := &restclient.Config{
			Host: "127.0.0.1",
			ContentConfig: restclient.ContentConfig{
				GroupVersion: &unversioned.GroupVersion{Version: version},
			},
		}

		var versionedPath string
		err := SetKubernetesDefaults(conf)
		if err == nil {
			_, versionedPath, err = restclient.DefaultServerURL(conf.Host, conf.APIPath, *conf.GroupVersion, false)
		}

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
		if e, a := expected.Prefix, versionedPath; e != a {
			t.Errorf("expected %#v, got %#v", e, a)
		}
		if e, a := expected.Codec, conf.Codec; !reflect.DeepEqual(e, a) {
			t.Errorf("expected %#v, got %#v", e, a)
		}
	}
}
