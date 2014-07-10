/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubelet

import (
	"net/http"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type fakeHTTPClient struct {
	req string
	res http.Response
	err error
}

func (f *fakeHTTPClient) Get(url string) (*http.Response, error) {
	f.req = url
	return &f.res, f.err
}

func TestHttpHealth(t *testing.T) {
	fakeClient := fakeHTTPClient{
		res: http.Response{
			StatusCode: http.StatusOK,
		},
	}

	check := HTTPHealthChecker{
		client: &fakeClient,
	}

	container := api.Container{
		LivenessProbe: api.LivenessProbe{
			HTTPGet: api.HTTPGetProbe{
				Port: "8080",
				Path: "/foo/bar",
			},
			Type: "http",
		},
	}

	ok, err := check.IsHealthy(container)
	if !ok {
		t.Error("Unexpected unhealthy")
	}
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if fakeClient.req != "http://localhost:8080/foo/bar" {
		t.Errorf("Unexpected url: %s", fakeClient.req)
	}
}

func TestFindPort(t *testing.T) {
	container := api.Container{
		Ports: []api.Port{
			{
				Name:     "foo",
				HostPort: 8080,
			},
			{
				Name:     "bar",
				HostPort: 9000,
			},
		},
	}
	check := HTTPHealthChecker{}
	validatePort(t, check.findPort(container, "foo"), 8080)
}

func validatePort(t *testing.T, port int64, expectedPort int64) {
	if port != expectedPort {
		t.Errorf("Unexpected port: %d, expected: %d", port, expectedPort)
	}
}
