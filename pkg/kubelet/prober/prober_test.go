/*
Copyright 2015 The Kubernetes Authors.

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

package prober

import (
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestFormatURL(t *testing.T) {
	testCases := []struct {
		scheme string
		host   string
		port   int
		path   string
		result string
	}{
		{"http", "localhost", 93, "", "http://localhost:93"},
		{"https", "localhost", 93, "/path", "https://localhost:93/path"},
		{"http", "localhost", 93, "?foo", "http://localhost:93?foo"},
		{"https", "localhost", 93, "/path?bar", "https://localhost:93/path?bar"},
	}
	for _, test := range testCases {
		url := formatURL(test.scheme, test.host, test.port, test.path)
		if url.String() != test.result {
			t.Errorf("Expected %s, got %s", test.result, url.String())
		}
	}
}

func TestFindPortByName(t *testing.T) {
	container := api.Container{
		Ports: []api.ContainerPort{
			{
				Name:          "foo",
				ContainerPort: 8080,
			},
			{
				Name:          "bar",
				ContainerPort: 9000,
			},
		},
	}
	want := 8080
	got, err := findPortByName(container, "foo")
	if got != want || err != nil {
		t.Errorf("Expected %v, got %v, err: %v", want, got, err)
	}
}

func TestGetURLParts(t *testing.T) {
	testCases := []struct {
		probe *api.HTTPGetAction
		ok    bool
		host  string
		port  int
		path  string
	}{
		{&api.HTTPGetAction{Host: "", Port: intstr.FromInt(-1), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetAction{Host: "", Port: intstr.FromString(""), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetAction{Host: "", Port: intstr.FromString("-1"), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetAction{Host: "", Port: intstr.FromString("not-found"), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetAction{Host: "", Port: intstr.FromString("found"), Path: ""}, true, "127.0.0.1", 93, ""},
		{&api.HTTPGetAction{Host: "", Port: intstr.FromInt(76), Path: ""}, true, "127.0.0.1", 76, ""},
		{&api.HTTPGetAction{Host: "", Port: intstr.FromString("118"), Path: ""}, true, "127.0.0.1", 118, ""},
		{&api.HTTPGetAction{Host: "hostname", Port: intstr.FromInt(76), Path: "path"}, true, "hostname", 76, "path"},
	}

	for _, test := range testCases {
		state := api.PodStatus{PodIP: "127.0.0.1"}
		container := api.Container{
			Ports: []api.ContainerPort{{Name: "found", ContainerPort: 93}},
			LivenessProbe: &api.Probe{
				Handler: api.Handler{
					HTTPGet: test.probe,
				},
			},
		}

		scheme := test.probe.Scheme
		if scheme == "" {
			scheme = api.URISchemeHTTP
		}
		host := test.probe.Host
		if host == "" {
			host = state.PodIP
		}
		port, err := extractPort(test.probe.Port, container)
		if test.ok && err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		path := test.probe.Path

		if !test.ok && err == nil {
			t.Errorf("Expected error for %+v, got %s%s:%d/%s", test, scheme, host, port, path)
		}
		if test.ok {
			if host != test.host || port != test.port || path != test.path {
				t.Errorf("Expected %s:%d/%s, got %s:%d/%s",
					test.host, test.port, test.path, host, port, path)
			}
		}
	}
}

func TestGetTCPAddrParts(t *testing.T) {
	testCases := []struct {
		probe *api.TCPSocketAction
		ok    bool
		host  string
		port  int
	}{
		{&api.TCPSocketAction{Port: intstr.FromInt(-1)}, false, "", -1},
		{&api.TCPSocketAction{Port: intstr.FromString("")}, false, "", -1},
		{&api.TCPSocketAction{Port: intstr.FromString("-1")}, false, "", -1},
		{&api.TCPSocketAction{Port: intstr.FromString("not-found")}, false, "", -1},
		{&api.TCPSocketAction{Port: intstr.FromString("found")}, true, "1.2.3.4", 93},
		{&api.TCPSocketAction{Port: intstr.FromInt(76)}, true, "1.2.3.4", 76},
		{&api.TCPSocketAction{Port: intstr.FromString("118")}, true, "1.2.3.4", 118},
	}

	for _, test := range testCases {
		host := "1.2.3.4"
		container := api.Container{
			Ports: []api.ContainerPort{{Name: "found", ContainerPort: 93}},
			LivenessProbe: &api.Probe{
				Handler: api.Handler{
					TCPSocket: test.probe,
				},
			},
		}
		port, err := extractPort(test.probe.Port, container)
		if !test.ok && err == nil {
			t.Errorf("Expected error for %+v, got %s:%d", test, host, port)
		}
		if test.ok && err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if test.ok {
			if host != test.host || port != test.port {
				t.Errorf("Expected %s:%d, got %s:%d", test.host, test.port, host, port)
			}
		}
	}
}

func TestHTTPHeaders(t *testing.T) {
	testCases := []struct {
		input  []api.HTTPHeader
		output http.Header
	}{
		{[]api.HTTPHeader{}, http.Header{}},
		{[]api.HTTPHeader{
			{Name: "X-Muffins-Or-Cupcakes", Value: "Muffins"},
		}, http.Header{"X-Muffins-Or-Cupcakes": {"Muffins"}}},
		{[]api.HTTPHeader{
			{Name: "X-Muffins-Or-Cupcakes", Value: "Muffins"},
			{Name: "X-Muffins-Or-Plumcakes", Value: "Muffins!"},
		}, http.Header{"X-Muffins-Or-Cupcakes": {"Muffins"},
			"X-Muffins-Or-Plumcakes": {"Muffins!"}}},
		{[]api.HTTPHeader{
			{Name: "X-Muffins-Or-Cupcakes", Value: "Muffins"},
			{Name: "X-Muffins-Or-Cupcakes", Value: "Cupcakes, too"},
		}, http.Header{"X-Muffins-Or-Cupcakes": {"Muffins", "Cupcakes, too"}}},
	}
	for _, test := range testCases {
		headers := buildHeader(test.input)
		if !reflect.DeepEqual(test.output, headers) {
			t.Errorf("Expected %#v, got %#v", test.output, headers)
		}
	}
}

func TestProbe(t *testing.T) {
	prober := &prober{
		refManager: kubecontainer.NewRefManager(),
		recorder:   &record.FakeRecorder{},
	}
	containerID := kubecontainer.ContainerID{Type: "test", ID: "foobar"}

	execProbe := &api.Probe{
		Handler: api.Handler{
			Exec: &api.ExecAction{},
		},
	}
	tests := []struct {
		probe          *api.Probe
		execError      bool
		expectError    bool
		execResult     probe.Result
		expectedResult results.Result
	}{
		{ // No probe
			probe:          nil,
			expectedResult: results.Success,
		},
		{ // No handler
			probe:          &api.Probe{},
			expectError:    true,
			expectedResult: results.Failure,
		},
		{ // Probe fails
			probe:          execProbe,
			execResult:     probe.Failure,
			expectedResult: results.Failure,
		},
		{ // Probe succeeds
			probe:          execProbe,
			execResult:     probe.Success,
			expectedResult: results.Success,
		},
		{ // Probe result is unknown
			probe:          execProbe,
			execResult:     probe.Unknown,
			expectedResult: results.Failure,
		},
		{ // Probe has an error
			probe:          execProbe,
			execError:      true,
			expectError:    true,
			execResult:     probe.Unknown,
			expectedResult: results.Failure,
		},
	}

	for i, test := range tests {
		for _, probeType := range [...]probeType{liveness, readiness} {
			testID := fmt.Sprintf("%d-%s", i, probeType)
			testContainer := api.Container{}
			switch probeType {
			case liveness:
				testContainer.LivenessProbe = test.probe
			case readiness:
				testContainer.ReadinessProbe = test.probe
			}
			if test.execError {
				prober.exec = fakeExecProber{test.execResult, errors.New("exec error")}
			} else {
				prober.exec = fakeExecProber{test.execResult, nil}
			}

			result, err := prober.probe(probeType, &api.Pod{}, api.PodStatus{}, testContainer, containerID)
			if test.expectError && err == nil {
				t.Errorf("[%s] Expected probe error but no error was returned.", testID)
			}
			if !test.expectError && err != nil {
				t.Errorf("[%s] Didn't expect probe error but got: %v", testID, err)
			}
			if test.expectedResult != result {
				t.Errorf("[%s] Expected result to be %v but was %v", testID, test.expectedResult, result)
			}
		}
	}
}
