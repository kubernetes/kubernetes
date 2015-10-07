/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/exec"
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
		{&api.HTTPGetAction{Host: "", Port: util.NewIntOrStringFromInt(-1), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetAction{Host: "", Port: util.NewIntOrStringFromString(""), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetAction{Host: "", Port: util.NewIntOrStringFromString("-1"), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetAction{Host: "", Port: util.NewIntOrStringFromString("not-found"), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetAction{Host: "", Port: util.NewIntOrStringFromString("found"), Path: ""}, true, "127.0.0.1", 93, ""},
		{&api.HTTPGetAction{Host: "", Port: util.NewIntOrStringFromInt(76), Path: ""}, true, "127.0.0.1", 76, ""},
		{&api.HTTPGetAction{Host: "", Port: util.NewIntOrStringFromString("118"), Path: ""}, true, "127.0.0.1", 118, ""},
		{&api.HTTPGetAction{Host: "hostname", Port: util.NewIntOrStringFromInt(76), Path: "path"}, true, "hostname", 76, "path"},
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
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromInt(-1)}, false, "", -1},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("")}, false, "", -1},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("-1")}, false, "", -1},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("not-found")}, false, "", -1},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("found")}, true, "1.2.3.4", 93},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromInt(76)}, true, "1.2.3.4", 76},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("118")}, true, "1.2.3.4", 118},
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

// TestProbeContainer tests the functionality of probeContainer.
// Test cases are:
//
// No probe.
// Only LivenessProbe.
// Only ReadinessProbe.
// Both probes.
//
// Also, for each probe, there will be several cases covering whether the initial
// delay has passed, whether the probe handler will return Success, Failure,
// Unknown or error.
//
// PLEASE READ THE PROBE DOCS BEFORE CHANGING THIS TEST IF YOU ARE UNSURE HOW PROBES ARE SUPPOSED TO WORK:
// (See https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/user-guide/pod-states.md#pod-conditions)
func TestProbeContainer(t *testing.T) {
	prober := &prober{
		refManager: kubecontainer.NewRefManager(),
		recorder:   &record.FakeRecorder{},
	}
	containerID := kubecontainer.ContainerID{"test", "foobar"}
	createdAt := time.Now().Unix()

	tests := []struct {
		testContainer     api.Container
		expectError       bool
		expectedLiveness  probe.Result
		expectedReadiness probe.Result
	}{
		// No probes.
		{
			testContainer:     api.Container{},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Success,
		},
		// Only LivenessProbe. expectedReadiness should always be true here.
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{InitialDelaySeconds: 100},
			},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Success,
		},
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{InitialDelaySeconds: -100},
			},
			expectedLiveness:  probe.Unknown,
			expectedReadiness: probe.Success,
		},
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
			expectedLiveness:  probe.Failure,
			expectedReadiness: probe.Success,
		},
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Success,
		},
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
			expectedLiveness:  probe.Unknown,
			expectedReadiness: probe.Success,
		},
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
			expectError:       true,
			expectedLiveness:  probe.Unknown,
			expectedReadiness: probe.Success,
		},
		// // Only ReadinessProbe. expectedLiveness should always be probe.Success here.
		{
			testContainer: api.Container{
				ReadinessProbe: &api.Probe{InitialDelaySeconds: 100},
			},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Unknown,
		},
		{
			testContainer: api.Container{
				ReadinessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Success,
		},
		{
			testContainer: api.Container{
				ReadinessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Success,
		},
		{
			testContainer: api.Container{
				ReadinessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Success,
		},
		{
			testContainer: api.Container{
				ReadinessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
			expectError:       false,
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Success,
		},
		// Both LivenessProbe and ReadinessProbe.
		{
			testContainer: api.Container{
				LivenessProbe:  &api.Probe{InitialDelaySeconds: 100},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: 100},
			},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Unknown,
		},
		{
			testContainer: api.Container{
				LivenessProbe:  &api.Probe{InitialDelaySeconds: 100},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: -100},
			},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Unknown,
		},
		{
			testContainer: api.Container{
				LivenessProbe:  &api.Probe{InitialDelaySeconds: -100},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: 100},
			},
			expectedLiveness:  probe.Unknown,
			expectedReadiness: probe.Unknown,
		},
		{
			testContainer: api.Container{
				LivenessProbe:  &api.Probe{InitialDelaySeconds: -100},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: -100},
			},
			expectedLiveness:  probe.Unknown,
			expectedReadiness: probe.Unknown,
		},
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: -100},
			},
			expectedLiveness:  probe.Unknown,
			expectedReadiness: probe.Unknown,
		},
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: -100},
			},
			expectedLiveness:  probe.Failure,
			expectedReadiness: probe.Unknown,
		},
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
				ReadinessProbe: &api.Probe{
					InitialDelaySeconds: -100,
					Handler: api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
			expectedLiveness:  probe.Success,
			expectedReadiness: probe.Success,
		},
	}

	for i, test := range tests {
		if test.expectError {
			prober.exec = fakeExecProber{test.expectedLiveness, errors.New("exec error")}
		} else {
			prober.exec = fakeExecProber{test.expectedLiveness, nil}
		}

		liveness, err := prober.ProbeLiveness(&api.Pod{}, api.PodStatus{}, test.testContainer, containerID, createdAt)
		if test.expectError && err == nil {
			t.Errorf("[%d] Expected liveness probe error but no error was returned.", i)
		}
		if !test.expectError && err != nil {
			t.Errorf("[%d] Didn't expect liveness probe error but got: %v", i, err)
		}
		if test.expectedLiveness != liveness {
			t.Errorf("[%d] Expected liveness result to be %v but was %v", i, test.expectedLiveness, liveness)
		}

		// TODO: Test readiness errors
		prober.exec = fakeExecProber{test.expectedReadiness, nil}
		readiness, err := prober.ProbeReadiness(&api.Pod{}, api.PodStatus{}, test.testContainer, containerID)
		if err != nil {
			t.Errorf("[%d] Unexpected readiness probe error: %v", i, err)
		}
		if test.expectedReadiness != readiness {
			t.Errorf("[%d] Expected readiness result to be %v but was %v", i, test.expectedReadiness, readiness)
		}
	}
}

type fakeExecProber struct {
	result probe.Result
	err    error
}

func (p fakeExecProber) Probe(_ exec.Cmd) (probe.Result, string, error) {
	return p.result, "", p.err
}
