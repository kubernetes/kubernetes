/*
Copyright 2015 Google Inc. All rights reserved.

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
	"errors"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"

	"github.com/fsouza/go-dockerclient"
)

func TestFindPortByName(t *testing.T) {
	container := api.Container{
		Ports: []api.ContainerPort{
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
	want := 8080
	got := findPortByName(container, "foo")
	if got != want {
		t.Errorf("Expected %v, got %v", want, got)
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
			Ports: []api.ContainerPort{{Name: "found", HostPort: 93}},
			LivenessProbe: &api.Probe{
				Handler: api.Handler{
					HTTPGet: test.probe,
				},
			},
		}
		p, err := extractPort(test.probe.Port, container)
		if test.ok && err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		host, port, path := extractGetParams(test.probe, state, p)
		if !test.ok && err == nil {
			t.Errorf("Expected error for %+v, got %s:%d/%s", test, host, port, path)
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
			Ports: []api.ContainerPort{{Name: "found", HostPort: 93}},
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

type fakeExecProber struct {
	result probe.Result
	err    error
}

func (p fakeExecProber) Probe(_ exec.Cmd) (probe.Result, error) {
	return p.result, p.err
}

func makeTestKubelet(result probe.Result, err error) *Kubelet {
	kl := &Kubelet{
		readinessManager:    kubecontainer.NewReadinessManager(),
		containerRefManager: kubecontainer.NewRefManager(),
	}

	kl.prober = &prober{
		exec: fakeExecProber{
			result: result,
			err:    err,
		},
		readinessManager: kl.readinessManager,
		refManager:       kl.containerRefManager,
		recorder:         &record.FakeRecorder{},
	}
	return kl
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
func TestProbeContainer(t *testing.T) {
	dc := &docker.APIContainers{
		ID:      "foobar",
		Created: time.Now().Unix(),
	}
	tests := []struct {
		testContainer     api.Container
		expectError       bool
		expectedResult    probe.Result
		expectedReadiness bool
	}{
		// No probes.
		{
			testContainer:     api.Container{},
			expectedResult:    probe.Success,
			expectedReadiness: true,
		},
		// Only LivenessProbe.
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{InitialDelaySeconds: 100},
			},
			expectedResult:    probe.Success,
			expectedReadiness: true,
		},
		{
			testContainer: api.Container{
				LivenessProbe: &api.Probe{InitialDelaySeconds: -100},
			},
			expectedResult:    probe.Unknown,
			expectedReadiness: false,
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
			expectedResult:    probe.Failure,
			expectedReadiness: false,
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
			expectedResult:    probe.Success,
			expectedReadiness: true,
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
			expectedResult:    probe.Unknown,
			expectedReadiness: false,
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
			expectedResult:    probe.Unknown,
			expectedReadiness: false,
		},
		// Only ReadinessProbe.
		{
			testContainer: api.Container{
				ReadinessProbe: &api.Probe{InitialDelaySeconds: 100},
			},
			expectedResult:    probe.Success,
			expectedReadiness: false,
		},
		{
			testContainer: api.Container{
				ReadinessProbe: &api.Probe{InitialDelaySeconds: -100},
			},
			expectedResult:    probe.Success,
			expectedReadiness: false,
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
			expectedResult:    probe.Success,
			expectedReadiness: true,
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
			expectedResult:    probe.Success,
			expectedReadiness: true,
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
			expectedResult:    probe.Success,
			expectedReadiness: true,
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
			expectedResult:    probe.Success,
			expectedReadiness: true,
		},
		// Both LivenessProbe and ReadinessProbe.
		{
			testContainer: api.Container{
				LivenessProbe:  &api.Probe{InitialDelaySeconds: 100},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: 100},
			},
			expectedResult:    probe.Success,
			expectedReadiness: false,
		},
		{
			testContainer: api.Container{
				LivenessProbe:  &api.Probe{InitialDelaySeconds: 100},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: -100},
			},
			expectedResult:    probe.Success,
			expectedReadiness: false,
		},
		{
			testContainer: api.Container{
				LivenessProbe:  &api.Probe{InitialDelaySeconds: -100},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: 100},
			},
			expectedResult:    probe.Unknown,
			expectedReadiness: false,
		},
		{
			testContainer: api.Container{
				LivenessProbe:  &api.Probe{InitialDelaySeconds: -100},
				ReadinessProbe: &api.Probe{InitialDelaySeconds: -100},
			},
			expectedResult:    probe.Unknown,
			expectedReadiness: false,
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
			expectedResult:    probe.Unknown,
			expectedReadiness: false,
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
			expectedResult:    probe.Failure,
			expectedReadiness: false,
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
			expectedResult:    probe.Success,
			expectedReadiness: true,
		},
	}

	for _, test := range tests {
		var kl *Kubelet

		if test.expectError {
			kl = makeTestKubelet(test.expectedResult, errors.New("error"))
		} else {
			kl = makeTestKubelet(test.expectedResult, nil)
		}
		result, err := kl.prober.Probe(&api.Pod{}, api.PodStatus{}, test.testContainer, dc.ID, dc.Created)
		if test.expectError && err == nil {
			t.Error("Expected error but did no error was returned.")
		}
		if !test.expectError && err != nil {
			t.Errorf("Expected error but got: %v", err)
		}
		if test.expectedResult != result {
			t.Errorf("Expected result was %v but probeContainer() returned %v", test.expectedResult, result)
		}
		if test.expectedReadiness != kl.readinessManager.GetReadiness(dc.ID) {
			t.Errorf("Expected readiness was %v but probeContainer() set %v", test.expectedReadiness, kl.readinessManager.GetReadiness(dc.ID))
		}
	}
}
