/*
Copyright 2018 The Kubernetes Authors.

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

package runtime

import (
	"errors"
	"net"
	"os"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/cri-api/pkg/apis/runtime/v1"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

var errTest = errors.New("test")

func TestNewContainerRuntime(t *testing.T) {
	for _, tc := range []struct {
		name        string
		prepare     func(*fakeImpl)
		shouldError bool
	}{
		{
			name:        "valid",
			shouldError: false,
		},
		{
			name: "invalid: new runtime service fails",
			prepare: func(mock *fakeImpl) {
				mock.NewRemoteRuntimeServiceReturns(nil, errTest)
			},
			shouldError: true,
		},
		{
			name: "invalid: new image service fails",
			prepare: func(mock *fakeImpl) {
				mock.NewRemoteImageServiceReturns(nil, errTest)
			},
			shouldError: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			containerRuntime := NewContainerRuntime("")
			mock := &fakeImpl{}
			if tc.prepare != nil {
				tc.prepare(mock)
			}
			containerRuntime.SetImpl(mock)

			err := containerRuntime.Connect()

			assert.Equal(t, tc.shouldError, err != nil)
		})
	}
}

func TestIsRunning(t *testing.T) {
	for _, tc := range []struct {
		name        string
		prepare     func(*fakeImpl)
		shouldError bool
	}{
		{
			name:        "valid",
			shouldError: false,
		},
		{
			name: "invalid: runtime status fails",
			prepare: func(mock *fakeImpl) {
				mock.StatusReturns(nil, errTest)
			},
			shouldError: true,
		},
		{
			name: "invalid: runtime condition status not 'true'",
			prepare: func(mock *fakeImpl) {
				mock.StatusReturns(&v1.StatusResponse{Status: &v1.RuntimeStatus{
					Conditions: []*v1.RuntimeCondition{
						{
							Type:   v1.RuntimeReady,
							Status: false,
						},
					},
				},
				}, nil)
			},
			shouldError: true,
		},
		{
			name: "valid: runtime condition type does not match",
			prepare: func(mock *fakeImpl) {
				mock.StatusReturns(&v1.StatusResponse{Status: &v1.RuntimeStatus{
					Conditions: []*v1.RuntimeCondition{
						{
							Type:   v1.NetworkReady,
							Status: false,
						},
					},
				},
				}, nil)
			},
			shouldError: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			containerRuntime := NewContainerRuntime("")
			mock := &fakeImpl{}
			if tc.prepare != nil {
				tc.prepare(mock)
			}
			containerRuntime.SetImpl(mock)

			err := containerRuntime.IsRunning()

			assert.Equal(t, tc.shouldError, err != nil)
		})
	}
}

func TestListKubeContainers(t *testing.T) {
	for _, tc := range []struct {
		name        string
		expected    []string
		prepare     func(*fakeImpl)
		shouldError bool
	}{
		{
			name: "valid",
			prepare: func(mock *fakeImpl) {
				mock.ListPodSandboxReturns([]*v1.PodSandbox{
					{Id: "first"},
					{Id: "second"},
				}, nil)
			},
			expected:    []string{"first", "second"},
			shouldError: false,
		},
		{
			name: "invalid: list pod sandbox fails",
			prepare: func(mock *fakeImpl) {
				mock.ListPodSandboxReturns(nil, errTest)
			},
			shouldError: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			containerRuntime := NewContainerRuntime("")
			mock := &fakeImpl{}
			if tc.prepare != nil {
				tc.prepare(mock)
			}
			containerRuntime.SetImpl(mock)

			containers, err := containerRuntime.ListKubeContainers()

			assert.Equal(t, tc.shouldError, err != nil)
			assert.Equal(t, tc.expected, containers)
		})
	}
}

func TestSandboxImage(t *testing.T) {
	for _, tc := range []struct {
		name, expected string
		prepare        func(*fakeImpl)
		shouldError    bool
	}{
		{
			name: "valid",
			prepare: func(mock *fakeImpl) {
				mock.StatusReturns(&v1.StatusResponse{Info: map[string]string{
					"config": `{"sandboxImage": "pause"}`,
				}}, nil)
			},
			expected:    "pause",
			shouldError: false,
		},
		{
			name: "invalid: runtime status fails",
			prepare: func(mock *fakeImpl) {
				mock.StatusReturns(nil, errTest)
			},
			shouldError: true,
		},
		{
			name: "invalid: no config JSON",
			prepare: func(mock *fakeImpl) {
				mock.StatusReturns(&v1.StatusResponse{Info: map[string]string{
					"config": "wrong",
				}}, nil)
			},
			shouldError: true,
		},
		{
			name: "invalid: no config",
			prepare: func(mock *fakeImpl) {
				mock.StatusReturns(&v1.StatusResponse{Info: map[string]string{}}, nil)
			},
			shouldError: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			containerRuntime := NewContainerRuntime("")
			mock := &fakeImpl{}
			if tc.prepare != nil {
				tc.prepare(mock)
			}
			containerRuntime.SetImpl(mock)

			image, err := containerRuntime.SandboxImage()

			assert.Equal(t, tc.shouldError, err != nil)
			assert.Equal(t, tc.expected, image)
		})
	}
}

func TestRemoveContainers(t *testing.T) {
	for _, tc := range []struct {
		name        string
		containers  []string
		prepare     func(*fakeImpl)
		shouldError bool
	}{
		{
			name: "valid",
		},
		{
			name:        "valid: two containers",
			containers:  []string{"1", "2"},
			shouldError: false,
		},
		{
			name:       "invalid: remove pod sandbox fails",
			containers: []string{"1"},
			prepare: func(mock *fakeImpl) {
				mock.RemovePodSandboxReturns(errTest)
			},
			shouldError: true,
		},
		{
			name:       "invalid: stop pod sandbox fails",
			containers: []string{"1"},
			prepare: func(mock *fakeImpl) {
				mock.StopPodSandboxReturns(errTest)
			},
			shouldError: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			containerRuntime := NewContainerRuntime("")
			mock := &fakeImpl{}
			if tc.prepare != nil {
				tc.prepare(mock)
			}
			containerRuntime.SetImpl(mock)

			err := containerRuntime.RemoveContainers(tc.containers)

			assert.Equal(t, tc.shouldError, err != nil)
		})
	}
}

func TestPullImage(t *testing.T) {
	for _, tc := range []struct {
		name        string
		prepare     func(*fakeImpl)
		shouldError bool
	}{
		{
			name: "valid",
		},
		{
			name: "invalid: pull image fails",
			prepare: func(mock *fakeImpl) {
				mock.PullImageReturns("", errTest)
			},
			shouldError: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			containerRuntime := NewContainerRuntime("")
			mock := &fakeImpl{}
			if tc.prepare != nil {
				tc.prepare(mock)
			}
			containerRuntime.SetImpl(mock)

			err := containerRuntime.PullImage("")

			assert.Equal(t, tc.shouldError, err != nil)
		})
	}
}

func TestImageExists(t *testing.T) {
	for _, tc := range []struct {
		name     string
		expected bool
		prepare  func(*fakeImpl)
	}{
		{
			name: "valid",
			prepare: func(mock *fakeImpl) {
				mock.ImageStatusReturns(&v1.ImageStatusResponse{
					Image: &v1.Image{},
				}, nil)
			},
			expected: true,
		},
		{
			name: "invalid: image status fails",
			prepare: func(mock *fakeImpl) {
				mock.ImageStatusReturns(nil, errTest)
			},
			expected: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			containerRuntime := NewContainerRuntime("")
			mock := &fakeImpl{}
			if tc.prepare != nil {
				tc.prepare(mock)
			}
			containerRuntime.SetImpl(mock)

			exists := containerRuntime.ImageExists("")

			assert.Equal(t, tc.expected, exists)
		})
	}
}

func TestIsExistingSocket(t *testing.T) {
	// this test is not expected to work on Windows
	if runtime.GOOS == "windows" {
		return
	}

	const tempPrefix = "test.kubeadm.runtime.isExistingSocket."
	tests := []struct {
		name string
		proc func(*testing.T)
	}{
		{
			name: "Valid domain socket is detected as such",
			proc: func(t *testing.T) {
				tmpFile, err := os.CreateTemp("", tempPrefix)
				if err != nil {
					t.Fatalf("unexpected error by TempFile: %v", err)
				}
				theSocket := tmpFile.Name()
				os.Remove(theSocket)
				tmpFile.Close()

				con, err := net.Listen("unix", theSocket)
				if err != nil {
					t.Fatalf("unexpected error while dialing a socket: %v", err)
				}
				defer con.Close()

				if !isExistingSocket("unix://" + theSocket) {
					t.Fatalf("isExistingSocket(%q) gave unexpected result. Should have been true, instead of false", theSocket)
				}
			},
		},
		{
			name: "Regular file is not a domain socket",
			proc: func(t *testing.T) {
				tmpFile, err := os.CreateTemp("", tempPrefix)
				if err != nil {
					t.Fatalf("unexpected error by TempFile: %v", err)
				}
				theSocket := tmpFile.Name()
				defer os.Remove(theSocket)
				tmpFile.Close()

				if isExistingSocket(theSocket) {
					t.Fatalf("isExistingSocket(%q) gave unexpected result. Should have been false, instead of true", theSocket)
				}
			},
		},
		{
			name: "Non existent socket is not a domain socket",
			proc: func(t *testing.T) {
				const theSocket = "/non/existent/socket"
				if isExistingSocket(theSocket) {
					t.Fatalf("isExistingSocket(%q) gave unexpected result. Should have been false, instead of true", theSocket)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, test.proc)
	}
}

func TestDetectCRISocketImpl(t *testing.T) {
	tests := []struct {
		name            string
		existingSockets []string
		expectedError   bool
		expectedSocket  string
	}{
		{
			name:            "No existing sockets, use default",
			existingSockets: []string{},
			expectedError:   false,
			expectedSocket:  constants.DefaultCRISocket,
		},
		{
			name:            "One valid CRI socket leads to success",
			existingSockets: []string{"unix:///foo/bar.sock"},
			expectedError:   false,
			expectedSocket:  "unix:///foo/bar.sock",
		},
		{
			name: "Multiple CRI sockets lead to an error",
			existingSockets: []string{
				"unix:///foo/bar.sock",
				"unix:///foo/baz.sock",
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			socket, err := detectCRISocketImpl(func(path string) bool {
				for _, existing := range test.existingSockets {
					if path == existing {
						return true
					}
				}
				return false
			}, test.existingSockets)

			if (err != nil) != test.expectedError {
				t.Fatalf("detectCRISocketImpl returned unexpected result\n\tExpected error: %t\n\tGot error: %t", test.expectedError, err != nil)
			}
			if !test.expectedError && socket != test.expectedSocket {
				t.Fatalf("detectCRISocketImpl returned unexpected CRI socket\n\tExpected socket: %s\n\tReturned socket: %s",
					test.expectedSocket, socket)
			}
		})
	}
}

func TestPullImagesInParallel(t *testing.T) {
	for _, tc := range []struct {
		name         string
		ifNotPresent bool
		prepare      func(*fakeImpl)
		shouldError  bool
	}{
		{
			name: "valid",
		},
		{
			name:         "valid: ifNotPresent is true",
			ifNotPresent: true,
		},
		{
			name: "invalid: pull fails",
			prepare: func(mock *fakeImpl) {
				mock.PullImageReturns("", errTest)
			},
			shouldError: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			containerRuntime := NewContainerRuntime("")
			mock := &fakeImpl{}
			if tc.prepare != nil {
				tc.prepare(mock)
			}
			containerRuntime.SetImpl(mock)

			err := containerRuntime.PullImagesInParallel([]string{"first", "second"}, tc.ifNotPresent)

			assert.Equal(t, tc.shouldError, err != nil)
		})
	}
}
