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

package cmd

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"testing"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

type fakeRemoteExecutor struct {
	req     *client.Request
	execErr error
}

func (f *fakeRemoteExecutor) Execute(req *client.Request, config *client.Config, command []string, stdin io.Reader, stdout, stderr io.Writer, tty bool) error {
	f.req = req
	return f.execErr
}

func TestPodAndContainer(t *testing.T) {
	tests := []struct {
		args              []string
		p                 *execParams
		name              string
		expectError       bool
		expectedPod       string
		expectedContainer string
		expectedArgs      []string
	}{
		{
			p:           &execParams{},
			expectError: true,
			name:        "empty",
		},
		{
			p:           &execParams{podName: "foo"},
			expectError: true,
			name:        "no cmd",
		},
		{
			p:           &execParams{podName: "foo", containerName: "bar"},
			expectError: true,
			name:        "no cmd, w/ container",
		},
		{
			p:            &execParams{podName: "foo"},
			args:         []string{"cmd"},
			expectedPod:  "foo",
			expectedArgs: []string{"cmd"},
			name:         "pod in flags",
		},
		{
			p:           &execParams{},
			args:        []string{"foo"},
			expectError: true,
			name:        "no cmd, w/o flags",
		},
		{
			p:            &execParams{},
			args:         []string{"foo", "cmd"},
			expectedPod:  "foo",
			expectedArgs: []string{"cmd"},
			name:         "cmd, w/o flags",
		},
		{
			p:                 &execParams{containerName: "bar"},
			args:              []string{"foo", "cmd"},
			expectedPod:       "foo",
			expectedContainer: "bar",
			expectedArgs:      []string{"cmd"},
			name:              "cmd, container in flag",
		},
	}
	for _, test := range tests {
		cmd := &cobra.Command{}
		podName, containerName, args, err := extractPodAndContainer(cmd, test.args, test.p)
		if podName != test.expectedPod {
			t.Errorf("expected: %s, got: %s (%s)", test.expectedPod, podName, test.name)
		}
		if containerName != test.expectedContainer {
			t.Errorf("expected: %s, got: %s (%s)", test.expectedContainer, containerName, test.name)
		}
		if test.expectError && err == nil {
			t.Errorf("unexpected non-error (%s)", test.name)
		}
		if !test.expectError && err != nil {
			t.Errorf("unexpected error: %v (%s)", err, test.name)
		}
		if !reflect.DeepEqual(test.expectedArgs, args) {
			t.Errorf("expected: %v, got %v (%s)", test.expectedArgs, args, test.name)
		}
	}
}

func TestExec(t *testing.T) {
	tests := []struct {
		name, version, podPath, execPath, container string
		pod                                         *api.Pod
		execErr                                     bool
	}{
		{
			name:     "v1beta3 - pod exec",
			version:  "v1beta3",
			podPath:  "/api/v1beta3/namespaces/test/pods/foo",
			execPath: "/api/v1beta3/namespaces/test/pods/foo/exec",
			pod:      execPod(),
		},
		{
			name:     "v1beta3 - pod exec error",
			version:  "v1beta3",
			podPath:  "/api/v1beta3/namespaces/test/pods/foo",
			execPath: "/api/v1beta3/namespaces/test/pods/foo/exec",
			pod:      execPod(),
			execErr:  true,
		},
		{
			name:     "v1 - pod exec",
			version:  "v1",
			podPath:  "/api/v1/namespaces/test/pods/foo",
			execPath: "/api/v1/namespaces/test/pods/foo/exec",
			pod:      execPod(),
		},
		{
			name:     "v1 - pod exec error",
			version:  "v1",
			podPath:  "/api/v1/namespaces/test/pods/foo",
			execPath: "/api/v1/namespaces/test/pods/foo/exec",
			pod:      execPod(),
			execErr:  true,
		},
	}
	for _, test := range tests {
		f, tf, codec := NewAPIFactory()
		tf.Client = &client.FakeRESTClient{
			Codec: codec,
			Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == test.podPath && m == "GET":
					body := objBody(codec, test.pod)
					return &http.Response{StatusCode: 200, Body: body}, nil
				default:
					// Ensures no GET is performed when deleting by name
					t.Errorf("%s: unexpected request: %#v\n%#v", test.name, req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = &client.Config{Version: test.version}
		bufOut := bytes.NewBuffer([]byte{})
		bufErr := bytes.NewBuffer([]byte{})
		bufIn := bytes.NewBuffer([]byte{})
		ex := &fakeRemoteExecutor{}
		if test.execErr {
			ex.execErr = fmt.Errorf("exec error")
		}
		params := &execParams{
			podName:       "foo",
			containerName: "bar",
		}
		cmd := &cobra.Command{}
		err := RunExec(f, cmd, bufIn, bufOut, bufErr, params, []string{"test", "command"}, ex)
		if test.execErr && err != ex.execErr {
			t.Errorf("%s: Unexpected exec error: %v", test.name, err)
		}
		if !test.execErr && ex.req.URL().Path != test.execPath {
			t.Errorf("%s: Did not get expected path for exec request", test.name)
		}
		if !test.execErr && err != nil {
			t.Errorf("%s: Unexpected error: %v", test.name, err)
		}
	}
}

func execPod() *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
			Containers: []api.Container{
				{
					Name: "bar",
				},
			},
		},
		Status: api.PodStatus{
			Phase: api.PodRunning,
		},
	}
}
