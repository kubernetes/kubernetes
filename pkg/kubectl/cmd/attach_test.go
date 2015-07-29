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
	"testing"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

type fakeRemoteAttach struct {
	req       *client.Request
	attachErr error
}

func (f *fakeRemoteAttach) Attach(req *client.Request, config *client.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool) error {
	f.req = req
	return f.attachErr
}

func TestPodAndContainerAttach(t *testing.T) {
	tests := []struct {
		args              []string
		p                 *AttachOptions
		name              string
		expectError       bool
		expectedPod       string
		expectedContainer string
	}{
		{
			p:           &AttachOptions{},
			expectError: true,
			name:        "empty",
		},
		{
			p:           &AttachOptions{},
			args:        []string{"foo", "bar"},
			expectError: true,
			name:        "too many args",
		},
		{
			p:           &AttachOptions{},
			args:        []string{"foo"},
			expectedPod: "foo",
			name:        "no container, no flags",
		},
		{
			p:                 &AttachOptions{ContainerName: "bar"},
			args:              []string{"foo"},
			expectedPod:       "foo",
			expectedContainer: "bar",
			name:              "container in flag",
		},
	}
	for _, test := range tests {
		f, tf, codec := NewAPIFactory()
		tf.Client = &client.FakeRESTClient{
			Codec:  codec,
			Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) { return nil, nil }),
		}
		tf.Namespace = "test"
		tf.ClientConfig = &client.Config{}

		cmd := &cobra.Command{}
		options := test.p
		err := options.Complete(f, cmd, test.args)
		if test.expectError && err == nil {
			t.Errorf("unexpected non-error (%s)", test.name)
		}
		if !test.expectError && err != nil {
			t.Errorf("unexpected error: %v (%s)", err, test.name)
		}
		if err != nil {
			continue
		}
		if options.PodName != test.expectedPod {
			t.Errorf("expected: %s, got: %s (%s)", test.expectedPod, options.PodName, test.name)
		}
		if options.ContainerName != test.expectedContainer {
			t.Errorf("expected: %s, got: %s (%s)", test.expectedContainer, options.ContainerName, test.name)
		}
	}
}

func TestAttach(t *testing.T) {
	version := testapi.Version()
	tests := []struct {
		name, version, podPath, attachPath, container string
		pod                                           *api.Pod
		attachErr                                     bool
	}{
		{
			name:       "pod attach",
			version:    version,
			podPath:    "/api/" + version + "/namespaces/test/pods/foo",
			attachPath: "/api/" + version + "/namespaces/test/pods/foo/attach",
			pod:        attachPod(),
		},
		{
			name:       "pod attach error",
			version:    version,
			podPath:    "/api/" + version + "/namespaces/test/pods/foo",
			attachPath: "/api/" + version + "/namespaces/test/pods/foo/attach",
			pod:        attachPod(),
			attachErr:  true,
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
					t.Errorf("%s: unexpected request: %s %#v\n%#v", test.name, req.Method, req.URL, req)
					return nil, fmt.Errorf("unexpected request")
				}
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = &client.Config{Version: test.version}
		bufOut := bytes.NewBuffer([]byte{})
		bufErr := bytes.NewBuffer([]byte{})
		bufIn := bytes.NewBuffer([]byte{})
		ex := &fakeRemoteAttach{}
		if test.attachErr {
			ex.attachErr = fmt.Errorf("attach error")
		}
		params := &AttachOptions{
			ContainerName: "bar",
			In:            bufIn,
			Out:           bufOut,
			Err:           bufErr,
			Attach:        ex,
		}
		cmd := &cobra.Command{}
		if err := params.Complete(f, cmd, []string{"foo"}); err != nil {
			t.Fatal(err)
		}
		err := params.Run()
		if test.attachErr && err != ex.attachErr {
			t.Errorf("%s: Unexpected exec error: %v", test.name, err)
			continue
		}
		if !test.attachErr && err != nil {
			t.Errorf("%s: Unexpected error: %v", test.name, err)
			continue
		}
		if !test.attachErr && ex.req.URL().Path != test.attachPath {
			t.Errorf("%s: Did not get expected path for exec request", test.name)
			continue
		}
	}
}

func attachPod() *api.Pod {
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
