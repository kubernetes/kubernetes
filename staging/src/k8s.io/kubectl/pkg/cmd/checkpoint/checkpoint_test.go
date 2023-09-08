/*
Copyright 2014 The Kubernetes Authors.

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

package checkpoint

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"

	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

type fakeRemoteCheckpointer struct {
	url *url.URL
}

func (f *fakeRemoteCheckpointer) Checkpoint(request *restclient.Request) (string, error) {
	f.url = request.URL()
	return "checkpoint-archive.tar", nil
}

func TestPodAndContainer(t *testing.T) {
	tests := []struct {
		args              []string
		p                 *CheckpointOptions
		name              string
		expectError       bool
		expectedPod       string
		expectedContainer string
		expectedArgs      []string
		obj               *corev1.Pod
	}{
		{
			p:           &CheckpointOptions{},
			expectError: true,
			name:        "empty",
		},
		{
			p:           &CheckpointOptions{},
			args:        []string{"foo", "bar"},
			expectError: true,
			name:        "multiple arguments",
		},
		{
			p:           &CheckpointOptions{},
			args:        []string{"foo"},
			name:        "one argument",
			expectedPod: "foo",
			obj:         checkpointPod(false),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var err error
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client:               fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) { return nil, nil }),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			cmd := NewCmdCheckpoint(tf, genericiooptions.NewTestIOStreamsDiscard())
			options := test.p
			err = options.Complete(tf, cmd, test.args)
			if !test.expectError && err != nil {
				t.Errorf("%s: unexpected error: %v", test.name, err)
			}
			err = options.Validate()

			if test.expectError && err == nil {
				t.Errorf("%s: unexpected non-error", test.name)
			}
			if !test.expectError && err != nil {
				t.Errorf("%s: unexpected error: %v", test.name, err)
			}
			if err != nil {
				return
			}

			pod, _ := options.PodFunction(tf, test.obj, defaultPodCheckpointTimeout)
			if pod.Name != test.expectedPod {
				t.Errorf("%s: expected: %s, got: %s", test.name, test.expectedPod, options.ResourceName)
			}
			if options.ContainerName != test.expectedContainer {
				t.Errorf("%s: expected: %s, got: %s", test.name, test.expectedContainer, options.ContainerName)
			}
		})
	}
}

func TestCheckpoint(t *testing.T) {
	version := "v1"
	tests := []struct {
		pod            *corev1.Pod
		name           string
		version        string
		podPath        string
		fetchPodPath   string
		checkpointPath string
		errorMessage   string
		containerName  string
	}{
		{
			name:          "pod checkpoint error",
			version:       version,
			podPath:       "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath:  "/namespaces/test/pods/foo",
			pod:           checkpointPod(false),
			errorMessage:  "cannot checkpoint a container in non-running pod; current phase is Failed",
			containerName: "bar",
		},
		{
			name:           "pod checkpoint",
			version:        version,
			podPath:        "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath:   "/namespaces/test/pods/foo",
			checkpointPath: "/api/" + version + "/namespaces/test/pods/foo/checkpoint",
			pod:            checkpointPod(true),
			containerName:  "bar",
		},
		{
			name:         "pod checkpoint error",
			version:      version,
			podPath:      "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath: "/namespaces/test/pods/foo",
			pod:          checkpointPod(false),
			errorMessage: "cannot checkpoint a container in non-running pod; current phase is Failed",
		},
		{
			name:           "pod checkpoint",
			version:        version,
			podPath:        "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath:   "/namespaces/test/pods/foo",
			checkpointPath: "/api/" + version + "/namespaces/test/pods/foo/checkpoint",
			pod:            checkpointPod(true),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == test.podPath && m == "GET":
						body := cmdtesting.ObjBody(codec, test.pod)
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					case p == test.fetchPodPath && m == "GET":
						body := cmdtesting.ObjBody(codec, test.pod)
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", test.name, req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
			}
			tf.ClientConfigVal = &restclient.Config{
				APIPath: "/api",
				ContentConfig: restclient.ContentConfig{
					NegotiatedSerializer: scheme.Codecs,
					GroupVersion: &schema.GroupVersion{
						Version: test.version,
					},
				},
			}
			ex := &fakeRemoteCheckpointer{}
			params := &CheckpointOptions{
				ResourceName:  "foo",
				ContainerName: test.containerName,
				Checkpointer:  ex,
			}
			cmd := NewCmdCheckpoint(tf, genericiooptions.NewTestIOStreamsDiscard())
			args := []string{"pod/foo"}
			if err := params.Complete(tf, cmd, args); err != nil {
				t.Fatal(err)
			}
			err := params.Run(tf)
			if len(test.errorMessage) == 0 && err != nil {
				t.Errorf("%s: Unexpected checkpoint error: %v", test.name, err)
				return
			}
			if len(test.errorMessage) != 0 && err == nil {
				t.Errorf("%s: Unexpected error: %v", test.name, err)
				return
			}
			if len(test.errorMessage) != 0 && test.errorMessage != err.Error() {
				t.Errorf("%s: Unexpected error: %v", test.name, err)
				return

			}
			if len(test.errorMessage) != 0 {
				return
			}
			if ex.url.Path != test.checkpointPath {
				t.Errorf(
					"%s: Did not get expected path (%s) for checkpoint request (%s)",
					test.name,
					test.checkpointPath,
					ex.url.Path,
				)
				return
			}
			if strings.Count(ex.url.RawQuery, "container=bar") != 1 {
				t.Errorf("%s: Did not get expected container query param for exec request", test.name)
				return
			}
		})
	}
}

func checkpointPod(running bool) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyAlways,
			DNSPolicy:     corev1.DNSClusterFirst,
			Containers: []corev1.Container{
				{
					Name: "bar",
				},
			},
		},
		Status: corev1.PodStatus{
			Phase: func() corev1.PodPhase {
				if running {
					return corev1.PodRunning
				}
				return corev1.PodFailed
			}(),
		},
	}
}
