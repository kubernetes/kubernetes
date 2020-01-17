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

package exec

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/tools/remotecommand"

	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/term"
)

type fakeRemoteExecutor struct {
	method  string
	url     *url.URL
	execErr error
}

func (f *fakeRemoteExecutor) Execute(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue remotecommand.TerminalSizeQueue) error {
	f.method = method
	f.url = url
	return f.execErr
}

func TestPodAndContainer(t *testing.T) {
	tests := []struct {
		args              []string
		argsLenAtDash     int
		p                 *ExecOptions
		name              string
		expectError       bool
		expectedPod       string
		expectedContainer string
		expectedArgs      []string
		obj               *corev1.Pod
	}{
		{
			p:             &ExecOptions{},
			argsLenAtDash: -1,
			expectError:   true,
			name:          "empty",
		},
		{
			p:             &ExecOptions{},
			argsLenAtDash: -1,
			expectError:   true,
			name:          "no cmd",
			obj:           execPod(),
		},
		{
			p:             &ExecOptions{StreamOptions: StreamOptions{ContainerName: "bar"}},
			argsLenAtDash: -1,
			expectError:   true,
			name:          "no cmd, w/ container",
			obj:           execPod(),
		},
		{
			p:             &ExecOptions{},
			args:          []string{"foo", "cmd"},
			argsLenAtDash: 0,
			expectError:   true,
			name:          "no pod, pod name is behind dash",
			obj:           execPod(),
		},
		{
			p:             &ExecOptions{},
			args:          []string{"foo"},
			argsLenAtDash: -1,
			expectError:   true,
			name:          "no cmd, w/o flags",
			obj:           execPod(),
		},
		{
			p:             &ExecOptions{},
			args:          []string{"foo", "cmd"},
			argsLenAtDash: -1,
			expectedPod:   "foo",
			expectedArgs:  []string{"cmd"},
			name:          "cmd, w/o flags",
			obj:           execPod(),
		},
		{
			p:             &ExecOptions{},
			args:          []string{"foo", "cmd"},
			argsLenAtDash: 1,
			expectedPod:   "foo",
			expectedArgs:  []string{"cmd"},
			name:          "cmd, cmd is behind dash",
			obj:           execPod(),
		},
		{
			p:                 &ExecOptions{StreamOptions: StreamOptions{ContainerName: "bar"}},
			args:              []string{"foo", "cmd"},
			argsLenAtDash:     -1,
			expectedPod:       "foo",
			expectedContainer: "bar",
			expectedArgs:      []string{"cmd"},
			name:              "cmd, container in flag",
			obj:               execPod(),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var err error
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			ns := scheme.Codecs

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client:               fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) { return nil, nil }),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			cmd := NewCmdExec(tf, genericclioptions.NewTestIOStreamsDiscard())
			options := test.p
			options.ErrOut = bytes.NewBuffer([]byte{})
			options.Out = bytes.NewBuffer([]byte{})
			err = options.Complete(tf, cmd, test.args, test.argsLenAtDash)
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

			pod, err := options.ExecutablePodFn(tf, test.obj, defaultPodExecTimeout)
			if pod.Name != test.expectedPod {
				t.Errorf("%s: expected: %s, got: %s", test.name, test.expectedPod, options.PodName)
			}
			if options.ContainerName != test.expectedContainer {
				t.Errorf("%s: expected: %s, got: %s", test.name, test.expectedContainer, options.ContainerName)
			}
			if !reflect.DeepEqual(test.expectedArgs, options.Command) {
				t.Errorf("%s: expected: %v, got %v", test.name, test.expectedArgs, options.Command)
			}
		})
	}
}

func TestExec(t *testing.T) {
	version := "v1"
	tests := []struct {
		name, version, podPath, fetchPodPath, execPath string
		pod                                            *corev1.Pod
		execErr                                        bool
	}{
		{
			name:         "pod exec",
			version:      version,
			podPath:      "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath: "/namespaces/test/pods/foo",
			execPath:     "/api/" + version + "/namespaces/test/pods/foo/exec",
			pod:          execPod(),
		},
		{
			name:         "pod exec error",
			version:      version,
			podPath:      "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath: "/namespaces/test/pods/foo",
			execPath:     "/api/" + version + "/namespaces/test/pods/foo/exec",
			pod:          execPod(),
			execErr:      true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := scheme.Codecs

			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == test.podPath && m == "GET":
						body := cmdtesting.ObjBody(codec, test.pod)
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					case p == test.fetchPodPath && m == "GET":
						body := cmdtesting.ObjBody(codec, test.pod)
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", test.name, req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
			}
			tf.ClientConfigVal = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: scheme.Codecs, GroupVersion: &schema.GroupVersion{Version: test.version}}}
			ex := &fakeRemoteExecutor{}
			if test.execErr {
				ex.execErr = fmt.Errorf("exec error")
			}
			params := &ExecOptions{
				StreamOptions: StreamOptions{
					PodName:       "foo",
					ContainerName: "bar",
					IOStreams:     genericclioptions.NewTestIOStreamsDiscard(),
				},
				Executor: ex,
			}
			cmd := NewCmdExec(tf, genericclioptions.NewTestIOStreamsDiscard())
			args := []string{"pod/foo", "command"}
			if err := params.Complete(tf, cmd, args, -1); err != nil {
				t.Fatal(err)
			}
			err := params.Run()
			if test.execErr && err != ex.execErr {
				t.Errorf("%s: Unexpected exec error: %v", test.name, err)
				return
			}
			if !test.execErr && err != nil {
				t.Errorf("%s: Unexpected error: %v", test.name, err)
				return
			}
			if test.execErr {
				return
			}
			if ex.url.Path != test.execPath {
				t.Errorf("%s: Did not get expected path for exec request", test.name)
				return
			}
			if strings.Count(ex.url.RawQuery, "container=bar") != 1 {
				t.Errorf("%s: Did not get expected container query param for exec request", test.name)
				return
			}
			if ex.method != "POST" {
				t.Errorf("%s: Did not get method for exec request: %s", test.name, ex.method)
			}
		})
	}
}

func execPod() *corev1.Pod {
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
			Phase: corev1.PodRunning,
		},
	}
}

func TestSetupTTY(t *testing.T) {
	streams, _, _, stderr := genericclioptions.NewTestIOStreams()

	// test 1 - don't attach stdin
	o := &StreamOptions{
		// InterruptParent: ,
		Stdin:     false,
		IOStreams: streams,
		TTY:       true,
	}

	tty := o.SetupTTY()

	if o.In != nil {
		t.Errorf("don't attach stdin: o.In should be nil")
	}
	if tty.In != nil {
		t.Errorf("don't attach stdin: tty.In should be nil")
	}
	if o.TTY {
		t.Errorf("don't attach stdin: o.TTY should be false")
	}
	if tty.Raw {
		t.Errorf("don't attach stdin: tty.Raw should be false")
	}
	if len(stderr.String()) > 0 {
		t.Errorf("don't attach stdin: stderr wasn't empty: %s", stderr.String())
	}

	// tests from here on attach stdin
	// test 2 - don't request a TTY
	o.Stdin = true
	o.In = &bytes.Buffer{}
	o.TTY = false

	tty = o.SetupTTY()

	if o.In == nil {
		t.Errorf("attach stdin, no TTY: o.In should not be nil")
	}
	if tty.In != o.In {
		t.Errorf("attach stdin, no TTY: tty.In should equal o.In")
	}
	if o.TTY {
		t.Errorf("attach stdin, no TTY: o.TTY should be false")
	}
	if tty.Raw {
		t.Errorf("attach stdin, no TTY: tty.Raw should be false")
	}
	if len(stderr.String()) > 0 {
		t.Errorf("attach stdin, no TTY: stderr wasn't empty: %s", stderr.String())
	}

	// test 3 - request a TTY, but stdin is not a terminal
	o.Stdin = true
	o.In = &bytes.Buffer{}
	o.ErrOut = stderr
	o.TTY = true

	tty = o.SetupTTY()

	if o.In == nil {
		t.Errorf("attach stdin, TTY, not a terminal: o.In should not be nil")
	}
	if tty.In != o.In {
		t.Errorf("attach stdin, TTY, not a terminal: tty.In should equal o.In")
	}
	if o.TTY {
		t.Errorf("attach stdin, TTY, not a terminal: o.TTY should be false")
	}
	if tty.Raw {
		t.Errorf("attach stdin, TTY, not a terminal: tty.Raw should be false")
	}
	if !strings.Contains(stderr.String(), "input is not a terminal") {
		t.Errorf("attach stdin, TTY, not a terminal: expected 'input is not a terminal' to stderr")
	}

	// test 4 - request a TTY, stdin is a terminal
	o.Stdin = true
	o.In = &bytes.Buffer{}
	stderr.Reset()
	o.TTY = true

	overrideStdin := ioutil.NopCloser(&bytes.Buffer{})
	overrideStdout := &bytes.Buffer{}
	overrideStderr := &bytes.Buffer{}
	o.overrideStreams = func() (io.ReadCloser, io.Writer, io.Writer) {
		return overrideStdin, overrideStdout, overrideStderr
	}

	o.isTerminalIn = func(tty term.TTY) bool {
		return true
	}

	tty = o.SetupTTY()

	if o.In != overrideStdin {
		t.Errorf("attach stdin, TTY, is a terminal: o.In should equal overrideStdin")
	}
	if tty.In != o.In {
		t.Errorf("attach stdin, TTY, is a terminal: tty.In should equal o.In")
	}
	if !o.TTY {
		t.Errorf("attach stdin, TTY, is a terminal: o.TTY should be true")
	}
	if !tty.Raw {
		t.Errorf("attach stdin, TTY, is a terminal: tty.Raw should be true")
	}
	if len(stderr.String()) > 0 {
		t.Errorf("attach stdin, TTY, is a terminal: stderr wasn't empty: %s", stderr.String())
	}
	if o.Out != overrideStdout {
		t.Errorf("attach stdin, TTY, is a terminal: o.Out should equal overrideStdout")
	}
	if tty.Out != o.Out {
		t.Errorf("attach stdin, TTY, is a terminal: tty.Out should equal o.Out")
	}
}
