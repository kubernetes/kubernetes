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

package cmd

import (
	"bytes"
	"errors"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"

	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/polymorphichelpers"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

func TestLog(t *testing.T) {
	tests := []struct {
		name, version, podPath, logPath string
		pod                             *api.Pod
	}{
		{
			name:    "v1 - pod log",
			version: "v1",
			podPath: "/namespaces/test/pods/foo",
			logPath: "/api/v1/namespaces/test/pods/foo/log",
			pod:     testPod(),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logContent := "test log content"
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			codec := legacyscheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := legacyscheme.Codecs

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == test.podPath && m == "GET":
						body := objBody(codec, test.pod)
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
					case p == test.logPath && m == "GET":
						body := ioutil.NopCloser(bytes.NewBufferString(logContent))
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
					default:
						t.Errorf("%s: unexpected request: %#v\n%#v", test.name, req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = defaultClientConfig()
			oldLogFn := polymorphichelpers.LogsForObjectFn
			defer func() {
				polymorphichelpers.LogsForObjectFn = oldLogFn
			}()
			clientset, err := tf.ClientSet()
			if err != nil {
				t.Fatal(err)
			}
			polymorphichelpers.LogsForObjectFn = logTestMock{client: clientset}.logsForObject

			streams, _, buf, _ := genericclioptions.NewTestIOStreams()

			cmd := NewCmdLogs(tf, streams)
			cmd.Flags().Set("namespace", "test")
			cmd.Run(cmd, []string{"foo"})

			if buf.String() != logContent {
				t.Errorf("%s: did not get expected log content. Got: %s", test.name, buf.String())
			}
		})
	}
}

func testPod() *api.Pod {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
			Containers: []api.Container{
				{
					Name: "bar",
				},
			},
		},
	}
}

func TestValidateLogFlags(t *testing.T) {
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()
	f.WithNamespace("")

	tests := []struct {
		name     string
		flags    map[string]string
		args     []string
		expected string
	}{
		{
			name:     "since & since-time",
			flags:    map[string]string{"since": "1h", "since-time": "2006-01-02T15:04:05Z"},
			args:     []string{"foo"},
			expected: "at most one of `sinceTime` or `sinceSeconds` may be specified",
		},
		{
			name:     "negative since-time",
			flags:    map[string]string{"since": "-1s"},
			args:     []string{"foo"},
			expected: "must be greater than 0",
		},
		{
			name:     "negative limit-bytes",
			flags:    map[string]string{"limit-bytes": "-100"},
			args:     []string{"foo"},
			expected: "must be greater than 0",
		},
		{
			name:     "negative tail",
			flags:    map[string]string{"tail": "-100"},
			args:     []string{"foo"},
			expected: "must be greater than or equal to 0",
		},
		{
			name:     "container name combined with --all-containers",
			flags:    map[string]string{"all-containers": "true"},
			args:     []string{"my-pod", "my-container"},
			expected: "--all-containers=true should not be specified with container",
		},
	}
	for _, test := range tests {
		streams := genericclioptions.NewTestIOStreamsDiscard()
		cmd := NewCmdLogs(f, streams)
		out := ""
		for flag, value := range test.flags {
			cmd.Flags().Set(flag, value)
		}
		// checkErr breaks tests in case of errors, plus we just
		// need to check errors returned by the command validation
		o := NewLogsOptions(streams, test.flags["all-containers"] == "true")
		cmd.Run = func(cmd *cobra.Command, args []string) {
			o.Complete(f, cmd, args)
			out = o.Validate().Error()
		}
		cmd.Run(cmd, test.args)

		if !strings.Contains(out, test.expected) {
			t.Errorf("%s: expected to find:\n\t%s\nfound:\n\t%s\n", test.name, test.expected, out)
		}
	}
}

func TestLogComplete(t *testing.T) {
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	tests := []struct {
		name     string
		args     []string
		flags    map[string]string
		expected string
	}{
		{
			name:     "No args case",
			flags:    map[string]string{"selector": ""},
			expected: "'logs (POD | TYPE/NAME) [CONTAINER_NAME]'.\nPOD or TYPE/NAME is a required argument for the logs command",
		},
		{
			name:     "One args case",
			args:     []string{"foo"},
			flags:    map[string]string{"selector": "foo"},
			expected: "only a selector (-l) or a POD name is allowed",
		},
		{
			name:     "Two args case",
			args:     []string{"foo", "foo1"},
			flags:    map[string]string{"container": "foo1"},
			expected: "only one of -c or an inline [CONTAINER] arg is allowed",
		},
		{
			name:     "More than two args case",
			args:     []string{"foo", "foo1", "foo2"},
			flags:    map[string]string{"tail": "1"},
			expected: "'logs (POD | TYPE/NAME) [CONTAINER_NAME]'.\nPOD or TYPE/NAME is a required argument for the logs command",
		},
		{
			name:     "follow and selecter conflict",
			flags:    map[string]string{"selector": "foo", "follow": "true"},
			expected: "only one of follow (-f) or selector (-l) is allowed",
		},
	}
	for _, test := range tests {
		cmd := NewCmdLogs(f, genericclioptions.NewTestIOStreamsDiscard())
		var err error
		out := ""
		for flag, value := range test.flags {
			cmd.Flags().Set(flag, value)
		}
		// checkErr breaks tests in case of errors, plus we just
		// need to check errors returned by the command validation
		o := NewLogsOptions(genericclioptions.NewTestIOStreamsDiscard(), false)
		err = o.Complete(f, cmd, test.args)
		out = err.Error()
		if !strings.Contains(out, test.expected) {
			t.Errorf("%s: expected to find:\n\t%s\nfound:\n\t%s\n", test.name, test.expected, out)
		}
	}
}

type logTestMock struct {
	client internalclientset.Interface
}

func (m logTestMock) logsForObject(restClientGetter genericclioptions.RESTClientGetter, object, options runtime.Object, timeout time.Duration) (*restclient.Request, error) {
	switch t := object.(type) {
	case *api.Pod:
		opts, ok := options.(*api.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}
		return m.client.Core().Pods(t.Namespace).GetLogs(t.Name, opts), nil
	default:
		return nil, fmt.Errorf("cannot get the logs from %T", object)
	}
}
