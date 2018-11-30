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

package logs

import (
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestLog(t *testing.T) {
	tests := []struct {
		name, version, podPath, logPath string
		pod                             *corev1.Pod
	}{
		{
			name: "v1 - pod log",
			pod:  testPod(),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logContent := "test log content"
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			streams, _, buf, _ := genericclioptions.NewTestIOStreams()

			mock := &logTestMock{
				logsContent: logContent,
			}

			opts := NewLogsOptions(streams, false)
			opts.Namespace = "test"
			opts.Object = test.pod
			opts.Options = &corev1.PodLogOptions{}
			opts.LogsForObject = mock.mockLogsForObject
			opts.ConsumeRequestFn = mock.mockConsumeRequest
			opts.RunLogs()

			if buf.String() != logContent {
				t.Errorf("%s: did not get expected log content. Got: %s", test.name, buf.String())
			}
		})
	}
}

func testPod() *corev1.Pod {
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
	}
}

func TestValidateLogOptions(t *testing.T) {
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()
	f.WithNamespace("")

	tests := []struct {
		name     string
		args     []string
		opts     func(genericclioptions.IOStreams) *LogsOptions
		expected string
	}{
		{
			name: "since & since-time",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				o := NewLogsOptions(streams, false)
				o.SinceSeconds = time.Hour
				o.SinceTime = "2006-01-02T15:04:05Z"

				var err error
				o.Options, err = o.ToLogOptions()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				return o
			},
			args:     []string{"foo"},
			expected: "at most one of `sinceTime` or `sinceSeconds` may be specified",
		},
		{
			name: "negative since-time",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				o := NewLogsOptions(streams, false)
				o.SinceSeconds = -1 * time.Second

				var err error
				o.Options, err = o.ToLogOptions()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				return o
			},
			args:     []string{"foo"},
			expected: "must be greater than 0",
		},
		{
			name: "negative limit-bytes",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				o := NewLogsOptions(streams, false)
				o.LimitBytes = -100

				var err error
				o.Options, err = o.ToLogOptions()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				return o
			},
			args:     []string{"foo"},
			expected: "must be greater than 0",
		},
		{
			name: "negative tail",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				o := NewLogsOptions(streams, false)
				o.Tail = -100

				var err error
				o.Options, err = o.ToLogOptions()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				return o
			},
			args:     []string{"foo"},
			expected: "must be greater than or equal to 0",
		},
		{
			name: "container name combined with --all-containers",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				o := NewLogsOptions(streams, true)
				o.Container = "my-container"

				var err error
				o.Options, err = o.ToLogOptions()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				return o
			},
			args:     []string{"my-pod", "my-container"},
			expected: "--all-containers=true should not be specified with container",
		},
		{
			name: "container name combined with second argument",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				o := NewLogsOptions(streams, false)
				o.Container = "my-container"
				o.ContainerNameSpecified = true

				var err error
				o.Options, err = o.ToLogOptions()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				return o
			},
			args:     []string{"my-pod", "my-container"},
			expected: "only one of -c or an inline",
		},
		{
			name: "follow and selector conflict",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				o := NewLogsOptions(streams, false)
				o.Selector = "foo"
				o.Follow = true

				var err error
				o.Options, err = o.ToLogOptions()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				return o
			},
			expected: "only one of follow (-f) or selector (-l) is allowed",
		},
	}
	for _, test := range tests {
		streams := genericclioptions.NewTestIOStreamsDiscard()

		o := test.opts(streams)
		o.Resources = test.args

		err := o.Validate()
		if err == nil {
			t.Fatalf("expected error %q, got none", test.expected)
		}

		if !strings.Contains(err.Error(), test.expected) {
			t.Errorf("%s: expected to find:\n\t%s\nfound:\n\t%s\n", test.name, test.expected, err.Error())
		}
	}
}

func TestLogComplete(t *testing.T) {
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	tests := []struct {
		name     string
		args     []string
		opts     func(genericclioptions.IOStreams) *LogsOptions
		expected string
	}{
		{
			name: "One args case",
			args: []string{"foo"},
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				o := NewLogsOptions(streams, false)
				o.Selector = "foo"
				return o
			},
			expected: "only a selector (-l) or a POD name is allowed",
		},
	}
	for _, test := range tests {
		cmd := NewCmdLogs(f, genericclioptions.NewTestIOStreamsDiscard())
		out := ""

		// checkErr breaks tests in case of errors, plus we just
		// need to check errors returned by the command validation
		o := test.opts(genericclioptions.NewTestIOStreamsDiscard())
		err := o.Complete(f, cmd, test.args)
		if err == nil {
			t.Fatalf("expected error %q, got none", test.expected)
		}

		out = err.Error()
		if !strings.Contains(out, test.expected) {
			t.Errorf("%s: expected to find:\n\t%s\nfound:\n\t%s\n", test.name, test.expected, out)
		}
	}
}

type logTestMock struct {
	logsContent string
}

func (l *logTestMock) mockConsumeRequest(req *restclient.Request, out io.Writer) error {
	fmt.Fprintf(out, l.logsContent)
	return nil
}

func (l *logTestMock) mockLogsForObject(restClientGetter genericclioptions.RESTClientGetter, object, options runtime.Object, timeout time.Duration, allContainers bool) ([]*restclient.Request, error) {
	switch object.(type) {
	case *corev1.Pod:
		_, ok := options.(*corev1.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}

		return []*restclient.Request{{}}, nil
	default:
		return nil, fmt.Errorf("cannot get the logs from %T", object)
	}
}
