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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"sync"
	"testing"
	"testing/iotest"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestLog(t *testing.T) {
	t.Skip("test skipped temporarily to enable 1.19 rebase to merge more quickly")

	tests := []struct {
		name                  string
		opts                  func(genericclioptions.IOStreams) *LogsOptions
		expectedErr           string
		expectedOutSubstrings []string
	}{
		{
			name: "v1 - pod log",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "some-pod",
							FieldPath: "spec.containers{some-container}",
						}: &responseWrapperMock{data: strings.NewReader("test log content\n")},
					},
				}

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = mock.mockConsumeRequest

				return o
			},
			expectedOutSubstrings: []string{"test log content\n"},
		},
		{
			name: "pod logs with prefix",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "test-pod",
							FieldPath: "spec.containers{test-container}",
						}: &responseWrapperMock{data: strings.NewReader("test log content\n")},
					},
				}

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = mock.mockConsumeRequest
				o.Prefix = true

				return o
			},
			expectedOutSubstrings: []string{"[pod/test-pod/test-container] test log content\n"},
		},
		{
			name: "pod logs with prefix: init container",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "test-pod",
							FieldPath: "spec.initContainers{test-container}",
						}: &responseWrapperMock{data: strings.NewReader("test log content\n")},
					},
				}

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = mock.mockConsumeRequest
				o.Prefix = true

				return o
			},
			expectedOutSubstrings: []string{"[pod/test-pod/test-container] test log content\n"},
		},
		{
			name: "pod logs with prefix: ephemeral container",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "test-pod",
							FieldPath: "spec.ephemeralContainers{test-container}",
						}: &responseWrapperMock{data: strings.NewReader("test log content\n")},
					},
				}

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = mock.mockConsumeRequest
				o.Prefix = true

				return o
			},
			expectedOutSubstrings: []string{"[pod/test-pod/test-container] test log content\n"},
		},
		{
			name: "get logs from multiple requests sequentially",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "some-pod-1",
							FieldPath: "spec.containers{some-container}",
						}: &responseWrapperMock{data: strings.NewReader("test log content from source 1\n")},
						{
							Kind:      "Pod",
							Name:      "some-pod-2",
							FieldPath: "spec.containers{some-container}",
						}: &responseWrapperMock{data: strings.NewReader("test log content from source 2\n")},
						{
							Kind:      "Pod",
							Name:      "some-pod-3",
							FieldPath: "spec.containers{some-container}",
						}: &responseWrapperMock{data: strings.NewReader("test log content from source 3\n")},
					},
				}

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = mock.mockConsumeRequest
				return o
			},
			expectedOutSubstrings: []string{
				"test log content from source 1\n",
				"test log content from source 2\n",
				"test log content from source 3\n",
			},
		},
		{
			name: "follow logs from multiple requests concurrently",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				wg := &sync.WaitGroup{}
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "some-pod-1",
							FieldPath: "spec.containers{some-container-1}",
						}: &responseWrapperMock{data: strings.NewReader("test log content from source 1\n")},
						{
							Kind:      "Pod",
							Name:      "some-pod-2",
							FieldPath: "spec.containers{some-container-2}",
						}: &responseWrapperMock{data: strings.NewReader("test log content from source 2\n")},
						{
							Kind:      "Pod",
							Name:      "some-pod-3",
							FieldPath: "spec.containers{some-container-3}",
						}: &responseWrapperMock{data: strings.NewReader("test log content from source 3\n")},
					},
					wg: wg,
				}
				wg.Add(3)

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = mock.mockConsumeRequest
				o.Follow = true
				return o
			},
			expectedOutSubstrings: []string{
				"test log content from source 1\n",
				"test log content from source 2\n",
				"test log content from source 3\n",
			},
		},
		{
			name: "fail to follow logs from multiple requests when there are more logs sources then MaxFollowConcurrency allows",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				wg := &sync.WaitGroup{}
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "test-pod-1",
							FieldPath: "spec.containers{test-container-1}",
						}: &responseWrapperMock{data: strings.NewReader("test log content\n")},
						{
							Kind:      "Pod",
							Name:      "test-pod-2",
							FieldPath: "spec.containers{test-container-2}",
						}: &responseWrapperMock{data: strings.NewReader("test log content\n")},
						{
							Kind:      "Pod",
							Name:      "test-pod-3",
							FieldPath: "spec.containers{test-container-3}",
						}: &responseWrapperMock{data: strings.NewReader("test log content\n")},
					},
					wg: wg,
				}
				wg.Add(3)

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = mock.mockConsumeRequest
				o.MaxFollowConcurrency = 2
				o.Follow = true
				return o
			},
			expectedErr: "you are attempting to follow 3 log streams, but maximum allowed concurrency is 2, use --max-log-requests to increase the limit",
		},
		{
			name: "fail if LogsForObject fails",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				o := NewLogsOptions(streams, false)
				o.LogsForObject = func(restClientGetter genericclioptions.RESTClientGetter, object, options runtime.Object, timeout time.Duration, allContainers bool) (map[corev1.ObjectReference]restclient.ResponseWrapper, error) {
					return nil, errors.New("Error from the LogsForObject")
				}
				return o
			},
			expectedErr: "Error from the LogsForObject",
		},
		{
			name: "fail to get logs, if ConsumeRequestFn fails",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "test-pod-1",
							FieldPath: "spec.containers{test-container-1}",
						}: &responseWrapperMock{},
						{
							Kind:      "Pod",
							Name:      "test-pod-2",
							FieldPath: "spec.containers{test-container-1}",
						}: &responseWrapperMock{},
					},
				}

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = func(req restclient.ResponseWrapper, out io.Writer) error {
					return errors.New("Error from the ConsumeRequestFn")
				}
				return o
			},
			expectedErr: "Error from the ConsumeRequestFn",
		},
		{
			name: "follow logs from multiple requests concurrently with prefix",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				wg := &sync.WaitGroup{}
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "test-pod-1",
							FieldPath: "spec.containers{test-container-1}",
						}: &responseWrapperMock{data: strings.NewReader("test log content from source 1\n")},
						{
							Kind:      "Pod",
							Name:      "test-pod-2",
							FieldPath: "spec.containers{test-container-2}",
						}: &responseWrapperMock{data: strings.NewReader("test log content from source 2\n")},
						{
							Kind:      "Pod",
							Name:      "test-pod-3",
							FieldPath: "spec.containers{test-container-3}",
						}: &responseWrapperMock{data: strings.NewReader("test log content from source 3\n")},
					},
					wg: wg,
				}
				wg.Add(3)

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = mock.mockConsumeRequest
				o.Follow = true
				o.Prefix = true
				return o
			},
			expectedOutSubstrings: []string{
				"[pod/test-pod-1/test-container-1] test log content from source 1\n",
				"[pod/test-pod-2/test-container-2] test log content from source 2\n",
				"[pod/test-pod-3/test-container-3] test log content from source 3\n",
			},
		},
		{
			name: "fail to follow logs from multiple requests, if ConsumeRequestFn fails",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				wg := &sync.WaitGroup{}
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "test-pod-1",
							FieldPath: "spec.containers{test-container-1}",
						}: &responseWrapperMock{},
						{
							Kind:      "Pod",
							Name:      "test-pod-2",
							FieldPath: "spec.containers{test-container-2}",
						}: &responseWrapperMock{},
						{
							Kind:      "Pod",
							Name:      "test-pod-3",
							FieldPath: "spec.containers{test-container-3}",
						}: &responseWrapperMock{},
					},
					wg: wg,
				}
				wg.Add(3)

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = func(req restclient.ResponseWrapper, out io.Writer) error {
					return errors.New("Error from the ConsumeRequestFn")
				}
				o.Follow = true
				return o
			},
			expectedErr: "Error from the ConsumeRequestFn",
		},
		{
			name: "fail to follow logs, if ConsumeRequestFn fails",
			opts: func(streams genericclioptions.IOStreams) *LogsOptions {
				mock := &logTestMock{
					logsForObjectRequests: map[corev1.ObjectReference]restclient.ResponseWrapper{
						{
							Kind:      "Pod",
							Name:      "test-pod-1",
							FieldPath: "spec.containers{test-container-1}",
						}: &responseWrapperMock{},
					},
				}

				o := NewLogsOptions(streams, false)
				o.LogsForObject = mock.mockLogsForObject
				o.ConsumeRequestFn = func(req restclient.ResponseWrapper, out io.Writer) error {
					return errors.New("Error from the ConsumeRequestFn")
				}
				o.Follow = true
				return o
			},
			expectedErr: "Error from the ConsumeRequestFn",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			streams, _, buf, _ := genericclioptions.NewTestIOStreams()

			opts := test.opts(streams)
			opts.Namespace = "test"
			opts.Object = testPod()
			opts.Options = &corev1.PodLogOptions{}
			err := opts.RunLogs()

			if err == nil && len(test.expectedErr) > 0 {
				t.Fatalf("expected error %q, got none", test.expectedErr)
			}

			if err != nil && !strings.Contains(err.Error(), test.expectedErr) {
				t.Errorf("%s: expected to find:\n\t%s\nfound:\n\t%s\n", test.name, test.expectedErr, err.Error())
			}

			bufStr := buf.String()
			if test.expectedOutSubstrings != nil {
				for _, substr := range test.expectedOutSubstrings {
					if !strings.Contains(bufStr, substr) {
						t.Errorf("%s: expected to contain %#v. Output: %#v", test.name, substr, bufStr)
					}
				}
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
			expected: "--tail must be greater than or equal to -1",
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

func TestDefaultConsumeRequest(t *testing.T) {
	tests := []struct {
		name        string
		request     restclient.ResponseWrapper
		expectedErr string
		expectedOut string
	}{
		{
			name: "error from request stream",
			request: &responseWrapperMock{
				err: errors.New("err from the stream"),
			},
			expectedErr: "err from the stream",
		},
		{
			name: "error while reading",
			request: &responseWrapperMock{
				data: iotest.TimeoutReader(strings.NewReader("Some data")),
			},
			expectedErr: iotest.ErrTimeout.Error(),
			expectedOut: "Some data",
		},
		{
			name: "read with empty string",
			request: &responseWrapperMock{
				data: strings.NewReader(""),
			},
			expectedOut: "",
		},
		{
			name: "read without new lines",
			request: &responseWrapperMock{
				data: strings.NewReader("some string without a new line"),
			},
			expectedOut: "some string without a new line",
		},
		{
			name: "read with newlines in the middle",
			request: &responseWrapperMock{
				data: strings.NewReader("foo\nbar"),
			},
			expectedOut: "foo\nbar",
		},
		{
			name: "read with newline at the end",
			request: &responseWrapperMock{
				data: strings.NewReader("foo\n"),
			},
			expectedOut: "foo\n",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			buf := &bytes.Buffer{}
			err := DefaultConsumeRequest(test.request, buf)

			if err != nil && !strings.Contains(err.Error(), test.expectedErr) {
				t.Errorf("%s: expected to find:\n\t%s\nfound:\n\t%s\n", test.name, test.expectedErr, err.Error())
			}

			if buf.String() != test.expectedOut {
				t.Errorf("%s: did not get expected log content. Got: %s", test.name, buf.String())
			}
		})
	}
}

type responseWrapperMock struct {
	data io.Reader
	err  error
}

func (r *responseWrapperMock) DoRaw(context.Context) ([]byte, error) {
	data, _ := ioutil.ReadAll(r.data)
	return data, r.err
}

func (r *responseWrapperMock) Stream(context.Context) (io.ReadCloser, error) {
	return ioutil.NopCloser(r.data), r.err
}

type logTestMock struct {
	logsForObjectRequests map[corev1.ObjectReference]restclient.ResponseWrapper

	// We need a WaitGroup in some test cases to make sure that we fetch logs concurrently.
	// These test cases will finish successfully without the WaitGroup, but the WaitGroup
	// will help us to identify regression when someone accidentally changes
	// concurrent fetching to sequential
	wg *sync.WaitGroup
}

func (l *logTestMock) mockConsumeRequest(request restclient.ResponseWrapper, out io.Writer) error {
	readCloser, err := request.Stream(context.Background())
	if err != nil {
		return err
	}
	defer readCloser.Close()

	// Just copy everything for a test sake
	_, err = io.Copy(out, readCloser)
	if l.wg != nil {
		l.wg.Done()
		l.wg.Wait()
	}
	return err
}

func (l *logTestMock) mockLogsForObject(restClientGetter genericclioptions.RESTClientGetter, object, options runtime.Object, timeout time.Duration, allContainers bool) (map[corev1.ObjectReference]restclient.ResponseWrapper, error) {
	switch object.(type) {
	case *corev1.Pod:
		_, ok := options.(*corev1.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}

		return l.logsForObjectRequests, nil
	default:
		return nil, fmt.Errorf("cannot get the logs from %T", object)
	}
}
