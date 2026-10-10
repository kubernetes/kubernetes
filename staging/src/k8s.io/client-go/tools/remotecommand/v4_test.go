/*
Copyright 2016 The Kubernetes Authors.

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

package remotecommand

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2/ktesting"
)

func TestV4ErrorDecoder(t *testing.T) {
	dec := errorDecoderV4{}

	type Test struct {
		message string
		err     string
	}

	for _, test := range []Test{
		{
			message: "",
			err:     "error stream closed before receiving a status message: command execution may have been interrupted",
		},
		{
			message: "{}",
			err:     "error stream protocol error: unknown error",
		},
		{
			message: "{",
			err:     "unexpected end of JSON input in \"{\"",
		},
		{
			message: `{"status": "Success" }`,
			err:     "",
		},
		{
			message: `{"status": "Failure", "message": "foobar" }`,
			err:     "foobar",
		},
		{
			message: `{"status": "Failure", "message": "foobar", "reason": "NonZeroExitCode", "details": {"causes": [{"reason": "foo"}] } }`,
			err:     "error stream protocol error: no ExitCode cause given",
		},
		{
			message: `{"status": "Failure", "message": "foobar", "reason": "NonZeroExitCode", "details": {"causes": [{"reason": "ExitCode"}] } }`,
			err:     "error stream protocol error: invalid exit code value \"\"",
		},
		{
			message: `{"status": "Failure", "message": "foobar", "reason": "NonZeroExitCode", "details": {"causes": [{"reason": "ExitCode", "message": "42"}] } }`,
			err:     "command terminated with exit code 42",
		},
	} {
		err := dec.decode([]byte(test.message))
		want := test.err
		if want == "" {
			want = "<nil>"
		}
		if got := fmt.Sprintf("%v", err); !strings.Contains(got, want) {
			t.Errorf("wrong error for message %q: want=%q, got=%q", test.message, want, got)
		}
	}
}

func TestV4WatchErrorStreamEmptyMessage(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	// Simulate an empty error stream (e.g., connection closed during kubelet restart).
	// With errorDecoderV4, this should return an error, not nil.
	h := newStreamProtocolV4(StreamOptions{}).(*streamProtocolV4)
	h.errorStream = strings.NewReader("")

	ch := watchErrorStream(logger, h.errorStream, &errorDecoderV4{})
	if ch == nil {
		t.Fatal("unexpected nil channel")
	}

	select {
	case err := <-ch:
		if err == nil {
			t.Error("expected an error for empty error stream with v4 decoder, got nil")
		} else if !strings.Contains(err.Error(), "error stream closed before receiving a status message") {
			t.Errorf("unexpected error message: %v", err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("timed out waiting for error")
	}
}
