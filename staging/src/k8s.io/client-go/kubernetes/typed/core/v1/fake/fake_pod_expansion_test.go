/*
Copyright 2020 The Kubernetes Authors.

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

package fake

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	cgtesting "k8s.io/client-go/testing"
)

func TestFakePodsGetLogs(t *testing.T) {
	fp := newFakePods(&FakeCoreV1{Fake: &cgtesting.Fake{}}, "default")
	req := fp.GetLogs("foo", &corev1.PodLogOptions{})
	body, err := req.Stream(context.Background())
	if err != nil {
		t.Fatal("Stream pod logs:", err)
	}
	var buf bytes.Buffer
	n, err := io.Copy(&buf, body)
	if err != nil {
		t.Fatal("Read pod logs:", err)
	}
	if n == 0 {
		t.Fatal("Empty log")
	}
	err = body.Close()
	if err != nil {
		t.Fatal("Close response body:", err)
	}
}

func TestFakePodsGetLogsReactorError(t *testing.T) {
	fake := &cgtesting.Fake{}
	fp := newFakePods(&FakeCoreV1{Fake: fake}, "default")
	expectedErr := errors.New("reactor get logs failure")
	fake.PrependReactor("get", "pods/log", func(action cgtesting.Action) (bool, runtime.Object, error) {
		genericAction, ok := action.(cgtesting.GenericAction)
		if !ok {
			t.Fatalf("expected GenericAction, got %T", action)
		}
		opts, ok := genericAction.GetValue().(*corev1.PodLogOptions)
		if !ok {
			t.Fatalf("expected *corev1.PodLogOptions, got %T", genericAction.GetValue())
		}
		if opts.Container != "ctr" {
			t.Fatalf("expected container ctr, got %q", opts.Container)
		}
		return true, nil, expectedErr
	})

	req := fp.GetLogs("foo", &corev1.PodLogOptions{Container: "ctr"})
	_, err := req.Stream(context.Background())
	if !errors.Is(err, expectedErr) {
		t.Fatalf("expected stream error %v, got %v", expectedErr, err)
	}
}

func TestFakePodsGetLogsReactorResponse(t *testing.T) {
	fake := &cgtesting.Fake{}
	fp := newFakePods(&FakeCoreV1{Fake: fake}, "default")
	expectedLogs := "reactor logs"
	fake.PrependReactor("get", "pods/log", func(action cgtesting.Action) (bool, runtime.Object, error) {
		return true, &runtime.Unknown{Raw: []byte(expectedLogs)}, nil
	})

	req := fp.GetLogs("foo", &corev1.PodLogOptions{})
	body, err := req.Stream(context.Background())
	if err != nil {
		t.Fatalf("Stream pod logs: %v", err)
	}
	defer func() {
		if err := body.Close(); err != nil {
			t.Fatalf("Close response body: %v", err)
		}
	}()

	logs, err := io.ReadAll(body)
	if err != nil {
		t.Fatalf("Read pod logs: %v", err)
	}
	if string(logs) != expectedLogs {
		t.Fatalf("expected logs %q, got %q", expectedLogs, string(logs))
	}
}

func TestFakePodsGetLogsReactorInvalidObject(t *testing.T) {
	fake := &cgtesting.Fake{}
	fp := newFakePods(&FakeCoreV1{Fake: fake}, "default")
	fake.PrependReactor("get", "pods/log", func(action cgtesting.Action) (bool, runtime.Object, error) {
		return true, &corev1.Pod{}, nil
	})

	req := fp.GetLogs("foo", &corev1.PodLogOptions{})
	_, err := req.Stream(context.Background())
	if err == nil {
		t.Fatal("expected stream error")
	}
	if !strings.Contains(err.Error(), "expected reactor to return *runtime.Unknown") {
		t.Fatalf("expected helpful reactor object type error, got: %v", err)
	}
}
