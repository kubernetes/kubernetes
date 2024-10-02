/*
Copyright 2024 The Kubernetes Authors.

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

package server

import (
	"context"
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

type fakeReadinessStorage struct {
	result error
}

func (s *fakeReadinessStorage) New() runtime.Object   { return nil }
func (s *fakeReadinessStorage) Destroy()              {}
func (s *fakeReadinessStorage) ReadinessCheck() error { return s.result }

func testGVR(index int) metav1.GroupVersionResource {
	return metav1.GroupVersionResource{
		Group:    "group",
		Version:  "version",
		Resource: fmt.Sprintf("resource-%d", index),
	}
}

func TestStorageReadinessHook(t *testing.T) {
	h := NewStorageReadinessHook(time.Second)

	numChecks := 5
	storages := make([]*fakeReadinessStorage, numChecks)
	for i := 0; i < numChecks; i++ {
		storages[i] = &fakeReadinessStorage{
			result: fmt.Errorf("failed"),
		}
		h.RegisterStorage(testGVR(i), storages[i])
	}

	for i := 0; i < numChecks; i++ {
		if ok := h.check(); ok {
			t.Errorf("%d: unexpected check pass", i)
		}
		storages[i].result = nil
	}
	if ok := h.check(); !ok {
		t.Errorf("unexpected check failure")
	}
}

func TestStorageReadinessHookTimeout(t *testing.T) {
	h := NewStorageReadinessHook(time.Second)

	storage := &fakeReadinessStorage{
		result: fmt.Errorf("failed"),
	}
	h.RegisterStorage(testGVR(0), storage)

	ctx := context.Background()
	hookCtx := PostStartHookContext{
		LoopbackClientConfig: nil,
		StopCh:               ctx.Done(),
		Context:              ctx,
	}
	if err := h.Hook(hookCtx); err != nil {
		t.Errorf("unexpected hook failure on timeout")
	}
}
