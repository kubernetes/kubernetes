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
	"errors"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/klog/v2"
)

// StorageReadinessHook implements PostStartHook functionality for checking readiness
// of underlying storage for registered resources.
type StorageReadinessHook struct {
	timeout time.Duration

	lock   sync.Mutex
	checks map[string]func() error
}

// NewStorageReadinessHook created new StorageReadinessHook.
func NewStorageReadinessHook(timeout time.Duration) *StorageReadinessHook {
	return &StorageReadinessHook{
		checks:  make(map[string]func() error),
		timeout: timeout,
	}
}

func (h *StorageReadinessHook) RegisterStorage(gvr metav1.GroupVersionResource, storage rest.StorageWithReadiness) {
	h.lock.Lock()
	defer h.lock.Unlock()

	if _, ok := h.checks[gvr.String()]; !ok {
		h.checks[gvr.String()] = storage.ReadinessCheck
	} else {
		klog.Errorf("Registering storage readiness hook for %s again: ", gvr.String())
	}
}

func (h *StorageReadinessHook) check() bool {
	h.lock.Lock()
	defer h.lock.Unlock()

	failedChecks := []string{}
	for gvr, check := range h.checks {
		if err := check(); err != nil {
			failedChecks = append(failedChecks, gvr)
		}
	}
	if len(failedChecks) == 0 {
		klog.Infof("Storage is ready for all registered resources")
		return true
	}
	klog.V(4).Infof("Storage is not ready for: %v", failedChecks)
	return false
}

func (h *StorageReadinessHook) Hook(ctx PostStartHookContext) error {
	deadlineCtx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()
	err := wait.PollUntilContextCancel(deadlineCtx, 100*time.Millisecond, true,
		func(_ context.Context) (bool, error) {
			if ok := h.check(); ok {
				return true, nil
			}
			return false, nil
		})
	if errors.Is(err, context.DeadlineExceeded) {
		klog.Warningf("Deadline exceeded while waiting for storage readiness... ignoring")
	}
	return nil
}
