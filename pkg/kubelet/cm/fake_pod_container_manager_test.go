/*
Copyright 2026 The Kubernetes Authors.

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

package cm

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
)

func TestFakePodContainerManagerDestroyBlocksAndSignals(t *testing.T) {
	manager := NewFakePodContainerManager()
	podUID := types.UID("pod-1")
	cgroupName := CgroupName{"pod-1"}
	manager.Cgroups[podUID] = cgroupName

	startedCh := make(chan struct{}, 1)
	blockCh := make(chan struct{})
	wg := &sync.WaitGroup{}
	wg.Add(1)
	manager.DestroyStartedCh = startedCh
	manager.DestroyBlockCh = blockCh
	manager.DestroyWG = wg
	manager.DestroyError = fmt.Errorf("boom")

	errCh := make(chan error, 1)
	go func() {
		errCh <- manager.Destroy(klog.Background(), cgroupName)
	}()

	select {
	case <-startedCh:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for Destroy to start")
	}

	if got := manager.GetDestroyCallCount(cgroupName); got != 1 {
		t.Fatalf("expected Destroy to be called once, got %d", got)
	}

	close(blockCh)
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for Destroy to finish")
	}

	err := <-errCh
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if _, ok := manager.Cgroups[podUID]; !ok {
		t.Fatal("cgroup should not be deleted when Destroy returns error")
	}
}

func TestFakePodContainerManagerDestroyDeletesCgroup(t *testing.T) {
	manager := NewFakePodContainerManager()
	podUID := types.UID("pod-2")
	cgroupName := CgroupName{"pod-2"}
	manager.Cgroups[podUID] = cgroupName

	wg := &sync.WaitGroup{}
	wg.Add(1)
	manager.DestroyWG = wg

	if err := manager.Destroy(klog.Background(), cgroupName); err != nil {
		t.Fatalf("Destroy returned error: %v", err)
	}

	wg.Wait()

	if got := manager.GetDestroyCallCount(cgroupName); got != 1 {
		t.Fatalf("expected Destroy to be called once, got %d", got)
	}
	if _, ok := manager.Cgroups[podUID]; ok {
		t.Fatal("cgroup should be deleted on successful Destroy")
	}
}
