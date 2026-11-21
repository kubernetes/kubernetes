/*
Copyright The Kubernetes Authors.

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

package criproxy

import (
	"errors"
	"sync"
	"testing"
	"time"
)

func TestInjectorsSnapshot(t *testing.T) {
	proxy := &RemoteRuntime{}
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblock()
	originalErr := errors.New("original injector")
	updatedErr := errors.New("updated injector")
	proxy.AddInjector(func(apiName string) error {
		if apiName != CheckpointPod {
			t.Errorf("got API %q, want %q", apiName, CheckpointPod)
		}
		close(entered)
		<-release
		return nil
	})
	proxy.AddInjector(func(string) error { return originalErr })

	result := make(chan error, 1)
	go func() { result <- proxy.runInjectors(CheckpointPod) }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("checkpoint injector did not start")
	}

	updated := make(chan struct{})
	go func() {
		proxy.ResetInjectors()
		proxy.AddInjector(func(string) error { return updatedErr })
		close(updated)
	}()
	select {
	case <-updated:
	case <-time.After(5 * time.Second):
		t.Fatal("reset and add blocked behind an active injector")
	}
	if err := proxy.runInjectors(ExecSync); !errors.Is(err, updatedErr) {
		t.Fatalf("new RPC returned %v, want %v", err, updatedErr)
	}
	unblock()
	select {
	case err := <-result:
		if !errors.Is(err, originalErr) {
			t.Fatalf("active RPC returned %v, want original snapshot error %v", err, originalErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("checkpoint injector did not finish after release")
	}
}

func TestInjectorsConcurrentUpdates(t *testing.T) {
	proxy := &RemoteRuntime{}
	start := make(chan struct{})
	var workers sync.WaitGroup
	workers.Add(2)
	go func() {
		defer workers.Done()
		<-start
		for range 1000 {
			proxy.ResetInjectors()
			proxy.AddInjector(func(string) error { return nil })
		}
	}()
	go func() {
		defer workers.Done()
		<-start
		for range 1000 {
			if err := proxy.runInjectors(ExecSync); err != nil {
				t.Errorf("probe RPC injector failed: %v", err)
			}
		}
	}()
	close(start)
	workers.Wait()
}
