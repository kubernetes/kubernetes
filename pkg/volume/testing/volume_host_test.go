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

package testing

import (
	"sync"
	gotest "testing"
)

func TestGetNodeLabelsConcurrent(t *gotest.T) {
	host := &fakeVolumeHost{}
	const n = 32
	var wg sync.WaitGroup
	wg.Add(n)
	errCh := make(chan error, n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			labels, err := host.GetNodeLabels()
			if err != nil {
				errCh <- err
				return
			}
			if labels["test-label"] != "test-value" {
				t.Errorf("GetNodeLabels() = %v, want test-label=test-value", labels)
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}
