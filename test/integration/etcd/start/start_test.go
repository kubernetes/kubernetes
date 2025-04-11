/*
Copyright 2025 The Kubernetes Authors.

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

package start

import (
	"fmt"
	"net"
	"sync"
	"testing"

	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestStart stress-tests starting multiple etcd instances in parallel.
func TestStart(t *testing.T) {
	tCtx := ktesting.Init(t)
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// If this fails, it'll call t.Fatal, which kills the goroutine.
			// That's okay in this case, wg.Done still gets called.
			framework.StartEtcd(tCtx.Logger().WithName(fmt.Sprintf("instance-%d", i)), t, true)
			// Increase the risk of port collisions by also listening on several ports ourselves.
			for e := 0; e < 10; e++ {
				l, err := net.Listen("tcp", "127.0.0.1:0")
				if err != nil {
					tCtx.Errorf("unexpected error listening on local address: %v", err)
				}
				tCtx.Cleanup(func() {
					if err := l.Close(); err != nil {
						tCtx.Errorf("unexpected error closing listener: %v", err)
					}
				})
			}
		}()
	}
	wg.Wait()
}
