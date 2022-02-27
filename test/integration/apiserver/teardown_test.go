/*
Copyright 2022 The Kubernetes Authors.

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

package apiserver

import (
	"math"
	"runtime"
	"testing"
	"time"

	apitesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestTestingAPIServerCleanTeardown(t *testing.T) {
	etcdConfig := framework.SharedEtcd()

	// Watch goroutines for 5 seconds before starting the server to get a high-water mark
	before := 0
	for i := 0; i < 500; i++ {
		if goroutines := runtime.NumGoroutine(); goroutines > before {
			before = goroutines
		}
		time.Sleep(10 * time.Millisecond)
	}
	beforeStack := make([]byte, 1024*1024)
	size := runtime.Stack(beforeStack, true)
	beforeStack = beforeStack[:size]

	s := apitesting.StartTestServerOrDie(t, nil, nil, etcdConfig)
	s.TearDownFn()

	// Watch goroutines for 5 seconds after stopping the server to get a low-water mark
	after := math.MaxInt32
	for i := 0; i < 500; i++ {
		if goroutines := runtime.NumGoroutine(); goroutines < after {
			after = goroutines
		}
		time.Sleep(10 * time.Millisecond)
	}

	if after > before {
		afterStack := make([]byte, 1024*1024)
		size := runtime.Stack(afterStack, true)
		afterStack = afterStack[:size]

		//t.Logf("before:\n%s", string(beforeStack))
		t.Logf("after:\n%s", string(afterStack))
		t.Errorf("expected %d or fewer, got %d", before, after)
	} else {
		t.Logf("expected %d or fewer, got %d", before, after)
	}
}

func TestFrameworkAPIServerCleanTeardown(t *testing.T) {
	// Watch goroutines for 5 seconds before starting the server to get a high-water mark
	before := 0
	for i := 0; i < 500; i++ {
		if goroutines := runtime.NumGoroutine(); goroutines > before {
			before = goroutines
		}
		time.Sleep(10 * time.Millisecond)
	}
	beforeStack := make([]byte, 1024*1024)
	size := runtime.Stack(beforeStack, true)
	beforeStack = beforeStack[:size]

	controlPlaneConfig := framework.NewIntegrationTestControlPlaneConfig()
	_, _, closeFn := framework.RunAnAPIServer(controlPlaneConfig)
	closeFn()

	// Watch goroutines for 5 seconds after stopping the server to get a low-water mark
	after := math.MaxInt32
	for i := 0; i < 500; i++ {
		if goroutines := runtime.NumGoroutine(); goroutines < after {
			after = goroutines
		}
		time.Sleep(10 * time.Millisecond)
	}

	if after > before {
		afterStack := make([]byte, 1024*1024)
		size := runtime.Stack(afterStack, true)
		afterStack = afterStack[:size]

		//t.Logf("before:\n%s", string(beforeStack))
		t.Logf("after:\n%s", string(afterStack))
		t.Errorf("expected %d or fewer, got %d", before, after)
	} else {
		t.Logf("expected %d or fewer, got %d", before, after)
	}
}
