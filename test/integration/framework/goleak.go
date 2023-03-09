/*
Copyright 2017 The Kubernetes Authors.

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

package framework

import (
	"testing"
	"time"

	"go.uber.org/goleak"
	"k8s.io/apiserver/pkg/server/healthz"
)

// IgnoreBackgroundGoroutines returns options for goleak.Find
// which ignore goroutines created by "go test" and init functions,
// like the one from go.opencensus.io/stats/view/worker.go.
//
// Goroutines that normally get created later when using the apiserver
// get created already when calling this function, therefore they
// also get ignored.
func IgnoreBackgroundGoroutines() []goleak.Option {
	// Ensure that on-demand goroutines are running.
	_ = healthz.LogHealthz.Check(nil)

	return []goleak.Option{goleak.IgnoreCurrent()}
}

// GoleakCheck sets up leak checking for a test or benchmark.
// The check runs as cleanup operation and records an
// error when goroutines were leaked.
func GoleakCheck(tb testing.TB, opts ...goleak.Option) {
	// Must be called *before* creating new goroutines.
	opts = append(opts, IgnoreBackgroundGoroutines()...)

	tb.Cleanup(func() {
		if err := goleakFindRetry(opts...); err != nil {
			tb.Error(err.Error())
		}
	})
}

func goleakFindRetry(opts ...goleak.Option) error {
	// Several tests don't wait for goroutines to stop. goleak.Find retries
	// internally, but not long enough. 5 seconds seemed to be enough for
	// most tests, even when testing in the CI.
	timeout := 5 * time.Second
	start := time.Now()
	for {
		err := goleak.Find(opts...)
		if err == nil {
			return nil
		}
		if time.Now().Sub(start) >= timeout {
			return err
		}
	}
}
