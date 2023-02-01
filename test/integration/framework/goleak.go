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
