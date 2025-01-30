/*
Copyright 2015 The Kubernetes Authors.

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

// Package init installs GrabBeforeEach and GrabAfterEach as callbacks
// for gathering data before and after a test.
package init

import (
	"context"

	"github.com/onsi/ginkgo/v2"

	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
)

func init() {
	framework.NewFrameworkExtensions = append(framework.NewFrameworkExtensions,
		func(f *framework.Framework) {
			ginkgo.BeforeEach(func(ctx context.Context) {
				metrics := e2emetrics.GrabBeforeEach(ctx, f)
				ginkgo.DeferCleanup(e2emetrics.GrabAfterEach, f, metrics)
			})
		},
	)
}
