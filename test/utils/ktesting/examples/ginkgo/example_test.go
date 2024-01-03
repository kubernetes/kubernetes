//go:build example
// +build example

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

package ginkgo_test

import (
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/test/utils/ktesting"
)

var _ = ginkgo.Describe("something", func() {
	ginkgo.It("fails", func(ctx context.Context) {
		tCtx := ktesting.InitCtx(ctx, ginkgo.GinkgoT(1))
		doSomething(ktesting.WithStep(tCtx, "trying something"))
	})

	// Ginkgo produces its own [TIMEDOUT] failure in this case.
	// The step gets injected by stepReporter.AttachProgressReporter.
	ginkgo.It("spec timeout", func(ctx context.Context) {
		tCtx := ktesting.InitCtx(ctx, ginkgo.GinkgoT(1))
		timeout(ktesting.WithStep(tCtx, "trying something"))
	}, ginkgo.SpecTimeout(30*time.Second))

	// Here Gomega produces the failure.
	// The step gets injected by stepContext.Fatal.
	ginkgo.It("canceled", func(ctx context.Context) {
		tCtx := ktesting.InitCtx(ctx, ginkgo.GinkgoT(1))
		tCtx = ktesting.WithTimeout(tCtx, 30*time.Second, "test timeout")
		timeout(ktesting.WithStep(tCtx, "trying something"))
	})
})

func doSomething(tCtx ktesting.TContext) {
	tCtx.Helper() // Failures are recorded for the caller.
	tCtx.Log("Log()")
	tCtx.Logger().Info("Logger().Info()")
	tCtx.Fatal("this failure is intentional")
	require.Equal(tCtx, 1, 2) // not reached
}

func timeout(tCtx ktesting.TContext) {
	tCtx.Helper() // Failures are recorded for the caller.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) int {
		select {
		case <-tCtx.Done():
		case <-time.After(time.Second):
		}
		return 42
	}).Should(gomega.Equal(1))
}
