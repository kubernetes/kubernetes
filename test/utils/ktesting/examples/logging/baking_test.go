//go:build example

/*
Copyright 2023 The Kubernetes Authors.

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

package logging

import (
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// heatOven demonstrates how log output and failures are handle when using
// tCtx.Fatal to report a test failure.
func heatOven(tCtx ktesting.TContext) {
	tCtx.Log("Log()")
	tCtx.Logger().Info("Logger().Info()")
	tCtx.Fatal("oven not found")
}

// turnOffOven demonstrates a Gomega assertion failure, in this example
// triggered as part of test cleanup via CleanupCtx.
func turnOffOven(tCtx ktesting.TContext) {
	tCtx.Expect(false).To(gomega.BeTrueBecause("turning off oven not implemented"))
}
