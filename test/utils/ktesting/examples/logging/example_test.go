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

// The tests below will fail and therefore are excluded from
// normal "make test" via the "example" build tag. To run
// the tests and check the output, use "go test -tags example ."

import (
	"flag"
	"fmt"
	"testing"

	"github.com/onsi/gomega"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestError(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Error("some", "thing")
}

func TestErrorf(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Errorf("some %s", "thing")
}

func TestFatal(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Fatal("some", "thing")
	tCtx.Log("not reached")
}

func TestFatalf(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Fatalf("some %s", "thing")
	tCtx.Log("not reached")
}

func TestFormat(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Logf("hello via tCtx.Logf (unstructured logging): x is %d", 1)
	tCtx.Logger().Info("hello via tCtx.Logger().Info (structured logging)", "x", 1)
	tCtx.Error("some", "thing")
}

func TestVerbosity(t *testing.T) {
	tCtx := ktesting.Init(t)

	var fs flag.FlagSet
	klog.InitFlags(&fs)
	tCtx.Logf("klog verbosity: %s", fs.Lookup("v").Value.String())
	tCtx.Logger().V(1).Info("V=1")
	tCtx.Logger().V(2).Info("V=2")
}

// TestWithStep demonstrates how the caller can supply additional context
// for log messages and failure texts to helper code which directly
// records test failures.
func TestWithStep(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) { turnOffOven(tCtx) })
	bake(tCtx.WithStep("bake cake"))
}

func bake(tCtx ktesting.TContext) {
	heatOven(tCtx.WithStep("set heat for baking"))
}

// TestWithError demonstrates the "return error" approach for helper code
// It uses the same heatOven helper as TestWithStep.
func TestWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) { turnOffOven(tCtx) })
	tCtx.AssertNoError(checkTemperature(tCtx, 42), "checking oven temperature")
	tCtx.AssertNoError(checkOvenReady(tCtx, false), "checking oven readiness")
	tCtx.AssertNoError(bakeCake(tCtx), "baking cake")
}

// checkTemperature demonstrates capturing a failure manually via TContext.
func checkTemperature(tCtx ktesting.TContext, temperature int) error {
	if temperature < 200 {
		return ktesting.NewFailure(fmt.Sprintf("oven temperature %d°C is too low for baking", temperature))
	}
	return nil
}

// checkOvenReady demonstrates capturing a failure via Gomega.
func checkOvenReady(tCtx ktesting.TContext, ready bool) (finalErr error) {
	tCtx, finalize := tCtx.WithError(&finalErr)
	defer finalize()

	tCtx.Expect(ready).To(gomega.BeTrueBecause("oven is not ready yet"))
	return nil
}

func bakeCake(tCtx ktesting.TContext) (finalErr error) {
	tCtx, finalize := tCtx.WithError(&finalErr)
	defer finalize()

	heatOven(tCtx)
	return nil
}
