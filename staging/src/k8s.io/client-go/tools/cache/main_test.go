/*
Copyright 2018 The Kubernetes Authors.

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

package cache

import (
	"testing"

	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	options := []goleak.Option{
		// These tests run goroutines which get stuck in Pop.
		// This cannot be fixed without modifying the API.
		goleak.IgnoreAnyFunction("k8s.io/client-go/tools/cache.TestFIFO_addReplace.func1"),
		goleak.IgnoreAnyFunction("k8s.io/client-go/tools/cache.TestFIFO_addUpdate.func1"),
		goleak.IgnoreAnyFunction("k8s.io/client-go/tools/cache.TestDeltaFIFO_addReplace.func1"),
		goleak.IgnoreAnyFunction("k8s.io/client-go/tools/cache.TestDeltaFIFO_addUpdate.func1"),

		// TODO: fix the following tests by adding WithContext APIs and cancellation.
		goleak.IgnoreAnyFunction("k8s.io/client-go/tools/cache.TestTransformingInformerRace.func3"),
		// Created by k8s.io/client-go/tools/cache.TestReflectorListAndWatch, cannot filter on that (https://github.com/uber-go/goleak/issues/135):
		goleak.IgnoreAnyFunction("k8s.io/client-go/tools/cache.(*Reflector).ListAndWatch"),
		goleak.IgnoreAnyFunction("k8s.io/client-go/tools/cache.(*Reflector).startResync"),
		// ???
		goleak.IgnoreAnyFunction("k8s.io/client-go/tools/cache.(*DeltaFIFO).Close"),
	}
	goleak.VerifyTestMain(m, options...)
}
