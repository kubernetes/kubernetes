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

package ktesting_test

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestCancelManual(t *testing.T) {
	tCtx := ktesting.Init(t)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// Blocks until tCtx.Cancel is called below.
		<-tCtx.Done()
	}()
	tCtx.Cancel("manually canceled")
	wg.Wait()
}

func TestCancelAutomatic(t *testing.T) {
	var wg sync.WaitGroup
	// This callback gets registered first and thus
	// gets invoked last.
	t.Cleanup(wg.Wait)
	tCtx := ktesting.Init(t)
	wg.Add(1)
	go func() {
		defer wg.Done()
		// Blocks until the context gets canceled automatically.
		<-tCtx.Done()
	}()
}

func TestCancelCtx(t *testing.T) {
	tCtx := ktesting.Init(t)
	var discardLogger klog.Logger
	tCtx = ktesting.WithLogger(tCtx, discardLogger)
	tCtx = ktesting.WithRESTConfig(tCtx, new(rest.Config))
	baseCtx := tCtx

	tCtx.Cleanup(func() {
		if tCtx.Err() == nil {
			t.Error("context should be canceled but isn't")
		}
	})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		if tCtx.Err() != nil {
			t.Errorf("context should not be canceled but is: %v", tCtx.Err())
		}
		assert.Equal(t, baseCtx.Logger(), tCtx.Logger(), "Logger()")
		assert.Equal(t, baseCtx.RESTConfig(), tCtx.RESTConfig(), "RESTConfig()")
		assert.Equal(t, baseCtx.RESTMapper(), tCtx.RESTMapper(), "RESTMapper()")
		assert.Equal(t, baseCtx.Client(), tCtx.Client(), "Client()")
		assert.Equal(t, baseCtx.Dynamic(), tCtx.Dynamic(), "Dynamic()")
		assert.Equal(t, baseCtx.APIExtensions(), tCtx.APIExtensions(), "APIExtensions()")
	})

	// Cancel, then let testing.T invoke test cleanup.
	tCtx.Cancel("test is complete")
}

func TestParallel(t *testing.T) {
	var wg sync.WaitGroup
	wg.Add(3)

	tCtx := ktesting.Init(t)

	// Each sub-test runs in parallel to the others and waits for the other two.
	test := func(tCtx ktesting.TContext) {
		tCtx.Parallel()
		wg.Done()
		wg.Wait()
	}
	tCtx.Run("one", test)
	tCtx.Run("two", test)
	tCtx.Run("three", test)
}

func TestWithTB(t *testing.T) {
	tCtx := ktesting.Init(t)

	cfg := new(rest.Config)
	mapper := new(restmapper.DeferredDiscoveryRESTMapper)
	client := clientset.New(nil)
	dynamic := dynamic.New(nil)
	apiextensions := apiextensions.New(nil)
	tCtx = ktesting.WithClients(tCtx, cfg, mapper, client, dynamic, apiextensions)

	t.Run("sub", func(t *testing.T) {
		tCtx := ktesting.WithTB(tCtx, t)

		assert.Equal(t, cfg, tCtx.RESTConfig(), "RESTConfig")
		assert.Equal(t, mapper, tCtx.RESTMapper(), "RESTMapper")
		assert.Equal(t, client, tCtx.Client(), "Client")
		assert.Equal(t, dynamic, tCtx.Dynamic(), "Dynamic")
		assert.Equal(t, apiextensions, tCtx.APIExtensions(), "APIExtensions")

		tCtx.Cancel("test is complete")
	})

	if err := tCtx.Err(); err != nil {
		t.Errorf("parent TContext should not have been cancelled: %v", err)
	}
}
