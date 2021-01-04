/*
Copyright 2021 The Kubernetes Authors.

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

package injection

import (
	"context"
	"testing"

	"k8s.io/client-go/controller"
	"k8s.io/client-go/rest"
)

type fakeInformer struct{}

// HasSynced implements controller.Informer
func (*fakeInformer) HasSynced() bool {
	return false
}

// Run implements controller.Informer
func (*fakeInformer) Run(<-chan struct{}) {}

var _ controller.Informer = (*fakeInformer)(nil)

func injectFooInformer(ctx context.Context) (context.Context, controller.Informer) {
	return ctx, nil
}

func injectBarInformer(ctx context.Context) (context.Context, controller.Informer) {
	return ctx, nil
}

func TestRegisterInformersAndSetup(t *testing.T) {
	i := &impl{}

	if want, got := 0, len(i.GetInformers()); got != want {
		t.Errorf("GetInformerFactories() = %d, wanted %d", want, got)
	}

	i.RegisterClient(injectFoo)
	i.RegisterClient(injectBar)

	i.RegisterInformerFactory(injectFooFactory)
	i.RegisterInformerFactory(injectBarFactory)

	i.RegisterInformer(injectFooInformer)
	i.RegisterInformer(injectBarInformer)

	_, infs := i.SetupInformers(context.Background(), &rest.Config{})

	if want, got := 2, len(infs); got != want {
		t.Errorf("SetupInformers() = %d, wanted %d", want, got)
	}
}
