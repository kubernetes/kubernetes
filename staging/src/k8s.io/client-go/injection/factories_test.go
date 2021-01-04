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
)

func injectFooFactory(ctx context.Context) context.Context {
	return ctx
}

func injectBarFactory(ctx context.Context) context.Context {
	return ctx
}

func TestRegisterInformerFactory(t *testing.T) {
	i := &impl{}

	if want, got := 0, len(i.GetInformerFactories()); got != want {
		t.Errorf("GetInformerFactories() = %d, wanted %d", want, got)
	}

	i.RegisterInformerFactory(injectFooFactory)

	if want, got := 1, len(i.GetInformerFactories()); got != want {
		t.Errorf("GetInformerFactories() = %d, wanted %d", want, got)
	}

	i.RegisterInformerFactory(injectBarFactory)

	if want, got := 2, len(i.GetInformerFactories()); got != want {
		t.Errorf("GetInformerFactories() = %d, wanted %d", want, got)
	}
}
