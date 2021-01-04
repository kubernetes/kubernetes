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

	"k8s.io/client-go/rest"

	"k8s.io/client-go/controller"
)

// InformerInjector holds the type of a callback that attaches a particular
// informer type to a context.
type InformerInjector func(context.Context) (context.Context, controller.Informer)

func (i *impl) RegisterInformer(ii InformerInjector) {
	i.m.Lock()
	defer i.m.Unlock()

	i.informers = append(i.informers, ii)
}

func (i *impl) GetInformers() []InformerInjector {
	i.m.RLock()
	defer i.m.RUnlock()

	// Copy the slice before returning.
	return append(i.informers[:0:0], i.informers...)
}

func (i *impl) SetupInformers(ctx context.Context, cfg *rest.Config) (context.Context, []controller.Informer) {
	// Based on the reconcilers we have linked, build up a set of clients and inject
	// them onto the context.
	for _, ci := range i.GetClients() {
		ctx = ci(ctx, cfg)
	}

	// Based on the reconcilers we have linked, build up a set of informer factories
	// and inject them onto the context.
	for _, ifi := range i.GetInformerFactories() {
		ctx = ifi(ctx)
	}

	// Based on the reconcilers we have linked, build up a set of informers
	// and inject them onto the context.
	var inf controller.Informer
	informers := make([]controller.Informer, 0, len(i.GetInformers()))
	for _, ii := range i.GetInformers() {
		ctx, inf = ii(ctx)
		informers = append(informers, inf)
	}
	return ctx, informers
}
