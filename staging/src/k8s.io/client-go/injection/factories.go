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
)

// InformerFactoryInjector holds the type of a callback that attaches a particular
// factory type to a context.
type InformerFactoryInjector func(context.Context) context.Context

func (i *impl) RegisterInformerFactory(ifi InformerFactoryInjector) {
	i.m.Lock()
	defer i.m.Unlock()

	i.factories = append(i.factories, ifi)
}

func (i *impl) GetInformerFactories() []InformerFactoryInjector {
	i.m.RLock()
	defer i.m.RUnlock()

	// Copy the slice before returning.
	return append(i.factories[:0:0], i.factories...)
}
