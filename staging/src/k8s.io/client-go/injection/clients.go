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
)

// ClientInjector holds the type of a callback that attaches a particular
// client type to a context.
type ClientInjector func(context.Context, *rest.Config) context.Context

func (i *impl) RegisterClient(ci ClientInjector) {
	i.m.Lock()
	defer i.m.Unlock()

	i.clients = append(i.clients, ci)
}

func (i *impl) GetClients() []ClientInjector {
	i.m.RLock()
	defer i.m.RUnlock()

	// Copy the slice before returning.
	return append(i.clients[:0:0], i.clients...)
}
