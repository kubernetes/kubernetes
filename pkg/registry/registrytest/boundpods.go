/*
Copyright 2014 Google Inc. All rights reserved.

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

package registrytest

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type BoundPodsRegistry struct {
	Err         error
	BoundPods   []api.BoundPods
	Broadcaster *watch.Broadcaster

	sync.Mutex
}

func (r *BoundPodsRegistry) Get(ctx api.Context, nodeName string) (runtime.Object, error) {
	r.Lock()
	defer r.Unlock()
	for _, pods := range r.BoundPods {
		if pods.Host == nodeName {
			return &pods, r.Err
		}
	}
	return nil, r.Err
}

func (r *BoundPodsRegistry) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return r.Broadcaster.Watch(), nil
}
