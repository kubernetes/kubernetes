/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package container

import (
	"time"

	"k8s.io/kubernetes/pkg/types"
)

type fakeCache struct {
	runtime Runtime
}

func NewFakeCache(runtime Runtime) Cache {
	return &fakeCache{runtime: runtime}
}

func (c *fakeCache) Get(id types.UID) (*PodStatus, error) {
	return c.runtime.GetPodStatus(id, "", "")
}

func (c *fakeCache) GetNewerThan(id types.UID, minTime time.Time) (*PodStatus, error) {
	return c.Get(id)
}

func (c *fakeCache) Set(id types.UID, status *PodStatus, err error, timestamp time.Time) {
}

func (c *fakeCache) Delete(id types.UID) {
}

func (c *fakeCache) UpdateTime(_ time.Time) {
}
