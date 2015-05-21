/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithm"
)

type Scheduler struct {
	Err     error
	Pod     *api.Pod
	Machine string
}

func (s *Scheduler) Schedule(pod *api.Pod, lister algorithm.MinionLister) (string, error) {
	s.Pod = pod
	return s.Machine, s.Err
}
