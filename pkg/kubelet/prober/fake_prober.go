/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package prober

import (
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/probe"
)

var _ Prober = FakeProber{}

type FakeProber struct {
	Readiness probe.Result
	Liveness  probe.Result
	Error     error
}

func (f FakeProber) ProbeLiveness(_ *api.Pod, _ api.PodStatus, c api.Container, _ kubecontainer.ContainerID, _ int64) (probe.Result, error) {
	if c.LivenessProbe == nil {
		return probe.Success, nil
	}
	return f.Liveness, f.Error
}

func (f FakeProber) ProbeReadiness(_ *api.Pod, _ api.PodStatus, c api.Container, _ kubecontainer.ContainerID) (probe.Result, error) {
	if c.ReadinessProbe == nil {
		return probe.Success, nil
	}
	return f.Readiness, f.Error
}
