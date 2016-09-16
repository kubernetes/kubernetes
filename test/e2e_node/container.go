/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_node

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
)

// One pod one container
// TODO: This should be migrated to the e2e framework.
type ConformanceContainer struct {
	Container        api.Container
	RestartPolicy    api.RestartPolicy
	Volumes          []api.Volume
	ImagePullSecrets []string

	PodClient          *framework.PodClient
	podName            string
	PodSecurityContext *api.PodSecurityContext
}

func (cc *ConformanceContainer) Create() {
	cc.podName = cc.Container.Name + string(uuid.NewUUID())
	imagePullSecrets := []api.LocalObjectReference{}
	for _, s := range cc.ImagePullSecrets {
		imagePullSecrets = append(imagePullSecrets, api.LocalObjectReference{Name: s})
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: cc.podName,
		},
		Spec: api.PodSpec{
			RestartPolicy: cc.RestartPolicy,
			Containers: []api.Container{
				cc.Container,
			},
			SecurityContext:  cc.PodSecurityContext,
			Volumes:          cc.Volumes,
			ImagePullSecrets: imagePullSecrets,
		},
	}
	cc.PodClient.Create(pod)
}

func (cc *ConformanceContainer) Delete() error {
	return cc.PodClient.Delete(cc.podName, api.NewDeleteOptions(0))
}

func (cc *ConformanceContainer) IsReady() (bool, error) {
	pod, err := cc.PodClient.Get(cc.podName)
	if err != nil {
		return false, err
	}
	return api.IsPodReady(pod), nil
}

func (cc *ConformanceContainer) GetPhase() (api.PodPhase, error) {
	pod, err := cc.PodClient.Get(cc.podName)
	if err != nil {
		return api.PodUnknown, err
	}
	return pod.Status.Phase, nil
}

func (cc *ConformanceContainer) GetStatus() (api.ContainerStatus, error) {
	pod, err := cc.PodClient.Get(cc.podName)
	if err != nil {
		return api.ContainerStatus{}, err
	}
	statuses := pod.Status.ContainerStatuses
	if len(statuses) != 1 || statuses[0].Name != cc.Container.Name {
		return api.ContainerStatus{}, fmt.Errorf("unexpected container statuses %v", statuses)
	}
	return statuses[0], nil
}

func (cc *ConformanceContainer) Present() (bool, error) {
	_, err := cc.PodClient.Get(cc.podName)
	if err == nil {
		return true, nil
	}
	if errors.IsNotFound(err) {
		return false, nil
	}
	return false, err
}

type ContainerState int

const (
	ContainerStateWaiting ContainerState = iota
	ContainerStateRunning
	ContainerStateTerminated
	ContainerStateUnknown
)

func GetContainerState(state api.ContainerState) ContainerState {
	if state.Waiting != nil {
		return ContainerStateWaiting
	}
	if state.Running != nil {
		return ContainerStateRunning
	}
	if state.Terminated != nil {
		return ContainerStateTerminated
	}
	return ContainerStateUnknown
}
