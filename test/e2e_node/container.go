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

package e2e_node

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"
)

// One pod one container
type ConformanceContainer struct {
	Container        api.Container
	Client           *client.Client
	RestartPolicy    api.RestartPolicy
	Volumes          []api.Volume
	ImagePullSecrets []string
	NodeName         string
	Namespace        string

	podName string
}

func (cc *ConformanceContainer) Create() error {
	cc.podName = cc.Container.Name + string(util.NewUUID())
	imagePullSecrets := []api.LocalObjectReference{}
	for _, s := range cc.ImagePullSecrets {
		imagePullSecrets = append(imagePullSecrets, api.LocalObjectReference{Name: s})
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      cc.podName,
			Namespace: cc.Namespace,
		},
		Spec: api.PodSpec{
			NodeName:      cc.NodeName,
			RestartPolicy: cc.RestartPolicy,
			Containers: []api.Container{
				cc.Container,
			},
			Volumes:          cc.Volumes,
			ImagePullSecrets: imagePullSecrets,
		},
	}

	_, err := cc.Client.Pods(cc.Namespace).Create(pod)
	return err
}

func (cc *ConformanceContainer) Delete() error {
	return cc.Client.Pods(cc.Namespace).Delete(cc.podName, api.NewDeleteOptions(0))
}

func (cc *ConformanceContainer) IsReady() (bool, error) {
	pod, err := cc.Client.Pods(cc.Namespace).Get(cc.podName)
	if err != nil {
		return false, err
	}
	return api.IsPodReady(pod), nil
}

func (cc *ConformanceContainer) GetPhase() (api.PodPhase, error) {
	pod, err := cc.Client.Pods(cc.Namespace).Get(cc.podName)
	if err != nil {
		return api.PodUnknown, err
	}
	return pod.Status.Phase, nil
}

func (cc *ConformanceContainer) GetStatus() (api.ContainerStatus, error) {
	pod, err := cc.Client.Pods(cc.Namespace).Get(cc.podName)
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
	_, err := cc.Client.Pods(cc.Namespace).Get(cc.podName)
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
