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
	"errors"
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/types"
)

// One pod one container
type ConformanceContainer struct {
	Container     api.Container
	Client        *client.Client
	RestartPolicy api.RestartPolicy
	Volumes       []api.Volume
	NodeName      string
	Namespace     string

	podName string
}

type ConformanceContainerEqualMatcher struct {
	Expected interface{}
}

func CContainerEqual(expected interface{}) types.GomegaMatcher {
	return &ConformanceContainerEqualMatcher{
		Expected: expected,
	}
}

func (matcher *ConformanceContainerEqualMatcher) Match(actual interface{}) (bool, error) {
	if actual == nil && matcher.Expected == nil {
		return false, fmt.Errorf("Refusing to compare <nil> to <nil>.\nBe explicit and use BeNil() instead.  This is to avoid mistakes where both sides of an assertion are erroneously uninitialized.")
	}
	val := api.Semantic.DeepDerivative(matcher.Expected, actual)
	return val, nil
}

func (matcher *ConformanceContainerEqualMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to equal", matcher.Expected)
}

func (matcher *ConformanceContainerEqualMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to equal", matcher.Expected)
}

func (cc *ConformanceContainer) Create() error {
	cc.podName = cc.Container.Name + string(util.NewUUID())
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
			Volumes: cc.Volumes,
		},
	}

	_, err := cc.Client.Pods(cc.Namespace).Create(pod)
	return err
}

//Same with 'delete'
func (cc *ConformanceContainer) Stop() error {
	return cc.Client.Pods(cc.Namespace).Delete(cc.podName, &api.DeleteOptions{})
}

func (cc *ConformanceContainer) Delete() error {
	return cc.Client.Pods(cc.Namespace).Delete(cc.podName, &api.DeleteOptions{})
}

func (cc *ConformanceContainer) Get() (ConformanceContainer, error) {
	pod, err := cc.Client.Pods(cc.Namespace).Get(cc.podName)
	if err != nil {
		return ConformanceContainer{}, err
	}

	containers := pod.Spec.Containers
	if containers == nil || len(containers) != 1 {
		return ConformanceContainer{}, errors.New("Failed to get container")
	}
	return ConformanceContainer{containers[0], cc.Client, pod.Spec.RestartPolicy, pod.Spec.Volumes, pod.Spec.NodeName, cc.Namespace, cc.podName}, nil
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
	if apierrs.IsNotFound(err) {
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
