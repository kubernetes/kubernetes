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

package common

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// ContainerStatusRetryTimeout specifies the timeout to check container status.
	ContainerStatusRetryTimeout = time.Minute * 5
	// ContainerStatusPollInterval specifies the polling interval to check container status.
	ContainerStatusPollInterval = time.Second * 1
)

// ConformanceContainer specifies One pod one container.
type ConformanceContainer struct {
	Container        v1.Container
	RestartPolicy    v1.RestartPolicy
	Volumes          []v1.Volume
	ImagePullSecrets []string

	PodClient          *framework.PodClient
	podName            string
	PodSecurityContext *v1.PodSecurityContext
}

// Create is a function that creates a ConformanceContainer.
func (cc *ConformanceContainer) Create() {
	cc.podName = cc.Container.Name + string(uuid.NewUUID())
	imagePullSecrets := []v1.LocalObjectReference{}
	for _, s := range cc.ImagePullSecrets {
		imagePullSecrets = append(imagePullSecrets, v1.LocalObjectReference{Name: s})
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: cc.podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: cc.RestartPolicy,
			Containers: []v1.Container{
				cc.Container,
			},
			SecurityContext:  cc.PodSecurityContext,
			Volumes:          cc.Volumes,
			ImagePullSecrets: imagePullSecrets,
		},
	}
	cc.PodClient.Create(pod)
}

// Delete is a function that deletes a ConformanceContainer.
func (cc *ConformanceContainer) Delete() error {
	return cc.PodClient.Delete(context.TODO(), cc.podName, *metav1.NewDeleteOptions(0))
}

// IsReady is a function that checks a ConformanceContainer status and returns true if a ConformanceContainer is ready.
func (cc *ConformanceContainer) IsReady() (bool, error) {
	pod, err := cc.PodClient.Get(context.TODO(), cc.podName, metav1.GetOptions{})
	if err != nil {
		return false, err
	}
	return podutil.IsPodReady(pod), nil
}

// GetPhase is a function that gets a ConformanceContainer phase.
func (cc *ConformanceContainer) GetPhase() (v1.PodPhase, error) {
	pod, err := cc.PodClient.Get(context.TODO(), cc.podName, metav1.GetOptions{})
	if err != nil {
		return v1.PodUnknown, err
	}
	return pod.Status.Phase, nil
}

// GetStatus is a function that gets a ConformanceContainer status.
func (cc *ConformanceContainer) GetStatus() (v1.ContainerStatus, error) {
	pod, err := cc.PodClient.Get(context.TODO(), cc.podName, metav1.GetOptions{})
	if err != nil {
		return v1.ContainerStatus{}, err
	}
	statuses := pod.Status.ContainerStatuses
	if len(statuses) != 1 || statuses[0].Name != cc.Container.Name {
		return v1.ContainerStatus{}, fmt.Errorf("unexpected container statuses %v", statuses)
	}
	return statuses[0], nil
}

// Present is a function that retruns true if a ConformanceContainer exist.
func (cc *ConformanceContainer) Present() (bool, error) {
	_, err := cc.PodClient.Get(context.TODO(), cc.podName, metav1.GetOptions{})
	if err == nil {
		return true, nil
	}
	if apierrors.IsNotFound(err) {
		return false, nil
	}
	return false, err
}

// ContainerState specifies container state.
type ContainerState string

const (
	// ContainerStateWaiting specifies container is waiting.
	ContainerStateWaiting ContainerState = "Waiting"
	// ContainerStateRunning specifies container is running.
	ContainerStateRunning ContainerState = "Running"
	// ContainerStateTerminated specifies container is terminated.
	ContainerStateTerminated ContainerState = "Terminated"
	// ContainerStateUnknown specifies container is unknown.
	ContainerStateUnknown ContainerState = "Unknown"
)

// GetContainerState is a function that converts v1.ContainerState to ContainerState.
func GetContainerState(state v1.ContainerState) ContainerState {
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
