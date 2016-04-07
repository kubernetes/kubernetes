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
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/gomega"
)

const (
	pollInterval = time.Second * 5
)

//One pod one container
type ConformanceContainer struct {
	pod       *api.Pod
	container *api.Container
	client    *client.Client
}

func NewConformanceContainer(c *client.Client, node string, container api.Container) *ConformanceContainer {
	return &ConformanceContainer{
		client:    c,
		container: &container,
		pod: &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      container.Name + "-" + string(util.NewUUID()),
				Namespace: api.NamespaceDefault,
			},
			Spec: api.PodSpec{
				NodeName:      node,
				RestartPolicy: api.RestartPolicyNever,
				Containers:    []api.Container{container},
			},
		},
	}
}

func (cc *ConformanceContainer) Create() error {
	_, err := cc.client.Pods(api.NamespaceDefault).Create(cc.pod)
	return err
}

func (cc *ConformanceContainer) Delete() error {
	return cc.client.Pods(api.NamespaceDefault).Delete(cc.pod.Name, &api.DeleteOptions{})
}

func (cc *ConformanceContainer) Status() (*api.ContainerStatus, error) {
	pod, err := cc.client.Pods(api.NamespaceDefault).Get(cc.pod.Name)
	if err != nil {
		return nil, err
	}

	if len(pod.Spec.Containers) != 1 {
		return nil, fmt.Errorf("unexpected containers %v", pod.Spec.Containers)
	}
	if len(pod.Status.ContainerStatuses) != 1 || pod.Status.ContainerStatuses[0].Name != cc.container.Name {
		return nil, fmt.Errorf("unexpected container statuses %v", pod.Status.ContainerStatuses)
	}
	return &pod.Status.ContainerStatuses[0], nil
}

func (cc *ConformanceContainer) Wait(timeout time.Duration, condition func(*api.ContainerStatus) bool) {
	Eventually(cc.check(condition), timeout, pollInterval).Should(BeTrue())
}

func (cc *ConformanceContainer) Always(timeout time.Duration, condition func(*api.ContainerStatus) bool) {
	Consistently(cc.check(condition), timeout, pollInterval).Should(BeTrue())
}

func (cc *ConformanceContainer) check(condition func(*api.ContainerStatus) bool) func() (bool, error) {
	return func() (bool, error) {
		status, err := cc.Status()
		if err != nil {
			return false, err
		}
		return condition(status), nil
	}
}

// TODO(random-liu): Add UpdateImage, Log, Exec, Attach

func isContainerSucceed(status *api.ContainerStatus) bool {
	return status.State.Terminated != nil && status.State.Terminated.ExitCode == 0
}
