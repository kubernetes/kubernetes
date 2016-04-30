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

package e2e

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	probTestContainerName       = "test-webserber"
	probTestInitialDelaySeconds = 30
)

var _ = framework.KubeDescribe("Probing container", func() {
	f := framework.NewDefaultFramework("container-probe")
	var podClient client.PodInterface
	probe := webserverProbeBuilder{}

	BeforeEach(func() {
		podClient = f.Client.Pods(f.Namespace.Name)
	})

	It("with readiness probe should not be ready before initial delay and never restart [Conformance]", func() {
		p, err := podClient.Create(makePodSpec(probe.withInitialDelay().build(), nil))
		framework.ExpectNoError(err)

		Expect(wait.Poll(framework.Poll, 240*time.Second, func() (bool, error) {
			p, err := podClient.Get(p.Name)
			if err != nil {
				return false, err
			}
			ready := api.IsPodReady(p)
			if !ready {
				framework.Logf("pod is not yet ready; pod has phase %q.", p.Status.Phase)
				return false, nil
			}
			return true, nil
		})).NotTo(HaveOccurred(), "pod never became ready")

		p, err = podClient.Get(p.Name)
		framework.ExpectNoError(err)
		isReady, err := framework.PodRunningReady(p)
		framework.ExpectNoError(err)
		Expect(isReady).To(BeTrue(), "pod should be ready")

		// We assume the pod became ready when the container became ready. This
		// is true for a single container pod.
		readyTime, err := getTransitionTimeForReadyCondition(p)
		framework.ExpectNoError(err)
		startedTime, err := getContainerStartedTime(p, probTestContainerName)
		framework.ExpectNoError(err)

		framework.Logf("Container started at %v, pod became ready at %v", startedTime, readyTime)
		initialDelay := probTestInitialDelaySeconds * time.Second
		if readyTime.Sub(startedTime) < initialDelay {
			framework.Failf("Pod became ready before it's %v initial delay", initialDelay)
		}

		restartCount := getRestartCount(p)
		Expect(restartCount == 0).To(BeTrue(), "pod should have a restart count of 0 but got %v", restartCount)
	})

	It("with readiness probe that fails should never be ready and never restart [Conformance]", func() {
		p, err := podClient.Create(makePodSpec(probe.withFailing().build(), nil))
		framework.ExpectNoError(err)

		err = wait.Poll(framework.Poll, 180*time.Second, func() (bool, error) {
			p, err := podClient.Get(p.Name)
			if err != nil {
				return false, err
			}
			return api.IsPodReady(p), nil
		})
		if err != wait.ErrWaitTimeout {
			framework.Failf("expecting wait timeout error but got: %v", err)
		}

		p, err = podClient.Get(p.Name)
		framework.ExpectNoError(err)

		isReady, err := framework.PodRunningReady(p)
		Expect(isReady).NotTo(BeTrue(), "pod should be not ready")

		restartCount := getRestartCount(p)
		Expect(restartCount == 0).To(BeTrue(), "pod should have a restart count of 0 but got %v", restartCount)
	})

})

func getContainerStartedTime(p *api.Pod, containerName string) (time.Time, error) {
	for _, status := range p.Status.ContainerStatuses {
		if status.Name != containerName {
			continue
		}
		if status.State.Running == nil {
			return time.Time{}, fmt.Errorf("Container is not running")
		}
		return status.State.Running.StartedAt.Time, nil
	}
	return time.Time{}, fmt.Errorf("cannot find container named %q", containerName)
}

func getTransitionTimeForReadyCondition(p *api.Pod) (time.Time, error) {
	for _, cond := range p.Status.Conditions {
		if cond.Type == api.PodReady {
			return cond.LastTransitionTime.Time, nil
		}
	}
	return time.Time{}, fmt.Errorf("No ready condition can be found for pod")
}

func getRestartCount(p *api.Pod) int {
	count := 0
	for _, containerStatus := range p.Status.ContainerStatuses {
		count += int(containerStatus.RestartCount)
	}
	return count
}

func makePodSpec(readinessProbe, livenessProbe *api.Probe) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "test-webserver-" + string(util.NewUUID())},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:           probTestContainerName,
					Image:          "gcr.io/google_containers/test-webserver:e2e",
					LivenessProbe:  livenessProbe,
					ReadinessProbe: readinessProbe,
				},
			},
		},
	}
	return pod
}

type webserverProbeBuilder struct {
	failing      bool
	initialDelay bool
}

func (b webserverProbeBuilder) withFailing() webserverProbeBuilder {
	b.failing = true
	return b
}

func (b webserverProbeBuilder) withInitialDelay() webserverProbeBuilder {
	b.initialDelay = true
	return b
}

func (b webserverProbeBuilder) build() *api.Probe {
	probe := &api.Probe{
		Handler: api.Handler{
			HTTPGet: &api.HTTPGetAction{
				Port: intstr.FromInt(80),
				Path: "/",
			},
		},
	}
	if b.initialDelay {
		probe.InitialDelaySeconds = probTestInitialDelaySeconds
	}
	if b.failing {
		probe.HTTPGet.Port = intstr.FromInt(81)
	}
	return probe
}
