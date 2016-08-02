/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	probTestContainerName       = "test-webserver"
	probTestInitialDelaySeconds = 15
)

var _ = framework.KubeDescribe("Probing container", func() {
	f := framework.NewDefaultFramework("container-probe")
	var podClient *framework.PodClient
	probe := webserverProbeBuilder{}

	BeforeEach(func() {
		podClient = f.PodClient()
	})

	It("with readiness probe should not be ready before initial delay and never restart [Conformance]", func() {
		p := podClient.Create(makePodSpec(probe.withInitialDelay().build(), nil))
		f.WaitForPodReady(p.Name)

		p, err := podClient.Get(p.Name)
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
		p := podClient.Create(makePodSpec(probe.withFailing().build(), nil))
		Consistently(func() (bool, error) {
			p, err := podClient.Get(p.Name)
			if err != nil {
				return false, err
			}
			return api.IsPodReady(p), nil
		}, 1*time.Minute, 1*time.Second).ShouldNot(BeTrue(), "pod should not be ready")

		p, err := podClient.Get(p.Name)
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
		ObjectMeta: api.ObjectMeta{Name: "test-webserver-" + string(uuid.NewUUID())},
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
