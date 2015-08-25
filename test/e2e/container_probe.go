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
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Probing container", func() {
	framework := Framework{BaseName: "container-probe"}
	var podClient client.PodInterface
	probe := webserverProbeBuilder{}

	BeforeEach(func() {
		framework.beforeEach()
		podClient = framework.Client.Pods(framework.Namespace.Name)
	})

	AfterEach(framework.afterEach)

	It("with readiness probe should not be ready before initial delay and never restart", func() {
		p, err := podClient.Create(makePodSpec(probe.withInitialDelay().build(), nil))
		expectNoError(err)
		startTime := time.Now()

		Expect(wait.Poll(poll, 90*time.Second, func() (bool, error) {
			p, err := podClient.Get(p.Name)
			if err != nil {
				return false, err
			}
			ready := api.IsPodReady(p)
			if !ready {
				Logf("pod is not yet ready; pod has phase %q.", p.Status.Phase)
				return false, nil
			}
			return true, nil
		})).NotTo(HaveOccurred(), "pod never became ready")

		if time.Since(startTime) < 30*time.Second {
			Failf("Pod became ready before it's initial delay")
		}

		p, err = podClient.Get(p.Name)
		expectNoError(err)

		isReady, err := podRunningReady(p)
		expectNoError(err)
		Expect(isReady).To(BeTrue(), "pod should be ready")

		restartCount := getRestartCount(p)
		Expect(restartCount == 0).To(BeTrue(), "pod should have a restart count of 0 but got %v", restartCount)
	})

	It("with readiness probe that fails should never be ready and never restart", func() {
		p, err := podClient.Create(makePodSpec(probe.withFailing().build(), nil))
		expectNoError(err)

		err = wait.Poll(poll, 90*time.Second, func() (bool, error) {
			p, err := podClient.Get(p.Name)
			if err != nil {
				return false, err
			}
			return api.IsPodReady(p), nil
		})
		if err != wait.ErrWaitTimeout {
			Failf("expecting wait timeout error but got: %v", err)
		}

		p, err = podClient.Get(p.Name)
		expectNoError(err)

		isReady, err := podRunningReady(p)
		Expect(isReady).NotTo(BeTrue(), "pod should be not ready")

		restartCount := getRestartCount(p)
		Expect(restartCount == 0).To(BeTrue(), "pod should have a restart count of 0 but got %v", restartCount)
	})

})

func getRestartCount(p *api.Pod) int {
	count := 0
	for _, containerStatus := range p.Status.ContainerStatuses {
		count += containerStatus.RestartCount
	}
	return count
}

func makePodSpec(readinessProbe, livenessProbe *api.Probe) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "test-webserver-" + string(util.NewUUID())},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:           "test-webserver",
					Image:          "gcr.io/google_containers/test-webserver",
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
				Port: util.NewIntOrStringFromInt(80),
				Path: "/",
			},
		},
	}
	if b.initialDelay {
		probe.InitialDelaySeconds = 30
	}
	if b.failing {
		probe.HTTPGet.Port = util.NewIntOrStringFromInt(81)
	}
	return probe
}
