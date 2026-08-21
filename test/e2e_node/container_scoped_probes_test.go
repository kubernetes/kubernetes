//go:build linux

/*
Copyright The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

const (
	// readyFlagPath is touched and removed by test containers to drive their
	// own probes.
	readyFlagPath = "/tmp/ready"
)

// execProbe builds an exec probe running cmd.
func execProbe(cmd []string, initialDelay, period, failureThreshold int32) *v1.Probe {
	return &v1.Probe{
		ProbeHandler:        v1.ProbeHandler{Exec: &v1.ExecAction{Command: cmd}},
		InitialDelaySeconds: initialDelay,
		PeriodSeconds:       period,
		FailureThreshold:    failureThreshold,
	}
}

// flagProbe builds a probe that passes only while readyFlagPath exists.
func flagProbe(initialDelay, period, failureThreshold int32) *v1.Probe {
	return execProbe([]string{"test", "-f", readyFlagPath}, initialDelay, period, failureThreshold)
}

// containerStatusByName finds a container's status among a pod's regular and
// init container statuses.
func containerStatusByName(pod *v1.Pod, name string) *v1.ContainerStatus {
	for i, status := range pod.Status.ContainerStatuses {
		if status.Name == name {
			return &pod.Status.ContainerStatuses[i]
		}
	}
	for i, status := range pod.Status.InitContainerStatuses {
		if status.Name == name {
			return &pod.Status.InitContainerStatuses[i]
		}
	}
	return nil
}

func waitForPodReady(ctx context.Context, f *framework.Framework, pod *v1.Pod) {
	ginkgo.GinkgoHelper()
	err := e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "PodReady", f.Timeouts.PodStart,
		func(p *v1.Pod) (bool, error) {
			if p.Status.Phase != v1.PodRunning {
				return false, nil
			}
			for _, cond := range p.Status.Conditions {
				if cond.Type == v1.PodReady && cond.Status == v1.ConditionTrue {
					return true, nil
				}
			}
			return false, nil
		})
	framework.ExpectNoError(err)
}

var _ = SIGDescribe("Container Scoped Probes", framework.WithSerial(), framework.WithFeatureGate(features.ContainerScopedProbes), func() {
	f := framework.NewDefaultFramework("container-scoped-probes")
	addAfterEachForCleaningUpPods(f)
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
		if initialConfig.FeatureGates == nil {
			initialConfig.FeatureGates = map[string]bool{}
		}
		initialConfig.FeatureGates[string(features.ContainerScopedProbes)] = true
		// The two gates conflict: the container-scoped path implements the
		// deprecated gate's disabled semantics deterministically.
		initialConfig.FeatureGates[string(features.ChangeContainerStatusOnKubeletRestart)] = false
	})

	ginkgo.Context("across a kubelet restart", func() {
		// testStatusStableAcrossKubeletRestart runs the pod, waits for it to
		// settle into the state described by wantStarted/wantReady, then
		// restarts the kubelet while watching that the probed container's
		// reported state never budges.
		testStatusStableAcrossKubeletRestart := func(ctx context.Context, pod *v1.Pod, containerName string, wantStarted, wantReady bool) {
			client := e2epod.NewPodClient(f)
			pod = client.Create(ctx, pod)

			ginkgo.By("waiting for the probed container to reach its expected state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "ProbedContainerSettled", f.Timeouts.PodStart,
				func(p *v1.Pod) (bool, error) {
					status := containerStatusByName(p, containerName)
					if status == nil || status.State.Running == nil || status.Started == nil {
						return false, nil
					}
					return *status.Started == wantStarted && status.Ready == wantReady, nil
				})
			framework.ExpectNoError(err)

			stopCh := make(chan struct{})
			errCh := watchPodStatusDuringKubeletRestart(ctx, f, pod, stopCh, func(p *v1.Pod) error {
				status := containerStatusByName(p, containerName)
				if status == nil {
					return nil
				}
				if status.RestartCount > 0 {
					return fmt.Errorf("container %q restarted %d times across the kubelet restart", containerName, status.RestartCount)
				}
				if status.Started == nil || *status.Started != wantStarted {
					return fmt.Errorf("container %q Started = %v, want %v", containerName, status.Started, wantStarted)
				}
				if status.Ready != wantReady {
					return fmt.Errorf("container %q Ready = %v, want %v", containerName, status.Ready, wantReady)
				}
				return nil
			})

			ginkgo.By("restarting the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)
			restartKubelet(ctx)

			// Give the kubelet time to take the pod over and, if it is going to
			// get this wrong, to say so.
			time.Sleep(15 * time.Second)
			close(stopCh)

			for err := range errCh {
				framework.ExpectNoError(err, "pod status changed across the kubelet restart")
			}
		}

		ginkgo.It("should keep a ready container ready", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "probes-ready-across-restart"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:           "probed",
						Image:          defaultImage,
						Command:        []string{"sh", "-c", fmt.Sprintf("touch %s; sleep 3600", readyFlagPath)},
						StartupProbe:   flagProbe(0, 1, 30),
						ReadinessProbe: flagProbe(0, 1, 1),
						LivenessProbe:  flagProbe(0, 1, 3),
					}},
				},
			}
			testStatusStableAcrossKubeletRestart(ctx, pod, "probed", true, true)
		})

		ginkgo.It("should keep a not-ready container not ready", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "probes-notready-across-restart"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:    "probed",
						Image:   defaultImage,
						Command: []string{"sleep", "3600"},
						// The flag file is never created, so this container is
						// running but never ready.
						ReadinessProbe: flagProbe(0, 1, 1),
					}},
				},
			}
			testStatusStableAcrossKubeletRestart(ctx, pod, "probed", true, false)
		})

		ginkgo.It("should keep a container that has not started yet from being reported started", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "probes-notstarted-across-restart"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:    "probed",
						Image:   defaultImage,
						Command: []string{"sleep", "3600"},
						// The flag file is never created, so the startup probe
						// never passes and readiness never begins.
						StartupProbe:   flagProbe(0, 1, 3600),
						ReadinessProbe: execProbe([]string{"/bin/true"}, 0, 1, 1),
					}},
				},
			}
			testStatusStableAcrossKubeletRestart(ctx, pod, "probed", false, false)
		})

		ginkgo.It("should not disturb a probed container while another crash loops", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "probes-crashloop-across-restart"},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Containers: []v1.Container{{
						Name:           "probed",
						Image:          defaultImage,
						Command:        []string{"sh", "-c", fmt.Sprintf("touch %s; sleep 3600", readyFlagPath)},
						ReadinessProbe: flagProbe(0, 1, 1),
						LivenessProbe:  flagProbe(0, 1, 3),
					}, {
						Name:    "crasher",
						Image:   defaultImage,
						Command: []string{"sh", "-c", "exit 1"},
					}},
				},
			}
			testStatusStableAcrossKubeletRestart(ctx, pod, "probed", true, true)
		})

		// Regression test for https://github.com/kubernetes/kubernetes/issues/136910:
		// a sidecar's startup result was lost across a kubelet restart, which
		// deadlocked the restart of the regular containers behind it.
		ginkgo.It("should restart a regular container behind a started sidecar", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "probes-sidecar-across-restart"},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					InitContainers: []v1.Container{{
						Name:          "sidecar",
						Image:         defaultImage,
						Command:       []string{"sh", "-c", fmt.Sprintf("touch %s; sleep 3600", readyFlagPath)},
						RestartPolicy: ptr.To(v1.ContainerRestartPolicyAlways),
						StartupProbe:  flagProbe(0, 1, 30),
					}},
					Containers: []v1.Container{{
						Name:    "crasher",
						Image:   defaultImage,
						Command: []string{"sh", "-c", "trap 'exit 0' TERM; sleep 3600 & wait $!"},
					}},
				},
			}

			client := e2epod.NewPodClient(f)
			pod = client.Create(ctx, pod)
			waitForPodReady(ctx, f, pod)

			ginkgo.By("restarting the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)
			restartKubelet(ctx)

			ginkgo.By("killing the regular container")
			_, _, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, "crasher", "kill", "1")
			framework.ExpectNoError(err)

			ginkgo.By("waiting for the regular container to come back")
			err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "CrasherRestarted", f.Timeouts.PodStart,
				func(p *v1.Pod) (bool, error) {
					status := containerStatusByName(p, "crasher")
					return status != nil && status.RestartCount > 0 && status.State.Running != nil, nil
				})
			framework.ExpectNoError(err, "the regular container did not restart, so the sidecar was not reported as started")

			ginkgo.By("checking the sidecar itself never restarted")
			p, err := client.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			sidecar := containerStatusByName(p, "sidecar")
			gomega.Expect(sidecar).ToNot(gomega.BeNil())
			gomega.Expect(sidecar.RestartCount).To(gomega.BeZero(), "sidecar should not have restarted")
		})
	})

	ginkgo.Context("during graceful termination", func() {
		// A container that fails its probes as soon as it is asked to stop is
		// the case #105780 and #107894 are about: readiness must keep running
		// so the pod leaves service, while liveness must not get the container
		// killed and restarted on its way out.
		ginkgo.It("should go not ready before the container exits, without restarting it", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "probes-graceful-termination"},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: new(int64(120)),
					Containers: []v1.Container{{
						Name:  "probed",
						Image: defaultImage,
						// On SIGTERM, stop passing probes but keep running
						// well into the grace period.
						Command: []string{"sh", "-c", fmt.Sprintf(
							"touch %s; trap 'rm -f %s; sleep 60; exit 0' TERM; sleep 3600 & wait $!",
							readyFlagPath, readyFlagPath)},
						ReadinessProbe: flagProbe(0, 1, 1),
						LivenessProbe:  flagProbe(0, 1, 1),
					}},
				},
			}

			client := e2epod.NewPodClient(f)
			pod = client.Create(ctx, pod)
			waitForPodReady(ctx, f, pod)

			ginkgo.By("deleting the pod with a long grace period")
			err := client.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(120))
			framework.ExpectNoError(err)

			ginkgo.By("waiting for the container to go not ready while it is still running")
			err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "ContainerNotReadyWhileRunning", 60*time.Second,
				func(p *v1.Pod) (bool, error) {
					status := containerStatusByName(p, "probed")
					if status == nil || status.State.Running == nil {
						return false, nil
					}
					return !status.Ready, nil
				})
			framework.ExpectNoError(err, "the container should have gone not ready before it exited")

			ginkgo.By("checking the failing liveness probe did not restart the terminating container")
			gomega.Consistently(ctx, func(ctx context.Context) (int32, error) {
				p, err := client.Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return 0, err
				}
				status := containerStatusByName(p, "probed")
				if status == nil {
					return 0, nil
				}
				return status.RestartCount, nil
			}, 20*time.Second, 2*time.Second).Should(gomega.BeZero(),
				"liveness probing should stop when a container starts being killed")
		})
	})

	ginkgo.Context("startup gating", func() {
		ginkgo.It("should never report a container ready before it is started", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "probes-startup-gating"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:  "probed",
						Image: defaultImage,
						// Readiness would pass immediately, but must not be
						// consulted until the startup probe has passed.
						Command:        []string{"sh", "-c", fmt.Sprintf("sleep 15; touch %s; sleep 3600", readyFlagPath)},
						StartupProbe:   flagProbe(0, 1, 60),
						ReadinessProbe: execProbe([]string{"/bin/true"}, 0, 1, 1),
					}},
				},
			}

			client := e2epod.NewPodClient(f)
			pod = client.Create(ctx, pod)

			ginkgo.By("watching that Ready never leads Started")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "ContainerReady", f.Timeouts.PodStart,
				func(p *v1.Pod) (bool, error) {
					status := containerStatusByName(p, "probed")
					if status == nil {
						return false, nil
					}
					started := status.Started != nil && *status.Started
					if status.Ready && !started {
						return false, fmt.Errorf("container was reported Ready while Started was %v", status.Started)
					}
					return status.Ready, nil
				})
			framework.ExpectNoError(err)
		})
	})

	ginkgo.Context("probe scheduling", func() {
		// Regression test for https://github.com/kubernetes/kubernetes/issues/96614:
		// the first probe used to land on a ticker boundary set when the probe
		// worker happened to be created, so with a long period a container that
		// was healthy immediately could wait most of a period to be noticed.
		// Anchoring to the container's own start time removes that.
		ginkgo.It("should run the first probe relative to the container starting", func(ctx context.Context) {
			const period = 60

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "probes-first-probe-timing"},
				Spec: v1.PodSpec{
					// A slow init container moves the container's start well
					// away from the start of the pod sync.
					InitContainers: []v1.Container{{
						Name:    "delay",
						Image:   defaultImage,
						Command: []string{"sleep", "20"},
					}},
					Containers: []v1.Container{{
						Name:         "probed",
						Image:        defaultImage,
						Command:      []string{"sh", "-c", fmt.Sprintf("touch %s; sleep 3600", readyFlagPath)},
						StartupProbe: flagProbe(1, period, 10),
					}},
				},
			}

			client := e2epod.NewPodClient(f)
			pod = client.Create(ctx, pod)

			ginkgo.By("waiting for the container to be running")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "ContainerRunning", f.Timeouts.PodStart,
				func(p *v1.Pod) (bool, error) {
					status := containerStatusByName(p, "probed")
					return status != nil && status.State.Running != nil, nil
				})
			framework.ExpectNoError(err)
			running := time.Now()

			ginkgo.By("waiting for the startup probe to pass")
			err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "ContainerStarted", period*time.Second,
				func(p *v1.Pod) (bool, error) {
					status := containerStatusByName(p, "probed")
					return status != nil && status.Started != nil && *status.Started, nil
				})
			framework.ExpectNoError(err)

			// InitialDelaySeconds is 1, so the first probe is due about a
			// second after the container started, not up to a full period
			// later. The bound is generous to leave room for status
			// propagation.
			gomega.Expect(time.Since(running)).To(gomega.BeNumerically("<", period/2*time.Second),
				"the first probe did not run relative to the container starting")
		})
	})
})
