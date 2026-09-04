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

			// Give the kubelet time to take the pod over and run several sync
			// iterations so any transient status flap is exposed.
			time.Sleep(15 * time.Second)
			close(stopCh)

			for err := range errCh {
				framework.ExpectNoError(err, "pod status changed across the kubelet restart")
			}

			// Validate final status directly to guard against dropped watch events.
			p, err := client.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			status := containerStatusByName(p, containerName)
			gomega.Expect(status).ToNot(gomega.BeNil(), "container status should be present")
			gomega.Expect(status.RestartCount).To(gomega.BeZero(), "container should not have restarted across kubelet restart")
			gomega.Expect(status.Started).ToNot(gomega.BeNil())
			gomega.Expect(*status.Started).To(gomega.Equal(wantStarted), "container Started mismatch after restart")
			gomega.Expect(status.Ready).To(gomega.Equal(wantReady), "container Ready mismatch after restart")
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

		// A started sidecar must retain its startup result across a kubelet restart
		// so that regular containers depending on it are not blocked from restarting.
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

	ginkgo.Context("probe scheduling", func() {
		// The first probe run is anchored to the container start time so probe
		// timing is predictable and independent of when the worker was created.
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

			var startedAt time.Time
			ginkgo.By("waiting for the container to be running")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "ContainerRunning", f.Timeouts.PodStart,
				func(p *v1.Pod) (bool, error) {
					status := containerStatusByName(p, "probed")
					if status != nil && status.State.Running != nil {
						startedAt = status.State.Running.StartedAt.Time
						return true, nil
					}
					return false, nil
				})
			framework.ExpectNoError(err)

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
			gomega.Expect(time.Since(startedAt)).To(gomega.BeNumerically("<", period/2*time.Second),
				"the first probe did not run relative to the container starting")
		})
	})

	ginkgo.Context("across container restart", func() {
		// When a container restarts, probe results from the previous container
		// instance must not leak to the replacement. The new instance must be
		// reported NotReady until its own probes run and succeed.
		ginkgo.It("should reset readiness when a container crashes and restarts", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "probes-readiness-reset-on-restart"},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Volumes: []v1.Volume{{
						Name: "state",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					}},
					Containers: []v1.Container{{
						Name:  "probed",
						Image: defaultImage,
						// On the first run, touch the readyFlagPath to become ready,
						// then crash. On subsequent runs (persisted via an emptyDir),
						// do not touch readyFlagPath so readiness stays failing.
						Command: []string{"sh", "-c", fmt.Sprintf(
							"if [ ! -f /state/restarted ]; then touch /state/restarted; touch %s; sleep 5; exit 1; else sleep 3600; fi",
							readyFlagPath)},
						VolumeMounts: []v1.VolumeMount{{
							Name:      "state",
							MountPath: "/state",
						}},
						ReadinessProbe: flagProbe(0, 1, 1),
					}},
				},
			}

			client := e2epod.NewPodClient(f)
			pod = client.Create(ctx, pod)

			ginkgo.By("waiting for the initial container instance to become ready")
			waitForPodReady(ctx, f, pod)

			ginkgo.By("waiting for the container to crash and restart")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "ContainerRestarted", f.Timeouts.PodStart,
				func(p *v1.Pod) (bool, error) {
					status := containerStatusByName(p, "probed")
					return status != nil && status.RestartCount > 0 && status.State.Running != nil, nil
				})
			framework.ExpectNoError(err)

			ginkgo.By("verifying the replacement container does not inherit the previous instance's Ready state")
			gomega.Consistently(ctx, func(ctx context.Context) (bool, error) {
				p, err := client.Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				status := containerStatusByName(p, "probed")
				if status == nil || status.State.Running == nil {
					return false, fmt.Errorf("container is no longer running")
				}
				return status.Ready, nil
			}, 10*time.Second, 1*time.Second).Should(gomega.BeFalseBecause("the replacement container should not inherit Ready: true from the previous instance"))
		})
	})
})
