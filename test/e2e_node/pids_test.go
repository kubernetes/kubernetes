/*
Copyright 2019 The Kubernetes Authors.

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
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// makePodToVerifyPids returns a pod that verifies specified cgroup with pids
func makePodToVerifyPids(baseName string, pidsLimit resource.Quantity) *v1.Pod {
	// convert the cgroup name to its literal form
	cgroupFsName := ""
	cgroupName := cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, baseName)
	if kubeletCfg.CgroupDriver == "systemd" {
		cgroupFsName = cgroupName.ToSystemd()
	} else {
		cgroupFsName = cgroupName.ToCgroupfs()
	}

	// this command takes the expected value and compares it against the actual value for the pod cgroup pids.max
	command := ""
	if IsCgroup2UnifiedMode() {
		command = fmt.Sprintf("expected=%v; actual=$(cat /tmp/%v/pids.max); if [ \"$expected\" -ne \"$actual\" ]; then exit 1; fi; ", pidsLimit.Value(), cgroupFsName)
	} else {
		command = fmt.Sprintf("expected=%v; actual=$(cat /tmp/pids/%v/pids.max); if [ \"$expected\" -ne \"$actual\" ]; then exit 1; fi; ", pidsLimit.Value(), cgroupFsName)
	}

	framework.Logf("Pod to run command: %v", command)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image:   busyboxImage,
					Name:    "container" + string(uuid.NewUUID()),
					Command: []string{"sh", "-c", command},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "sysfscgroup",
							MountPath: "/tmp",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "sysfscgroup",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
					},
				},
			},
		},
	}
	return pod
}

func runPodPidsLimitTests(f *framework.Framework) {
	ginkgo.It("should set pids.max for Pod", func(ctx context.Context) {
		ginkgo.By("by creating a G pod")
		pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Image: imageutils.GetPauseImageName(),
						Name:  "container" + string(uuid.NewUUID()),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceName("cpu"):    resource.MustParse("10m"),
								v1.ResourceName("memory"): resource.MustParse("100Mi"),
							},
						},
					},
				},
			},
		})
		podUID := string(pod.UID)
		ginkgo.By("checking if the expected pids settings were applied")
		verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("1024"))
		e2epod.NewPodClient(f).Create(ctx, verifyPod)
		err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
	})
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("PodPidsLimit", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("pids-limit-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("With config updated with pids limits", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.PodPidsLimit = int64(1024)
		})
		runPodPidsLimitTests(f)
		addAfterEachForCleaningUpPods(f)
	})
})

// Non-Serial: tests that verify per-pod PID limits without changing kubelet config.
// These assume the node podPidsLimit is either unlimited (-1, the default) or at
// least as large as the pod limits used below; otherwise the effective limit would
// be the node's, and the assertions would not hold.
var _ = SIGDescribe("PerPodPIDLimit", framework.WithFeatureGate(features.PerPodPIDLimit), func() {
	f := framework.NewDefaultFramework("per-pod-pid-limit-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		if kubeletCfg.PodPidsLimit > 0 && kubeletCfg.PodPidsLimit < 2048 {
			ginkgo.Skip(fmt.Sprintf("node podPidsLimit %d is lower than the pod limits exercised by this test", kubeletCfg.PodPidsLimit))
		}
	})

	ginkgo.It("should apply per-pod PID limit lower than node default", func(ctx context.Context) {
		ginkgo.By("creating a pod with pid limit below the node default")
		pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				Resources: &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourcePID: resource.MustParse("2048"),
					},
				},
				Containers: []v1.Container{
					{
						Image: imageutils.GetPauseImageName(),
						Name:  "container" + string(uuid.NewUUID()),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("10m"),
								v1.ResourceMemory: resource.MustParse("100Mi"),
							},
						},
					},
				},
			},
		})
		podUID := string(pod.UID)
		ginkgo.By("checking if the per-pod PID limit was applied")
		verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("2048"))
		e2epod.NewPodClient(f).Create(ctx, verifyPod)
		err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should apply single pod-level PID limit shared by multiple containers", func(ctx context.Context) {
		ginkgo.By("creating a multi-container pod with a pod-level pid limit")
		pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				Resources: &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourcePID: resource.MustParse("2048"),
					},
				},
				Containers: []v1.Container{
					{
						Image: imageutils.GetPauseImageName(),
						Name:  "app" + string(uuid.NewUUID()),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("10m"),
								v1.ResourceMemory: resource.MustParse("50Mi"),
							},
						},
					},
					{
						Image: imageutils.GetPauseImageName(),
						Name:  "sidecar" + string(uuid.NewUUID()),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("10m"),
								v1.ResourceMemory: resource.MustParse("50Mi"),
							},
						},
					},
				},
			},
		})
		podUID := string(pod.UID)
		ginkgo.By("checking that both containers share the same pod-level PID limit (2048)")
		verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("2048"))
		e2epod.NewPodClient(f).Create(ctx, verifyPod)
		err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should enforce per-pod PID limit with hostPID enabled", func(ctx context.Context) {
		ginkgo.By("creating a hostPID pod with per-pod PID limit to verify cgroup pids.max")
		pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "hostpid-cgroup-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				HostPID: true,
				Resources: &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourcePID: resource.MustParse("1024"),
					},
				},
				Containers: []v1.Container{
					{
						Image: imageutils.GetPauseImageName(),
						Name:  "container" + string(uuid.NewUUID()),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("10m"),
								v1.ResourceMemory: resource.MustParse("100Mi"),
							},
						},
					},
				},
			},
		})
		podUID := string(pod.UID)
		ginkgo.By("verifying pids.max is set to 1024 in the pod cgroup despite hostPID")
		verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("1024"))
		e2epod.NewPodClient(f).Create(ctx, verifyPod)
		err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should preserve container-derived QOS class for a pod with only a PID limit", func(ctx context.Context) {
		ginkgo.By("creating a pod with guaranteed containers and only a pod-level pid limit")
		pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				Resources: &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourcePID: resource.MustParse("2048"),
					},
				},
				Containers: []v1.Container{
					{
						Image: imageutils.GetPauseImageName(),
						Name:  "container" + string(uuid.NewUUID()),
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("100Mi"),
							},
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("100Mi"),
							},
						},
					},
				},
			},
		})

		ginkgo.By("verifying the pid limit did not change the QOS class or default pod-level requests")
		gomega.Expect(pod.Status.QOSClass).To(gomega.Equal(v1.PodQOSGuaranteed),
			"a pod-level pids limit must not change the container-derived QOS class")
		gomega.Expect(pod.Spec.Resources.Requests).To(gomega.BeEmpty(),
			"pod-level requests must not be defaulted for a pod that only sets limits.pids")

		ginkgo.By("verifying pids.max at the Guaranteed-tier pod cgroup path")
		pidsMax, err := getPodCgroupPidsMax(pod)
		framework.ExpectNoError(err)
		gomega.Expect(pidsMax).To(gomega.Equal(int64(2048)))
	})

	addAfterEachForCleaningUpPods(f)
})

// Serial because the test updates kubelet configuration to set a specific podPidsLimit.
var _ = SIGDescribe("PerPodPIDLimit", framework.WithSerial(), framework.WithFeatureGate(features.PerPodPIDLimit), func() {
	f := framework.NewDefaultFramework("per-pod-pid-limit-serial-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("With podPidsLimit=4096", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.PodPidsLimit = int64(4096)
		})

		ginkgo.It("should cap per-pod PID limit at node default when pod requests more", func(ctx context.Context) {
			ginkgo.By("creating a pod with pid limit above the node default")
			pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourcePID: resource.MustParse("8192"),
						},
					},
					Containers: []v1.Container{
						{
							Image: imageutils.GetPauseImageName(),
							Name:  "container" + string(uuid.NewUUID()),
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("10m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			})
			podUID := string(pod.UID)
			ginkgo.By("checking that the node default (4096) was applied since it is lower")
			verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("4096"))
			e2epod.NewPodClient(f).Create(ctx, verifyPod)
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)

			ginkgo.By("checking that a PIDLimitCapped warning event was emitted for the pod")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{
					FieldSelector: fmt.Sprintf("involvedObject.name=%s,reason=PIDLimitCapped", pod.Name),
				})
				if err != nil {
					return err
				}
				if len(events.Items) == 0 {
					return fmt.Errorf("no PIDLimitCapped event for pod %s yet", pod.Name)
				}
				return nil
			}, 2*time.Minute, 5*time.Second).Should(gomega.Succeed())
		})

		ginkgo.It("should use node default when no per-pod PID limit is specified", func(ctx context.Context) {
			ginkgo.By("creating a pod without a pid limit")
			pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image: imageutils.GetPauseImageName(),
							Name:  "container" + string(uuid.NewUUID()),
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("10m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			})
			podUID := string(pod.UID)
			ginkgo.By("checking that the node default (4096) was applied")
			verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("4096"))
			e2epod.NewPodClient(f).Create(ctx, verifyPod)
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		addAfterEachForCleaningUpPods(f)
	})
})

// Non-Serial: PSA compatibility does not require specific kubelet configuration.
var _ = SIGDescribe("PerPodPIDLimit PSA Compatibility", framework.WithFeatureGate(features.PerPodPIDLimit), func() {
	ginkgo.Context("Baseline PSA profile", func() {
		f := framework.NewDefaultFramework("per-pod-pid-psa-baseline")
		f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

		ginkgo.It("should admit pod with pid limit under Baseline PSA", func(ctx context.Context) {
			ginkgo.By("creating a pod with pid limit in a Baseline namespace")
			pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "psa-baseline-pid-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourcePID: resource.MustParse("2048"),
						},
					},
					Containers: []v1.Container{
						{
							Image: imageutils.GetPauseImageName(),
							Name:  "container" + string(uuid.NewUUID()),
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("10m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			})
			ginkgo.By("verifying the pod was admitted and is running")
			gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodRunning))
		})

		addAfterEachForCleaningUpPods(f)
	})

	ginkgo.Context("Restricted PSA profile", func() {
		f := framework.NewDefaultFramework("per-pod-pid-psa-restricted")
		f.NamespacePodSecurityLevel = admissionapi.LevelRestricted

		ginkgo.It("should admit pod with pid limit under Restricted PSA", func(ctx context.Context) {
			ginkgo.By("creating a restricted-compliant pod with pid limit")
			pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "psa-restricted-pid-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourcePID: resource.MustParse("2048"),
						},
					},
					Containers: []v1.Container{
						{
							Image: imageutils.GetPauseImageName(),
							Name:  "container" + string(uuid.NewUUID()),
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("10m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
							SecurityContext: &v1.SecurityContext{
								RunAsNonRoot:             new(true),
								RunAsUser:                new(int64(1000)),
								AllowPrivilegeEscalation: new(false),
								Capabilities: &v1.Capabilities{
									Drop: []v1.Capability{"ALL"},
								},
								SeccompProfile: &v1.SeccompProfile{
									Type: v1.SeccompProfileTypeRuntimeDefault,
								},
							},
						},
					},
				},
			})
			ginkgo.By("verifying the pod was admitted and is running")
			gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodRunning))
		})

		addAfterEachForCleaningUpPods(f)
	})
})

// getPodCgroupPidsMax reads pids.max from the pod-level cgroup.
func getPodCgroupPidsMax(pod *v1.Pod) (int64, error) {
	podCgroupSuffix := cm.GetPodCgroupNameSuffix(pod.UID)
	var cgroupName cm.CgroupName
	switch pod.Status.QOSClass {
	case v1.PodQOSGuaranteed:
		cgroupName = cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, podCgroupSuffix)
	case v1.PodQOSBurstable:
		cgroupName = cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, strings.ToLower(string(v1.PodQOSBurstable)), podCgroupSuffix)
	case v1.PodQOSBestEffort:
		cgroupName = cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, strings.ToLower(string(v1.PodQOSBestEffort)), podCgroupSuffix)
	default:
		return 0, fmt.Errorf("unexpected QOS class %q for pod %s", pod.Status.QOSClass, pod.Name)
	}

	pidsMaxFile := filepath.Join("/sys/fs/cgroup", toCgroupFsName(cgroupName), "pids.max")
	out, err := os.ReadFile(pidsMaxFile)
	if err != nil {
		return 0, fmt.Errorf("failed to read %s: %w", pidsMaxFile, err)
	}
	return strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
}

var _ = SIGDescribe("PerPodPIDLimit Static Pod", feature.StandaloneMode, framework.WithFeatureGate(features.PerPodPIDLimit), func() {
	f := framework.NewDefaultFramework("per-pod-pid-static")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when creating a static pod with per-pod PID limit", func() {
		var ns, podPath, staticPodName string

		ginkgo.BeforeEach(func() {
			if !IsCgroup2UnifiedMode() {
				ginkgo.Skip("per-pod PID limits require cgroupsv2")
			}
			ns = f.Namespace.Name
			staticPodName = "static-pid-limit-" + string(uuid.NewUUID())
			podPath = kubeletCfg.StaticPodPath
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By(fmt.Sprintf("delete the static pod (%v/%v)", ns, staticPodName))
			err := deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("wait for pod to disappear (%v/%v)", ns, staticPodName))
			gomega.Eventually(ctx, func(ctx context.Context) error {
				_, err := getPodFromStandaloneKubelet(ctx, ns, staticPodName)
				if err != nil {
					return nil
				}
				return fmt.Errorf("pod (%v/%v) still exists", ns, staticPodName)
			}, f.Timeouts.PodStart, time.Second*5).Should(gomega.Succeed())
		})

		ginkgo.It("should enforce per-pod PID limit on a static pod without API server", func(ctx context.Context) {
			ginkgo.By("creating a static pod with pid limit 2048")
			podSpec := &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      staticPodName,
					Namespace: ns,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourcePID: resource.MustParse("2048"),
						},
					},
					Containers: []v1.Container{
						{
							Name:  "pause",
							Image: imageutils.GetPauseImageName(),
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceMemory: resource.MustParse("50Mi"),
								},
							},
						},
					},
				},
			}

			err := scheduleStaticPod(podPath, staticPodName, ns, podSpec)
			framework.ExpectNoError(err)

			ginkgo.By("waiting for the static pod to be running")
			var pod *v1.Pod
			gomega.Eventually(ctx, func(ctx context.Context) error {
				var err error
				pod, err = getPodFromStandaloneKubelet(ctx, ns, staticPodName)
				if err != nil {
					return fmt.Errorf("error getting pod (%v/%v): %w", ns, staticPodName, err)
				}
				isReady, err := testutils.PodRunningReady(pod)
				if err != nil {
					return fmt.Errorf("error checking pod readiness: %w", err)
				}
				if !isReady {
					return fmt.Errorf("pod (%v/%v) is not running", ns, staticPodName)
				}
				return nil
			}, f.Timeouts.PodStart, time.Second*5).Should(gomega.Succeed())

			ginkgo.By("verifying pids.max is set to 2048 in the pod cgroup")
			pidsMax, err := getPodCgroupPidsMax(pod)
			framework.ExpectNoError(err)
			gomega.Expect(pidsMax).To(gomega.Equal(int64(2048)),
				"expected pids.max=2048 for static pod with pid limit 2048")
		})
	})
})
