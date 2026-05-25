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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
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

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("PerPodPIDLimit", framework.WithSerial(), framework.WithFeatureGate(features.PerPodPIDLimit), func() {
	f := framework.NewDefaultFramework("per-pod-pid-limit-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("With PerPodPIDLimit feature gate enabled", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.PodPidsLimit = int64(4096)
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(features.PerPodPIDLimit)] = true
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
			ginkgo.By("checking if the per-pod PID limit was applied (min of node 4096, pod 2048 = 2048)")
			verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("2048"))
			e2epod.NewPodClient(f).Create(ctx, verifyPod)
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
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

			ginkgo.By("running a fork test pod to verify host processes are visible but fork fails at the PID limit")
			forkTestPod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "hostpid-fork-test-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					HostPID:       true,
					RestartPolicy: v1.RestartPolicyNever,
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourcePID: resource.MustParse("1024"),
						},
					},
					Containers: []v1.Container{
						{
							Image: busyboxImage,
							Name:  "fork-test",
							Command: []string{"sh", "-c",
								// Step 1: Verify hostPID — host processes must be visible
								`host_count=$(ps aux 2>/dev/null | wc -l); ` +
									`if [ "$host_count" -lt 10 ]; then ` +
									`echo "FAIL: hostPID not effective, only $host_count processes visible"; exit 1; fi; ` +
									`echo "OK: hostPID working, $host_count host processes visible"; ` +
									// Step 2: Fork processes beyond PID limit (1024), expect failure
									`i=0; while [ $i -lt 1500 ]; do sleep 3600 & i=$((i + 1)); done 2>/tmp/fork_err; ` +
									// Step 3: Check that fork was limited
									`if grep -q "fork" /tmp/fork_err 2>/dev/null; then ` +
									`echo "OK: PID limit enforced with hostPID - fork failed as expected"; exit 0; fi; ` +
									`echo "FAIL: spawned processes without hitting PID limit"; exit 1`,
							},
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("512Mi"),
								},
							},
						},
					},
				},
			}
			e2epod.NewPodClient(f).Create(ctx, forkTestPod)
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, forkTestPod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		addAfterEachForCleaningUpPods(f)
	})
})

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("PerPodPIDLimit PSA Compatibility", framework.WithSerial(), framework.WithFeatureGate(features.PerPodPIDLimit), func() {
	ginkgo.Context("Baseline PSA profile", func() {
		f := framework.NewDefaultFramework("per-pod-pid-psa-baseline")
		f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.PodPidsLimit = int64(4096)
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(features.PerPodPIDLimit)] = true
		})

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
			podUID := string(pod.UID)
			ginkgo.By("verifying the PID limit was applied")
			verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("2048"))
			e2epod.NewPodClient(f).Create(ctx, verifyPod)
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		addAfterEachForCleaningUpPods(f)
	})

	ginkgo.Context("Restricted PSA profile", func() {
		f := framework.NewDefaultFramework("per-pod-pid-psa-restricted")
		f.NamespacePodSecurityLevel = admissionapi.LevelRestricted

		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.PodPidsLimit = int64(4096)
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(features.PerPodPIDLimit)] = true
		})

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
			podUID := string(pod.UID)
			ginkgo.By("verifying the PID limit was applied")
			verifyPod := makePodToVerifyPids("pod"+podUID, resource.MustParse("2048"))
			e2epod.NewPodClient(f).Create(ctx, verifyPod)
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, verifyPod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		addAfterEachForCleaningUpPods(f)
	})
})
