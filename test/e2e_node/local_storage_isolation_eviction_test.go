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

package e2e_node

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/test/e2e/framework"
)

type podEvictSpec struct {
	evicted bool
	pod     v1.Pod
}

const (
	totalEvict = 3
)

// Eviction Policy is described here:
// https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/kubelet-eviction.md

var _ = framework.KubeDescribe("LocalStorageCapacityIsolationEviction [Slow] [Serial] [Disruptive] [Flaky] [Feature:LocalStorageCapacityIsolation]", func() {

	f := framework.NewDefaultFramework("localstorage-eviction-test")

	emptyDirVolumeName := "volume-emptydir-pod"
	podTestSpecs := []podEvictSpec{
		{evicted: true, // This pod should be evicted because emptyDir (defualt storage type) usage violation
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "emptydir-hog-pod"},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: "gcr.io/google_containers/busybox:1.24",
							Name:  "container-emptydir-hog-pod",
							Command: []string{
								"sh",
								"-c",
								"sleep 5; dd if=/dev/urandom of=target-file of=/cache/target-file bs=50000 count=1; while true; do sleep 5; done",
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      emptyDirVolumeName,
									MountPath: "/cache",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: emptyDirVolumeName,
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									SizeLimit: *resource.NewQuantity(int64(1000), resource.BinarySI),
								},
							},
						},
					},
				},
			},
		},

		{evicted: true, // This pod should be evicted because emptyDir (memory type) usage violation
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "emptydir-memory-pod"},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: "gcr.io/google_containers/busybox:1.24",
							Name:  "container-emptydir-memory-pod",
							Command: []string{
								"sh",
								"-c",
								"sleep 5; dd if=/dev/urandom of=target-file of=/cache/target-file bs=50000 count=1; while true; do sleep 5; done",
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      emptyDirVolumeName,
									MountPath: "/cache",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: emptyDirVolumeName,
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    "Memory",
									SizeLimit: *resource.NewQuantity(int64(10000), resource.BinarySI),
								},
							},
						},
					},
				},
			},
		},

		{evicted: false,
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "container-emptydir-pod-critical"},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: "gcr.io/google_containers/busybox:1.24",
							Name:  "container-emptydir-hog-pod",
							Command: []string{
								"sh",
								"-c",
								"sleep 5; dd if=/dev/urandom of=target-file of=/cache/target-file bs=50000 count=1; while true; do sleep 5; done",
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      emptyDirVolumeName,
									MountPath: "/cache",
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: emptyDirVolumeName,
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									SizeLimit: *resource.NewQuantity(int64(100000), resource.BinarySI),
								},
							},
						},
					},
				},
			},
		},

		{evicted: true, // This pod should be evicted because container overlay usage violation
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "container-hog-pod"},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: "gcr.io/google_containers/busybox:1.24",
							Name:  "container-hog-pod",
							Command: []string{
								"sh",
								"-c",
								"sleep 5; dd if=/dev/urandom of=target-file bs=50000 count=1; while true; do sleep 5; done",
							},
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceStorageOverlay: *resource.NewMilliQuantity(
										int64(40000),
										resource.BinarySI),
								},
							},
						},
					},
				},
			},
		},
	}

	evictionTestTimeout := 10 * time.Minute
	testCondition := "EmptyDir/ContainerOverlay usage limit violation"
	Context(fmt.Sprintf("EmptyDirEviction when we run containers that should cause %s", testCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates += ", LocalStorageCapacityIsolation=true"
		})
		err := utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=true")
		if err != nil {
			framework.Failf("Failed to enable feature gate for LocalStorageCapacityIsolation: %v", err)
			return
		}

		runLocalStorageIsolationEvictionTest(f, testCondition, podTestSpecs, evictionTestTimeout, hasInodePressure)
	})
})

// runLocalStorageEvictionTest sets up a testing environment given the provided nodes, and checks a few things:
//		pods that exceed their local storage limit are evicted
//		pods that didn't exceed their local storage limit are not evicted
// runLocalStorageEvictionTest then cleans up the testing environment by deleting provided nodes,
func runLocalStorageIsolationEvictionTest(f *framework.Framework, testCondition string, podTestSpecs []podEvictSpec, evictionTestTimeout time.Duration,
	hasPressureCondition func(*framework.Framework, string) (bool, error)) {

	Context(fmt.Sprintf("EmptyDirEviction when we run containers that should cause %s", testCondition), func() {

		BeforeEach(func() {
			By("seting up pods to be used by tests")

			for _, spec := range podTestSpecs {
				By(fmt.Sprintf("creating pod with container: %s", spec.pod.Name))
				f.PodClient().CreateSync(&spec.pod)
			}
		})

		It(fmt.Sprintf("Test should eventually see %s, and then evict the correct pods", testCondition), func() {
			evictNum := 0
			evictMap := make(map[string]string)
			Eventually(func() error {
				// Gather current information
				updatedPodList, err := f.ClientSet.Core().Pods(f.Namespace.Name).List(metav1.ListOptions{})
				if err != nil {
					return fmt.Errorf("failed to get the list of pod: %v", err)
				}
				updatedPods := updatedPodList.Items

				for _, p := range updatedPods {
					framework.Logf("fetching pod %s; phase= %v", p.Name, p.Status.Phase)
					for _, testPod := range podTestSpecs {
						if p.Name == testPod.pod.Name {
							if !testPod.evicted {
								Expect(p.Status.Phase).NotTo(Equal(v1.PodFailed),
									fmt.Sprintf("%s pod failed (and shouldn't have failed)", p.Name))
							} else {
								if _, ok := evictMap[p.Name]; !ok && p.Status.Phase == v1.PodFailed {
									evictNum++
									evictMap[p.Name] = p.Name
								}
							}
						}
					}

				}
				if evictNum == totalEvict {
					return nil
				}
				return fmt.Errorf("pods that caused %s have not been evicted", testCondition)
			}, evictionTestTimeout, evictionPollInterval).Should(BeNil())

			By("making sure we can start a new pod after the test")
			podName := "test-admit-pod"
			f.PodClient().CreateSync(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: framework.GetPauseImageNameForHostArch(),
							Name:  podName,
						},
					},
				},
			})
		})

		AfterEach(func() {
			By("deleting pods")
			for _, spec := range podTestSpecs {
				By(fmt.Sprintf("deleting pod: %s", spec.pod.Name))
				f.PodClient().DeleteSync(spec.pod.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
			}

			if CurrentGinkgoTestDescription().Failed {
				if framework.TestContext.DumpLogsOnFailure {
					logPodEvents(f)
					logNodeEvents(f)
				}
				By("sleeping to allow for cleanup of test")
				time.Sleep(postTestConditionMonitoringPeriod)
			}
		})
	})
}
