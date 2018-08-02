/*
Copyright 2014 The Kubernetes Authors.

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

package apps

import (
	"context"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	klabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	zookeeperManifestPath   = "test/e2e/testing-manifests/statefulset/zookeeper"
	mysqlGaleraManifestPath = "test/e2e/testing-manifests/statefulset/mysql-galera"
	redisManifestPath       = "test/e2e/testing-manifests/statefulset/redis"
	cockroachDBManifestPath = "test/e2e/testing-manifests/statefulset/cockroachdb"
	// We don't restart MySQL cluster regardless of restartCluster, since MySQL doesn't handle restart well
	restartCluster = true

	// Timeout for reads from databases running on stateful pods.
	readTimeout = 60 * time.Second
)

// GCE Quota requirements: 3 pds, one per stateful pod manifest declared above.
// GCE Api requirements: nodes and master need storage r/w permissions.
var _ = SIGDescribe("StatefulSet", func() {
	f := framework.NewDefaultFramework("statefulset")
	var ns string
	var c clientset.Interface

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	framework.KubeDescribe("Basic StatefulSet functionality [StatefulSetBasic]", func() {
		ssName := "ss"
		labels := map[string]string{
			"foo": "bar",
			"baz": "blah",
		}
		headlessSvcName := "test"
		var statefulPodMounts, podMounts []v1.VolumeMount
		var ss *apps.StatefulSet

		BeforeEach(func() {
			statefulPodMounts = []v1.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
			podMounts = []v1.VolumeMount{{Name: "home", MountPath: "/home"}}
			ss = framework.NewStatefulSet(ssName, ns, headlessSvcName, 2, statefulPodMounts, podMounts, labels)

			By("Creating service " + headlessSvcName + " in namespace " + ns)
			headlessService := framework.CreateServiceSpec(headlessSvcName, "", true, labels)
			_, err := c.CoreV1().Services(ns).Create(headlessService)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				framework.DumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all statefulset in ns %v", ns)
			framework.DeleteAllStatefulSets(c, ns)
		})

		// This can't be Conformance yet because it depends on a default
		// StorageClass and a dynamic provisioner.
		It("should provide basic identity", func() {
			By("Creating statefulset " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 3
			sst := framework.NewStatefulSetTester(c)
			sst.PauseNewPods(ss)

			_, err := c.AppsV1().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())

			By("Saturating stateful set " + ss.Name)
			sst.Saturate(ss)

			By("Verifying statefulset mounted data directory is usable")
			framework.ExpectNoError(sst.CheckMount(ss, "/data"))

			By("Verifying statefulset provides a stable hostname for each pod")
			framework.ExpectNoError(sst.CheckHostname(ss))

			By("Verifying statefulset set proper service name")
			framework.ExpectNoError(sst.CheckServiceName(ss, headlessSvcName))

			cmd := "echo $(hostname) > /data/hostname; sync;"
			By("Running " + cmd + " in all stateful pods")
			framework.ExpectNoError(sst.ExecInStatefulPods(ss, cmd))

			By("Restarting statefulset " + ss.Name)
			sst.Restart(ss)
			sst.WaitForRunningAndReady(*ss.Spec.Replicas, ss)

			By("Verifying statefulset mounted data directory is usable")
			framework.ExpectNoError(sst.CheckMount(ss, "/data"))

			cmd = "if [ \"$(cat /data/hostname)\" = \"$(hostname)\" ]; then exit 0; else exit 1; fi"
			By("Running " + cmd + " in all stateful pods")
			framework.ExpectNoError(sst.ExecInStatefulPods(ss, cmd))
		})

		// This can't be Conformance yet because it depends on a default
		// StorageClass and a dynamic provisioner.
		It("should adopt matching orphans and release non-matching pods", func() {
			By("Creating statefulset " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 1
			sst := framework.NewStatefulSetTester(c)
			sst.PauseNewPods(ss)

			// Replace ss with the one returned from Create() so it has the UID.
			// Save Kind since it won't be populated in the returned ss.
			kind := ss.Kind
			ss, err := c.AppsV1().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())
			ss.Kind = kind

			By("Saturating stateful set " + ss.Name)
			sst.Saturate(ss)
			pods := sst.GetPodList(ss)
			Expect(pods.Items).To(HaveLen(int(*ss.Spec.Replicas)))

			By("Checking that stateful set pods are created with ControllerRef")
			pod := pods.Items[0]
			controllerRef := metav1.GetControllerOf(&pod)
			Expect(controllerRef).ToNot(BeNil())
			Expect(controllerRef.Kind).To(Equal(ss.Kind))
			Expect(controllerRef.Name).To(Equal(ss.Name))
			Expect(controllerRef.UID).To(Equal(ss.UID))

			By("Orphaning one of the stateful set's pods")
			f.PodClient().Update(pod.Name, func(pod *v1.Pod) {
				pod.OwnerReferences = nil
			})

			By("Checking that the stateful set readopts the pod")
			Expect(framework.WaitForPodCondition(c, pod.Namespace, pod.Name, "adopted", framework.StatefulSetTimeout,
				func(pod *v1.Pod) (bool, error) {
					controllerRef := metav1.GetControllerOf(pod)
					if controllerRef == nil {
						return false, nil
					}
					if controllerRef.Kind != ss.Kind || controllerRef.Name != ss.Name || controllerRef.UID != ss.UID {
						return false, fmt.Errorf("pod has wrong controllerRef: %v", controllerRef)
					}
					return true, nil
				},
			)).To(Succeed(), "wait for pod %q to be readopted", pod.Name)

			By("Removing the labels from one of the stateful set's pods")
			prevLabels := pod.Labels
			f.PodClient().Update(pod.Name, func(pod *v1.Pod) {
				pod.Labels = nil
			})

			By("Checking that the stateful set releases the pod")
			Expect(framework.WaitForPodCondition(c, pod.Namespace, pod.Name, "released", framework.StatefulSetTimeout,
				func(pod *v1.Pod) (bool, error) {
					controllerRef := metav1.GetControllerOf(pod)
					if controllerRef != nil {
						return false, nil
					}
					return true, nil
				},
			)).To(Succeed(), "wait for pod %q to be released", pod.Name)

			// If we don't do this, the test leaks the Pod and PVC.
			By("Readding labels to the stateful set's pod")
			f.PodClient().Update(pod.Name, func(pod *v1.Pod) {
				pod.Labels = prevLabels
			})

			By("Checking that the stateful set readopts the pod")
			Expect(framework.WaitForPodCondition(c, pod.Namespace, pod.Name, "adopted", framework.StatefulSetTimeout,
				func(pod *v1.Pod) (bool, error) {
					controllerRef := metav1.GetControllerOf(pod)
					if controllerRef == nil {
						return false, nil
					}
					if controllerRef.Kind != ss.Kind || controllerRef.Name != ss.Name || controllerRef.UID != ss.UID {
						return false, fmt.Errorf("pod has wrong controllerRef: %v", controllerRef)
					}
					return true, nil
				},
			)).To(Succeed(), "wait for pod %q to be readopted", pod.Name)
		})

		// This can't be Conformance yet because it depends on a default
		// StorageClass and a dynamic provisioner.
		It("should not deadlock when a pod's predecessor fails", func() {
			By("Creating statefulset " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 2
			sst := framework.NewStatefulSetTester(c)
			sst.PauseNewPods(ss)

			_, err := c.AppsV1().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())

			sst.WaitForRunning(1, 0, ss)

			By("Resuming stateful pod at index 0.")
			sst.ResumeNextPod(ss)

			By("Waiting for stateful pod at index 1 to enter running.")
			sst.WaitForRunning(2, 1, ss)

			// Now we have 1 healthy and 1 unhealthy stateful pod. Deleting the healthy stateful pod should *not*
			// create a new stateful pod till the remaining stateful pod becomes healthy, which won't happen till
			// we set the healthy bit.

			By("Deleting healthy stateful pod at index 0.")
			sst.DeleteStatefulPodAtIndex(0, ss)

			By("Confirming stateful pod at index 0 is recreated.")
			sst.WaitForRunning(2, 1, ss)

			By("Resuming stateful pod at index 1.")
			sst.ResumeNextPod(ss)

			By("Confirming all stateful pods in statefulset are created.")
			sst.WaitForRunningAndReady(*ss.Spec.Replicas, ss)
		})

		// This can't be Conformance yet because it depends on a default
		// StorageClass and a dynamic provisioner.
		It("should perform rolling updates and roll backs of template modifications with PVCs", func() {
			By("Creating a new StatefulSet with PVCs")
			*(ss.Spec.Replicas) = 3
			rollbackTest(c, ns, ss)
		})

		/*
			Release : v1.9
			Testname: StatefulSet, Rolling Update
			Description: StatefulSet MUST support the RollingUpdate strategy to automatically replace Pods one at a time when the Pod template changes. The StatefulSet's status MUST indicate the CurrentRevision and UpdateRevision. If the template is changed to match a prior revision, StatefulSet MUST detect this as a rollback instead of creating a new revision. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("should perform rolling updates and roll backs of template modifications", func() {
			By("Creating a new StatefulSet")
			ss := framework.NewStatefulSet("ss2", ns, headlessSvcName, 3, nil, nil, labels)
			rollbackTest(c, ns, ss)
		})

		/*
			Release : v1.9
			Testname: StatefulSet, Rolling Update with Partition
			Description: StatefulSet's RollingUpdate strategy MUST support the Partition parameter for canaries and phased rollouts. If a Pod is deleted while a rolling update is in progress, StatefulSet MUST restore the Pod without violating the Partition. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("should perform canary updates and phased rolling updates of template modifications", func() {
			By("Creating a new StaefulSet")
			ss := framework.NewStatefulSet("ss2", ns, headlessSvcName, 3, nil, nil, labels)
			sst := framework.NewStatefulSetTester(c)
			sst.SetHttpProbe(ss)
			ss.Spec.UpdateStrategy = apps.StatefulSetUpdateStrategy{
				Type: apps.RollingUpdateStatefulSetStrategyType,
				RollingUpdate: func() *apps.RollingUpdateStatefulSetStrategy {
					return &apps.RollingUpdateStatefulSetStrategy{
						Partition: func() *int32 {
							i := int32(3)
							return &i
						}()}
				}(),
			}
			ss, err := c.AppsV1().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())
			sst.WaitForRunningAndReady(*ss.Spec.Replicas, ss)
			ss = sst.WaitForStatus(ss)
			currentRevision, updateRevision := ss.Status.CurrentRevision, ss.Status.UpdateRevision
			Expect(currentRevision).To(Equal(updateRevision),
				fmt.Sprintf("StatefulSet %s/%s created with update revision %s not equal to current revision %s",
					ss.Namespace, ss.Name, updateRevision, currentRevision))
			pods := sst.GetPodList(ss)
			for i := range pods.Items {
				Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(currentRevision),
					fmt.Sprintf("Pod %s/%s revision %s is not equal to currentRevision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
						currentRevision))
			}
			newImage := NewNginxImage
			oldImage := ss.Spec.Template.Spec.Containers[0].Image

			By(fmt.Sprintf("Updating stateful set template: update image from %s to %s", oldImage, newImage))
			Expect(oldImage).NotTo(Equal(newImage), "Incorrect test setup: should update to a different image")
			ss, err = framework.UpdateStatefulSetWithRetries(c, ns, ss.Name, func(update *apps.StatefulSet) {
				update.Spec.Template.Spec.Containers[0].Image = newImage
			})
			Expect(err).NotTo(HaveOccurred())

			By("Creating a new revision")
			ss = sst.WaitForStatus(ss)
			currentRevision, updateRevision = ss.Status.CurrentRevision, ss.Status.UpdateRevision
			Expect(currentRevision).NotTo(Equal(updateRevision),
				"Current revision should not equal update revision during rolling update")

			By("Not applying an update when the partition is greater than the number of replicas")
			for i := range pods.Items {
				Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(oldImage),
					fmt.Sprintf("Pod %s/%s has image %s not equal to current image %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Spec.Containers[0].Image,
						oldImage))
				Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(currentRevision),
					fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
						currentRevision))
			}

			By("Performing a canary update")
			ss.Spec.UpdateStrategy = apps.StatefulSetUpdateStrategy{
				Type: apps.RollingUpdateStatefulSetStrategyType,
				RollingUpdate: func() *apps.RollingUpdateStatefulSetStrategy {
					return &apps.RollingUpdateStatefulSetStrategy{
						Partition: func() *int32 {
							i := int32(2)
							return &i
						}()}
				}(),
			}
			ss, err = framework.UpdateStatefulSetWithRetries(c, ns, ss.Name, func(update *apps.StatefulSet) {
				update.Spec.UpdateStrategy = apps.StatefulSetUpdateStrategy{
					Type: apps.RollingUpdateStatefulSetStrategyType,
					RollingUpdate: func() *apps.RollingUpdateStatefulSetStrategy {
						return &apps.RollingUpdateStatefulSetStrategy{
							Partition: func() *int32 {
								i := int32(2)
								return &i
							}()}
					}(),
				}
			})
			Expect(err).NotTo(HaveOccurred())
			ss, pods = sst.WaitForPartitionedRollingUpdate(ss)
			for i := range pods.Items {
				if i < int(*ss.Spec.UpdateStrategy.RollingUpdate.Partition) {
					Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(oldImage),
						fmt.Sprintf("Pod %s/%s has image %s not equal to current image %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Spec.Containers[0].Image,
							oldImage))
					Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(currentRevision),
						fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
							currentRevision))
				} else {
					Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(newImage),
						fmt.Sprintf("Pod %s/%s has image %s not equal to new image  %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Spec.Containers[0].Image,
							newImage))
					Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(updateRevision),
						fmt.Sprintf("Pod %s/%s has revision %s not equal to new revision %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
							updateRevision))
				}
			}

			By("Restoring Pods to the correct revision when they are deleted")
			sst.DeleteStatefulPodAtIndex(0, ss)
			sst.DeleteStatefulPodAtIndex(2, ss)
			sst.WaitForRunningAndReady(3, ss)
			ss = sst.GetStatefulSet(ss.Namespace, ss.Name)
			pods = sst.GetPodList(ss)
			for i := range pods.Items {
				if i < int(*ss.Spec.UpdateStrategy.RollingUpdate.Partition) {
					Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(oldImage),
						fmt.Sprintf("Pod %s/%s has image %s not equal to current image %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Spec.Containers[0].Image,
							oldImage))
					Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(currentRevision),
						fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
							currentRevision))
				} else {
					Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(newImage),
						fmt.Sprintf("Pod %s/%s has image %s not equal to new image  %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Spec.Containers[0].Image,
							newImage))
					Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(updateRevision),
						fmt.Sprintf("Pod %s/%s has revision %s not equal to new revision %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
							updateRevision))
				}
			}

			By("Performing a phased rolling update")
			for i := int(*ss.Spec.UpdateStrategy.RollingUpdate.Partition) - 1; i >= 0; i-- {
				ss, err = framework.UpdateStatefulSetWithRetries(c, ns, ss.Name, func(update *apps.StatefulSet) {
					update.Spec.UpdateStrategy = apps.StatefulSetUpdateStrategy{
						Type: apps.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: func() *apps.RollingUpdateStatefulSetStrategy {
							j := int32(i)
							return &apps.RollingUpdateStatefulSetStrategy{
								Partition: &j,
							}
						}(),
					}
				})
				Expect(err).NotTo(HaveOccurred())
				ss, pods = sst.WaitForPartitionedRollingUpdate(ss)
				for i := range pods.Items {
					if i < int(*ss.Spec.UpdateStrategy.RollingUpdate.Partition) {
						Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(oldImage),
							fmt.Sprintf("Pod %s/%s has image %s not equal to current image %s",
								pods.Items[i].Namespace,
								pods.Items[i].Name,
								pods.Items[i].Spec.Containers[0].Image,
								oldImage))
						Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(currentRevision),
							fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
								pods.Items[i].Namespace,
								pods.Items[i].Name,
								pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
								currentRevision))
					} else {
						Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(newImage),
							fmt.Sprintf("Pod %s/%s has image %s not equal to new image  %s",
								pods.Items[i].Namespace,
								pods.Items[i].Name,
								pods.Items[i].Spec.Containers[0].Image,
								newImage))
						Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(updateRevision),
							fmt.Sprintf("Pod %s/%s has revision %s not equal to new revision %s",
								pods.Items[i].Namespace,
								pods.Items[i].Name,
								pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
								updateRevision))
					}
				}
			}
			Expect(ss.Status.CurrentRevision).To(Equal(updateRevision),
				fmt.Sprintf("StatefulSet %s/%s current revision %s does not equal update revison %s on update completion",
					ss.Namespace,
					ss.Name,
					ss.Status.CurrentRevision,
					updateRevision))

		})

		// Do not mark this as Conformance.
		// The legacy OnDelete strategy only exists for backward compatibility with pre-v1 APIs.
		It("should implement legacy replacement when the update strategy is OnDelete", func() {
			By("Creating a new StatefulSet")
			ss := framework.NewStatefulSet("ss2", ns, headlessSvcName, 3, nil, nil, labels)
			sst := framework.NewStatefulSetTester(c)
			sst.SetHttpProbe(ss)
			ss.Spec.UpdateStrategy = apps.StatefulSetUpdateStrategy{
				Type: apps.OnDeleteStatefulSetStrategyType,
			}
			ss, err := c.AppsV1().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())
			sst.WaitForRunningAndReady(*ss.Spec.Replicas, ss)
			ss = sst.WaitForStatus(ss)
			currentRevision, updateRevision := ss.Status.CurrentRevision, ss.Status.UpdateRevision
			Expect(currentRevision).To(Equal(updateRevision),
				fmt.Sprintf("StatefulSet %s/%s created with update revision %s not equal to current revision %s",
					ss.Namespace, ss.Name, updateRevision, currentRevision))
			pods := sst.GetPodList(ss)
			for i := range pods.Items {
				Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(currentRevision),
					fmt.Sprintf("Pod %s/%s revision %s is not equal to current revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
						currentRevision))
			}

			By("Restoring Pods to the current revision")
			sst.DeleteStatefulPodAtIndex(0, ss)
			sst.DeleteStatefulPodAtIndex(1, ss)
			sst.DeleteStatefulPodAtIndex(2, ss)
			sst.WaitForRunningAndReady(3, ss)
			ss = sst.GetStatefulSet(ss.Namespace, ss.Name)
			pods = sst.GetPodList(ss)
			for i := range pods.Items {
				Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(currentRevision),
					fmt.Sprintf("Pod %s/%s revision %s is not equal to current revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
						currentRevision))
			}
			newImage := NewNginxImage
			oldImage := ss.Spec.Template.Spec.Containers[0].Image

			By(fmt.Sprintf("Updating stateful set template: update image from %s to %s", oldImage, newImage))
			Expect(oldImage).NotTo(Equal(newImage), "Incorrect test setup: should update to a different image")
			ss, err = framework.UpdateStatefulSetWithRetries(c, ns, ss.Name, func(update *apps.StatefulSet) {
				update.Spec.Template.Spec.Containers[0].Image = newImage
			})
			Expect(err).NotTo(HaveOccurred())

			By("Creating a new revision")
			ss = sst.WaitForStatus(ss)
			currentRevision, updateRevision = ss.Status.CurrentRevision, ss.Status.UpdateRevision
			Expect(currentRevision).NotTo(Equal(updateRevision),
				"Current revision should not equal update revision during rolling update")

			By("Recreating Pods at the new revision")
			sst.DeleteStatefulPodAtIndex(0, ss)
			sst.DeleteStatefulPodAtIndex(1, ss)
			sst.DeleteStatefulPodAtIndex(2, ss)
			sst.WaitForRunningAndReady(3, ss)
			ss = sst.GetStatefulSet(ss.Namespace, ss.Name)
			pods = sst.GetPodList(ss)
			for i := range pods.Items {
				Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(newImage),
					fmt.Sprintf("Pod %s/%s has image %s not equal to new image %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Spec.Containers[0].Image,
						newImage))
				Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(updateRevision),
					fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
						updateRevision))
			}
		})

		/*
			Release : v1.9
			Testname: StatefulSet, Scaling
			Description: StatefulSet MUST create Pods in ascending order by ordinal index when scaling up, and delete Pods in descending order when scaling down. Scaling up or down MUST pause if any Pods belonging to the StatefulSet are unhealthy. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("Scaling should happen in predictable order and halt if any stateful pod is unhealthy", func() {
			psLabels := klabels.Set(labels)
			By("Initializing watcher for selector " + psLabels.String())
			watcher, err := f.ClientSet.CoreV1().Pods(ns).Watch(metav1.ListOptions{
				LabelSelector: psLabels.AsSelector().String(),
			})
			Expect(err).NotTo(HaveOccurred())

			By("Creating stateful set " + ssName + " in namespace " + ns)
			ss := framework.NewStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, psLabels)
			sst := framework.NewStatefulSetTester(c)
			sst.SetHttpProbe(ss)
			ss, err = c.AppsV1().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())

			By("Waiting until all stateful set " + ssName + " replicas will be running in namespace " + ns)
			sst.WaitForRunningAndReady(*ss.Spec.Replicas, ss)

			By("Confirming that stateful set scale up will halt with unhealthy stateful pod")
			sst.BreakHttpProbe(ss)
			sst.WaitForRunningAndNotReady(*ss.Spec.Replicas, ss)
			sst.WaitForStatusReadyReplicas(ss, 0)
			sst.UpdateReplicas(ss, 3)
			sst.ConfirmStatefulPodCount(1, ss, 10*time.Second, true)

			By("Scaling up stateful set " + ssName + " to 3 replicas and waiting until all of them will be running in namespace " + ns)
			sst.RestoreHttpProbe(ss)
			sst.WaitForRunningAndReady(3, ss)

			By("Verifying that stateful set " + ssName + " was scaled up in order")
			expectedOrder := []string{ssName + "-0", ssName + "-1", ssName + "-2"}
			ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), framework.StatefulSetTimeout)
			defer cancel()
			_, err = watchtools.UntilWithoutRetry(ctx, watcher, func(event watch.Event) (bool, error) {
				if event.Type != watch.Added {
					return false, nil
				}
				pod := event.Object.(*v1.Pod)
				if pod.Name == expectedOrder[0] {
					expectedOrder = expectedOrder[1:]
				}
				return len(expectedOrder) == 0, nil

			})
			Expect(err).NotTo(HaveOccurred())

			By("Scale down will halt with unhealthy stateful pod")
			watcher, err = f.ClientSet.CoreV1().Pods(ns).Watch(metav1.ListOptions{
				LabelSelector: psLabels.AsSelector().String(),
			})
			Expect(err).NotTo(HaveOccurred())

			sst.BreakHttpProbe(ss)
			sst.WaitForStatusReadyReplicas(ss, 0)
			sst.WaitForRunningAndNotReady(3, ss)
			sst.UpdateReplicas(ss, 0)
			sst.ConfirmStatefulPodCount(3, ss, 10*time.Second, true)

			By("Scaling down stateful set " + ssName + " to 0 replicas and waiting until none of pods will run in namespace" + ns)
			sst.RestoreHttpProbe(ss)
			sst.Scale(ss, 0)

			By("Verifying that stateful set " + ssName + " was scaled down in reverse order")
			expectedOrder = []string{ssName + "-2", ssName + "-1", ssName + "-0"}
			ctx, cancel = watchtools.ContextWithOptionalTimeout(context.Background(), framework.StatefulSetTimeout)
			defer cancel()
			_, err = watchtools.UntilWithoutRetry(ctx, watcher, func(event watch.Event) (bool, error) {
				if event.Type != watch.Deleted {
					return false, nil
				}
				pod := event.Object.(*v1.Pod)
				if pod.Name == expectedOrder[0] {
					expectedOrder = expectedOrder[1:]
				}
				return len(expectedOrder) == 0, nil

			})
			Expect(err).NotTo(HaveOccurred())
		})

		/*
			Release : v1.9
			Testname: StatefulSet, Burst Scaling
			Description: StatefulSet MUST support the Parallel PodManagementPolicy for burst scaling. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("Burst scaling should run to completion even with unhealthy pods", func() {
			psLabels := klabels.Set(labels)

			By("Creating stateful set " + ssName + " in namespace " + ns)
			ss := framework.NewStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, psLabels)
			ss.Spec.PodManagementPolicy = apps.ParallelPodManagement
			sst := framework.NewStatefulSetTester(c)
			sst.SetHttpProbe(ss)
			ss, err := c.AppsV1().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())

			By("Waiting until all stateful set " + ssName + " replicas will be running in namespace " + ns)
			sst.WaitForRunningAndReady(*ss.Spec.Replicas, ss)

			By("Confirming that stateful set scale up will not halt with unhealthy stateful pod")
			sst.BreakHttpProbe(ss)
			sst.WaitForRunningAndNotReady(*ss.Spec.Replicas, ss)
			sst.WaitForStatusReadyReplicas(ss, 0)
			sst.UpdateReplicas(ss, 3)
			sst.ConfirmStatefulPodCount(3, ss, 10*time.Second, false)

			By("Scaling up stateful set " + ssName + " to 3 replicas and waiting until all of them will be running in namespace " + ns)
			sst.RestoreHttpProbe(ss)
			sst.WaitForRunningAndReady(3, ss)

			By("Scale down will not halt with unhealthy stateful pod")
			sst.BreakHttpProbe(ss)
			sst.WaitForStatusReadyReplicas(ss, 0)
			sst.WaitForRunningAndNotReady(3, ss)
			sst.UpdateReplicas(ss, 0)
			sst.ConfirmStatefulPodCount(0, ss, 10*time.Second, false)

			By("Scaling down stateful set " + ssName + " to 0 replicas and waiting until none of pods will run in namespace" + ns)
			sst.RestoreHttpProbe(ss)
			sst.Scale(ss, 0)
			sst.WaitForStatusReplicas(ss, 0)
		})

		/*
			Release : v1.9
			Testname: StatefulSet, Recreate Failed Pod
			Description: StatefulSet MUST delete and recreate Pods it owns that go into a Failed state, such as when they are rejected or evicted by a Node. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("Should recreate evicted statefulset", func() {
			podName := "test-pod"
			statefulPodName := ssName + "-0"
			By("Looking for a node to schedule stateful set and pod")
			nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
			node := nodes.Items[0]

			By("Creating pod with conflicting port in namespace " + f.Namespace.Name)
			conflictingPort := v1.ContainerPort{HostPort: 21017, ContainerPort: 21017, Name: "conflict"}
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "nginx",
							Image: imageutils.GetE2EImage(imageutils.Nginx),
							Ports: []v1.ContainerPort{conflictingPort},
						},
					},
					NodeName: node.Name,
				},
			}
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
			framework.ExpectNoError(err)

			By("Creating statefulset with conflicting port in namespace " + f.Namespace.Name)
			ss := framework.NewStatefulSet(ssName, f.Namespace.Name, headlessSvcName, 1, nil, nil, labels)
			statefulPodContainer := &ss.Spec.Template.Spec.Containers[0]
			statefulPodContainer.Ports = append(statefulPodContainer.Ports, conflictingPort)
			ss.Spec.Template.Spec.NodeName = node.Name
			_, err = f.ClientSet.AppsV1().StatefulSets(f.Namespace.Name).Create(ss)
			framework.ExpectNoError(err)

			By("Waiting until pod " + podName + " will start running in namespace " + f.Namespace.Name)
			if err := f.WaitForPodRunning(podName); err != nil {
				framework.Failf("Pod %v did not start running: %v", podName, err)
			}

			var initialStatefulPodUID types.UID
			By("Waiting until stateful pod " + statefulPodName + " will be recreated and deleted at least once in namespace " + f.Namespace.Name)
			w, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Watch(metav1.SingleObject(metav1.ObjectMeta{Name: statefulPodName}))
			framework.ExpectNoError(err)
			ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), framework.StatefulPodTimeout)
			defer cancel()
			// we need to get UID from pod in any state and wait until stateful set controller will remove pod atleast once
			_, err = watchtools.UntilWithoutRetry(ctx, w, func(event watch.Event) (bool, error) {
				pod := event.Object.(*v1.Pod)
				switch event.Type {
				case watch.Deleted:
					framework.Logf("Observed delete event for stateful pod %v in namespace %v", pod.Name, pod.Namespace)
					if initialStatefulPodUID == "" {
						return false, nil
					}
					return true, nil
				}
				framework.Logf("Observed stateful pod in namespace: %v, name: %v, uid: %v, status phase: %v. Waiting for statefulset controller to delete.",
					pod.Namespace, pod.Name, pod.UID, pod.Status.Phase)
				initialStatefulPodUID = pod.UID
				return false, nil
			})
			if err != nil {
				framework.Failf("Pod %v expected to be re-created at least once", statefulPodName)
			}

			By("Removing pod with conflicting port in namespace " + f.Namespace.Name)
			err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(pod.Name, metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)

			By("Waiting when stateful pod " + statefulPodName + " will be recreated in namespace " + f.Namespace.Name + " and will be in running state")
			// we may catch delete event, that's why we are waiting for running phase like this, and not with watchtools.UntilWithoutRetry
			Eventually(func() error {
				statefulPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(statefulPodName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				if statefulPod.Status.Phase != v1.PodRunning {
					return fmt.Errorf("Pod %v is not in running phase: %v", statefulPod.Name, statefulPod.Status.Phase)
				} else if statefulPod.UID == initialStatefulPodUID {
					return fmt.Errorf("Pod %v wasn't recreated: %v == %v", statefulPod.Name, statefulPod.UID, initialStatefulPodUID)
				}
				return nil
			}, framework.StatefulPodTimeout, 2*time.Second).Should(BeNil())
		})
		/* Comment it for now until scale sub-resource is finalized in ref:pull/53679 for scale-sub resource specific comment.
		It("should have a working scale subresource", func() {
			By("Creating statefulset " + ssName + " in namespace " + ns)
			ss := framework.NewStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, labels)
			sst := framework.NewStatefulSetTester(c)
			sst.SetHttpProbe(ss)
			ss, err := c.AppsV1beta1().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())
			sst.WaitForRunningAndReady(*ss.Spec.Replicas, ss)
			ss = sst.WaitForStatus(ss)

			By("getting scale subresource")
			scale := framework.NewStatefulSetScale(ss)
			scaleResult := &appsv1beta2.Scale{}

			err = c.AppsV1beta2().RESTClient().Get().AbsPath("/apis/apps/v1beta2").Namespace(ns).Resource("statefulsets").Name(ssName).SubResource("scale").Do().Into(scale)
			if err != nil {
				framework.Failf("Failed to get scale subresource: %v", err)
			}
			Expect(scale.Spec.Replicas).To(Equal(int32(1)))
			Expect(scale.Status.Replicas).To(Equal(int32(1)))

			By("updating a scale subresource")
			scale.ResourceVersion = "" //unconditionally update to 2 replicas
			scale.Spec.Replicas = 2
			err = c.AppsV1beta2().RESTClient().Put().AbsPath("/apis/apps/v1beta2").Namespace(ns).Resource("statefulsets").Name(ssName).SubResource("scale").Body(scale).Do().Into(scaleResult)
			if err != nil {
				framework.Failf("Failed to put scale subresource: %v", err)
			}
			Expect(scaleResult.Spec.Replicas).To(Equal(int32(2)))

			By("verifying the statefulset Spec.Replicas was modified")
			ss, err = c.AppsV1beta1().StatefulSets(ns).Get(ssName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("Failed to get statefulset resource: %v", err)
			}
			Expect(*(ss.Spec.Replicas)).To(Equal(int32(2)))
		})
		*/
	})

	framework.KubeDescribe("Deploy clustered applications [Feature:StatefulSet] [Slow]", func() {
		var sst *framework.StatefulSetTester
		var appTester *clusterAppTester

		BeforeEach(func() {
			sst = framework.NewStatefulSetTester(c)
			appTester = &clusterAppTester{tester: sst, ns: ns}
		})

		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				framework.DumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all statefulset in ns %v", ns)
			framework.DeleteAllStatefulSets(c, ns)
		})

		// Do not mark this as Conformance.
		// StatefulSet Conformance should not be dependent on specific applications.
		It("should creating a working zookeeper cluster", func() {
			appTester.statefulPod = &zookeeperTester{tester: sst}
			appTester.run()
		})

		// Do not mark this as Conformance.
		// StatefulSet Conformance should not be dependent on specific applications.
		It("should creating a working redis cluster", func() {
			appTester.statefulPod = &redisTester{tester: sst}
			appTester.run()
		})

		// Do not mark this as Conformance.
		// StatefulSet Conformance should not be dependent on specific applications.
		It("should creating a working mysql cluster", func() {
			appTester.statefulPod = &mysqlGaleraTester{tester: sst}
			appTester.run()
		})

		// Do not mark this as Conformance.
		// StatefulSet Conformance should not be dependent on specific applications.
		It("should creating a working CockroachDB cluster", func() {
			appTester.statefulPod = &cockroachDBTester{tester: sst}
			appTester.run()
		})
	})
})

func kubectlExecWithRetries(args ...string) (out string) {
	var err error
	for i := 0; i < 3; i++ {
		if out, err = framework.RunKubectl(args...); err == nil {
			return
		}
		framework.Logf("Retrying %v:\nerror %v\nstdout %v", args, err, out)
	}
	framework.Failf("Failed to execute \"%v\" with retries: %v", args, err)
	return
}

type statefulPodTester interface {
	deploy(ns string) *apps.StatefulSet
	write(statefulPodIndex int, kv map[string]string)
	read(statefulPodIndex int, key string) string
	name() string
}

type clusterAppTester struct {
	ns          string
	statefulPod statefulPodTester
	tester      *framework.StatefulSetTester
}

func (c *clusterAppTester) run() {
	By("Deploying " + c.statefulPod.name())
	ss := c.statefulPod.deploy(c.ns)

	By("Creating foo:bar in member with index 0")
	c.statefulPod.write(0, map[string]string{"foo": "bar"})

	switch c.statefulPod.(type) {
	case *mysqlGaleraTester:
		// Don't restart MySQL cluster since it doesn't handle restarts well
	default:
		if restartCluster {
			By("Restarting stateful set " + ss.Name)
			c.tester.Restart(ss)
			c.tester.WaitForRunningAndReady(*ss.Spec.Replicas, ss)
		}
	}

	By("Reading value under foo from member with index 2")
	if err := pollReadWithTimeout(c.statefulPod, 2, "foo", "bar"); err != nil {
		framework.Failf("%v", err)
	}
}

type zookeeperTester struct {
	ss     *apps.StatefulSet
	tester *framework.StatefulSetTester
}

func (z *zookeeperTester) name() string {
	return "zookeeper"
}

func (z *zookeeperTester) deploy(ns string) *apps.StatefulSet {
	z.ss = z.tester.CreateStatefulSet(zookeeperManifestPath, ns)
	return z.ss
}

func (z *zookeeperTester) write(statefulPodIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", z.ss.Name, statefulPodIndex)
	ns := fmt.Sprintf("--namespace=%v", z.ss.Namespace)
	for k, v := range kv {
		cmd := fmt.Sprintf("/opt/zookeeper/bin/zkCli.sh create /%v %v", k, v)
		framework.Logf(framework.RunKubectlOrDie("exec", ns, name, "--", "/bin/sh", "-c", cmd))
	}
}

func (z *zookeeperTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", z.ss.Name, statefulPodIndex)
	ns := fmt.Sprintf("--namespace=%v", z.ss.Namespace)
	cmd := fmt.Sprintf("/opt/zookeeper/bin/zkCli.sh get /%v", key)
	return lastLine(framework.RunKubectlOrDie("exec", ns, name, "--", "/bin/sh", "-c", cmd))
}

type mysqlGaleraTester struct {
	ss     *apps.StatefulSet
	tester *framework.StatefulSetTester
}

func (m *mysqlGaleraTester) name() string {
	return "mysql: galera"
}

func (m *mysqlGaleraTester) mysqlExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/usr/bin/mysql -u root -B -e '%v'", cmd)
	// TODO: Find a readiness probe for mysql that guarantees writes will
	// succeed and ditch retries. Current probe only reads, so there's a window
	// for a race.
	return kubectlExecWithRetries(fmt.Sprintf("--namespace=%v", ns), "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (m *mysqlGaleraTester) deploy(ns string) *apps.StatefulSet {
	m.ss = m.tester.CreateStatefulSet(mysqlGaleraManifestPath, ns)

	framework.Logf("Deployed statefulset %v, initializing database", m.ss.Name)
	for _, cmd := range []string{
		"create database statefulset;",
		"use statefulset; create table foo (k varchar(20), v varchar(20));",
	} {
		framework.Logf(m.mysqlExec(cmd, ns, fmt.Sprintf("%v-0", m.ss.Name)))
	}
	return m.ss
}

func (m *mysqlGaleraTester) write(statefulPodIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", m.ss.Name, statefulPodIndex)
	for k, v := range kv {
		cmd := fmt.Sprintf("use  statefulset; insert into foo (k, v) values (\"%v\", \"%v\");", k, v)
		framework.Logf(m.mysqlExec(cmd, m.ss.Namespace, name))
	}
}

func (m *mysqlGaleraTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", m.ss.Name, statefulPodIndex)
	return lastLine(m.mysqlExec(fmt.Sprintf("use statefulset; select v from foo where k=\"%v\";", key), m.ss.Namespace, name))
}

type redisTester struct {
	ss     *apps.StatefulSet
	tester *framework.StatefulSetTester
}

func (m *redisTester) name() string {
	return "redis: master/slave"
}

func (m *redisTester) redisExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/opt/redis/redis-cli -h %v %v", podName, cmd)
	return framework.RunKubectlOrDie(fmt.Sprintf("--namespace=%v", ns), "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (m *redisTester) deploy(ns string) *apps.StatefulSet {
	m.ss = m.tester.CreateStatefulSet(redisManifestPath, ns)
	return m.ss
}

func (m *redisTester) write(statefulPodIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", m.ss.Name, statefulPodIndex)
	for k, v := range kv {
		framework.Logf(m.redisExec(fmt.Sprintf("SET %v %v", k, v), m.ss.Namespace, name))
	}
}

func (m *redisTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", m.ss.Name, statefulPodIndex)
	return lastLine(m.redisExec(fmt.Sprintf("GET %v", key), m.ss.Namespace, name))
}

type cockroachDBTester struct {
	ss     *apps.StatefulSet
	tester *framework.StatefulSetTester
}

func (c *cockroachDBTester) name() string {
	return "CockroachDB"
}

func (c *cockroachDBTester) cockroachDBExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/cockroach/cockroach sql --insecure --host %s.cockroachdb -e \"%v\"", podName, cmd)
	return framework.RunKubectlOrDie(fmt.Sprintf("--namespace=%v", ns), "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (c *cockroachDBTester) deploy(ns string) *apps.StatefulSet {
	c.ss = c.tester.CreateStatefulSet(cockroachDBManifestPath, ns)
	framework.Logf("Deployed statefulset %v, initializing database", c.ss.Name)
	for _, cmd := range []string{
		"CREATE DATABASE IF NOT EXISTS foo;",
		"CREATE TABLE IF NOT EXISTS foo.bar (k STRING PRIMARY KEY, v STRING);",
	} {
		framework.Logf(c.cockroachDBExec(cmd, ns, fmt.Sprintf("%v-0", c.ss.Name)))
	}
	return c.ss
}

func (c *cockroachDBTester) write(statefulPodIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", c.ss.Name, statefulPodIndex)
	for k, v := range kv {
		cmd := fmt.Sprintf("UPSERT INTO foo.bar VALUES ('%v', '%v');", k, v)
		framework.Logf(c.cockroachDBExec(cmd, c.ss.Namespace, name))
	}
}
func (c *cockroachDBTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", c.ss.Name, statefulPodIndex)
	return lastLine(c.cockroachDBExec(fmt.Sprintf("SELECT v FROM foo.bar WHERE k='%v';", key), c.ss.Namespace, name))
}

func lastLine(out string) string {
	outLines := strings.Split(strings.Trim(out, "\n"), "\n")
	return outLines[len(outLines)-1]
}

func pollReadWithTimeout(statefulPod statefulPodTester, statefulPodNumber int, key, expectedVal string) error {
	err := wait.PollImmediate(time.Second, readTimeout, func() (bool, error) {
		val := statefulPod.read(statefulPodNumber, key)
		if val == "" {
			return false, nil
		} else if val != expectedVal {
			return false, fmt.Errorf("expected value %v, found %v", expectedVal, val)
		}
		return true, nil
	})

	if err == wait.ErrWaitTimeout {
		return fmt.Errorf("timed out when trying to read value for key %v from stateful pod %d", key, statefulPodNumber)
	}
	return err
}

// This function is used by two tests to test StatefulSet rollbacks: one using
// PVCs and one using no storage.
func rollbackTest(c clientset.Interface, ns string, ss *apps.StatefulSet) {
	sst := framework.NewStatefulSetTester(c)
	sst.SetHttpProbe(ss)
	ss, err := c.AppsV1().StatefulSets(ns).Create(ss)
	Expect(err).NotTo(HaveOccurred())
	sst.WaitForRunningAndReady(*ss.Spec.Replicas, ss)
	ss = sst.WaitForStatus(ss)
	currentRevision, updateRevision := ss.Status.CurrentRevision, ss.Status.UpdateRevision
	Expect(currentRevision).To(Equal(updateRevision),
		fmt.Sprintf("StatefulSet %s/%s created with update revision %s not equal to current revision %s",
			ss.Namespace, ss.Name, updateRevision, currentRevision))
	pods := sst.GetPodList(ss)
	for i := range pods.Items {
		Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(currentRevision),
			fmt.Sprintf("Pod %s/%s revision %s is not equal to current revision %s",
				pods.Items[i].Namespace,
				pods.Items[i].Name,
				pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
				currentRevision))
	}
	sst.SortStatefulPods(pods)
	err = sst.BreakPodHttpProbe(ss, &pods.Items[1])
	Expect(err).NotTo(HaveOccurred())
	ss, pods = sst.WaitForPodNotReady(ss, pods.Items[1].Name)
	newImage := NewNginxImage
	oldImage := ss.Spec.Template.Spec.Containers[0].Image

	By(fmt.Sprintf("Updating StatefulSet template: update image from %s to %s", oldImage, newImage))
	Expect(oldImage).NotTo(Equal(newImage), "Incorrect test setup: should update to a different image")
	ss, err = framework.UpdateStatefulSetWithRetries(c, ns, ss.Name, func(update *apps.StatefulSet) {
		update.Spec.Template.Spec.Containers[0].Image = newImage
	})
	Expect(err).NotTo(HaveOccurred())

	By("Creating a new revision")
	ss = sst.WaitForStatus(ss)
	currentRevision, updateRevision = ss.Status.CurrentRevision, ss.Status.UpdateRevision
	Expect(currentRevision).NotTo(Equal(updateRevision),
		"Current revision should not equal update revision during rolling update")

	By("Updating Pods in reverse ordinal order")
	pods = sst.GetPodList(ss)
	sst.SortStatefulPods(pods)
	err = sst.RestorePodHttpProbe(ss, &pods.Items[1])
	Expect(err).NotTo(HaveOccurred())
	ss, pods = sst.WaitForPodReady(ss, pods.Items[1].Name)
	ss, pods = sst.WaitForRollingUpdate(ss)
	Expect(ss.Status.CurrentRevision).To(Equal(updateRevision),
		fmt.Sprintf("StatefulSet %s/%s current revision %s does not equal update revision %s on update completion",
			ss.Namespace,
			ss.Name,
			ss.Status.CurrentRevision,
			updateRevision))
	for i := range pods.Items {
		Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(newImage),
			fmt.Sprintf(" Pod %s/%s has image %s not have new image %s",
				pods.Items[i].Namespace,
				pods.Items[i].Name,
				pods.Items[i].Spec.Containers[0].Image,
				newImage))
		Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(updateRevision),
			fmt.Sprintf("Pod %s/%s revision %s is not equal to update revision %s",
				pods.Items[i].Namespace,
				pods.Items[i].Name,
				pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
				updateRevision))
	}

	By("Rolling back to a previous revision")
	err = sst.BreakPodHttpProbe(ss, &pods.Items[1])
	Expect(err).NotTo(HaveOccurred())
	ss, pods = sst.WaitForPodNotReady(ss, pods.Items[1].Name)
	priorRevision := currentRevision
	currentRevision, updateRevision = ss.Status.CurrentRevision, ss.Status.UpdateRevision
	ss, err = framework.UpdateStatefulSetWithRetries(c, ns, ss.Name, func(update *apps.StatefulSet) {
		update.Spec.Template.Spec.Containers[0].Image = oldImage
	})
	Expect(err).NotTo(HaveOccurred())
	ss = sst.WaitForStatus(ss)
	currentRevision, updateRevision = ss.Status.CurrentRevision, ss.Status.UpdateRevision
	Expect(currentRevision).NotTo(Equal(updateRevision),
		"Current revision should not equal update revision during roll back")
	Expect(priorRevision).To(Equal(updateRevision),
		"Prior revision should equal update revision during roll back")

	By("Rolling back update in reverse ordinal order")
	pods = sst.GetPodList(ss)
	sst.SortStatefulPods(pods)
	sst.RestorePodHttpProbe(ss, &pods.Items[1])
	ss, pods = sst.WaitForPodReady(ss, pods.Items[1].Name)
	ss, pods = sst.WaitForRollingUpdate(ss)
	Expect(ss.Status.CurrentRevision).To(Equal(priorRevision),
		fmt.Sprintf("StatefulSet %s/%s current revision %s does not equal prior revision %s on rollback completion",
			ss.Namespace,
			ss.Name,
			ss.Status.CurrentRevision,
			updateRevision))

	for i := range pods.Items {
		Expect(pods.Items[i].Spec.Containers[0].Image).To(Equal(oldImage),
			fmt.Sprintf("Pod %s/%s has image %s not equal to previous image %s",
				pods.Items[i].Namespace,
				pods.Items[i].Name,
				pods.Items[i].Spec.Containers[0].Image,
				oldImage))
		Expect(pods.Items[i].Labels[apps.StatefulSetRevisionLabel]).To(Equal(priorRevision),
			fmt.Sprintf("Pod %s/%s revision %s is not equal to prior revision %s",
				pods.Items[i].Namespace,
				pods.Items[i].Name,
				pods.Items[i].Labels[apps.StatefulSetRevisionLabel],
				priorRevision))
	}
}
