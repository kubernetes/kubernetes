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
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	klabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2estatefulset "k8s.io/kubernetes/test/e2e/framework/statefulset"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
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

	// statefulSetPoll is a poll interval for StatefulSet tests
	statefulSetPoll = 10 * time.Second
	// statefulSetTimeout is a timeout interval for StatefulSet operations
	statefulSetTimeout = 10 * time.Minute
	// statefulPodTimeout is a timeout for stateful pods to change state
	statefulPodTimeout = 5 * time.Minute
)

var httpProbe = &v1.Probe{
	ProbeHandler: v1.ProbeHandler{
		HTTPGet: &v1.HTTPGetAction{
			Path: "/index.html",
			Port: intstr.IntOrString{IntVal: 80},
		},
	},
	PeriodSeconds:    1,
	SuccessThreshold: 1,
	FailureThreshold: 1,
}

// GCE Quota requirements: 3 pds, one per stateful pod manifest declared above.
// GCE Api requirements: nodes and master need storage r/w permissions.
var _ = SIGDescribe("StatefulSet", func() {
	f := framework.NewDefaultFramework("statefulset")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var ns string
	var c clientset.Interface

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	ginkgo.Describe("Basic StatefulSet functionality [StatefulSetBasic]", func() {
		ssName := "ss"
		labels := map[string]string{
			"foo": "bar",
			"baz": "blah",
		}
		headlessSvcName := "test"
		var statefulPodMounts, podMounts []v1.VolumeMount
		var ss *appsv1.StatefulSet

		ginkgo.BeforeEach(func() {
			statefulPodMounts = []v1.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
			podMounts = []v1.VolumeMount{{Name: "home", MountPath: "/home"}}
			ss = e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 2, statefulPodMounts, podMounts, labels)

			ginkgo.By("Creating service " + headlessSvcName + " in namespace " + ns)
			headlessService := e2eservice.CreateServiceSpec(headlessSvcName, "", true, labels)
			_, err := c.CoreV1().Services(ns).Create(context.TODO(), headlessService, metav1.CreateOptions{})
			framework.ExpectNoError(err)
		})

		ginkgo.AfterEach(func() {
			if ginkgo.CurrentSpecReport().Failed() {
				framework.DumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all statefulset in ns %v", ns)
			e2estatefulset.DeleteAllStatefulSets(c, ns)
		})

		// This can't be Conformance yet because it depends on a default
		// StorageClass and a dynamic provisioner.
		ginkgo.It("should provide basic identity", func() {
			ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
			e2epv.SkipIfNoDefaultStorageClass(c)
			*(ss.Spec.Replicas) = 3
			e2estatefulset.PauseNewPods(ss)

			_, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Saturating stateful set " + ss.Name)
			e2estatefulset.Saturate(c, ss)

			ginkgo.By("Verifying statefulset mounted data directory is usable")
			framework.ExpectNoError(e2estatefulset.CheckMount(c, ss, "/data"))

			ginkgo.By("Verifying statefulset provides a stable hostname for each pod")
			framework.ExpectNoError(e2estatefulset.CheckHostname(c, ss))

			ginkgo.By("Verifying statefulset set proper service name")
			framework.ExpectNoError(e2estatefulset.CheckServiceName(ss, headlessSvcName))

			cmd := "echo $(hostname) | dd of=/data/hostname conv=fsync"
			ginkgo.By("Running " + cmd + " in all stateful pods")
			framework.ExpectNoError(e2estatefulset.ExecInStatefulPods(c, ss, cmd))

			ginkgo.By("Restarting statefulset " + ss.Name)
			e2estatefulset.Restart(c, ss)
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)

			ginkgo.By("Verifying statefulset mounted data directory is usable")
			framework.ExpectNoError(e2estatefulset.CheckMount(c, ss, "/data"))

			cmd = "if [ \"$(cat /data/hostname)\" = \"$(hostname)\" ]; then exit 0; else exit 1; fi"
			ginkgo.By("Running " + cmd + " in all stateful pods")
			framework.ExpectNoError(e2estatefulset.ExecInStatefulPods(c, ss, cmd))
		})

		// This can't be Conformance yet because it depends on a default
		// StorageClass and a dynamic provisioner.
		ginkgo.It("should adopt matching orphans and release non-matching pods", func() {
			ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
			e2epv.SkipIfNoDefaultStorageClass(c)
			*(ss.Spec.Replicas) = 1
			e2estatefulset.PauseNewPods(ss)

			// Replace ss with the one returned from Create() so it has the UID.
			// Save Kind since it won't be populated in the returned ss.
			kind := ss.Kind
			ss, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			ss.Kind = kind

			ginkgo.By("Saturating stateful set " + ss.Name)
			e2estatefulset.Saturate(c, ss)
			pods := e2estatefulset.GetPodList(c, ss)
			gomega.Expect(pods.Items).To(gomega.HaveLen(int(*ss.Spec.Replicas)))

			ginkgo.By("Checking that stateful set pods are created with ControllerRef")
			pod := pods.Items[0]
			controllerRef := metav1.GetControllerOf(&pod)
			gomega.Expect(controllerRef).ToNot(gomega.BeNil())
			framework.ExpectEqual(controllerRef.Kind, ss.Kind)
			framework.ExpectEqual(controllerRef.Name, ss.Name)
			framework.ExpectEqual(controllerRef.UID, ss.UID)

			ginkgo.By("Orphaning one of the stateful set's pods")
			f.PodClient().Update(pod.Name, func(pod *v1.Pod) {
				pod.OwnerReferences = nil
			})

			ginkgo.By("Checking that the stateful set readopts the pod")
			gomega.Expect(e2epod.WaitForPodCondition(c, pod.Namespace, pod.Name, "adopted", statefulSetTimeout,
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
			)).To(gomega.Succeed(), "wait for pod %q to be readopted", pod.Name)

			ginkgo.By("Removing the labels from one of the stateful set's pods")
			prevLabels := pod.Labels
			f.PodClient().Update(pod.Name, func(pod *v1.Pod) {
				pod.Labels = nil
			})

			ginkgo.By("Checking that the stateful set releases the pod")
			gomega.Expect(e2epod.WaitForPodCondition(c, pod.Namespace, pod.Name, "released", statefulSetTimeout,
				func(pod *v1.Pod) (bool, error) {
					controllerRef := metav1.GetControllerOf(pod)
					if controllerRef != nil {
						return false, nil
					}
					return true, nil
				},
			)).To(gomega.Succeed(), "wait for pod %q to be released", pod.Name)

			// If we don't do this, the test leaks the Pod and PVC.
			ginkgo.By("Readding labels to the stateful set's pod")
			f.PodClient().Update(pod.Name, func(pod *v1.Pod) {
				pod.Labels = prevLabels
			})

			ginkgo.By("Checking that the stateful set readopts the pod")
			gomega.Expect(e2epod.WaitForPodCondition(c, pod.Namespace, pod.Name, "adopted", statefulSetTimeout,
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
			)).To(gomega.Succeed(), "wait for pod %q to be readopted", pod.Name)
		})

		// This can't be Conformance yet because it depends on a default
		// StorageClass and a dynamic provisioner.
		ginkgo.It("should not deadlock when a pod's predecessor fails", func() {
			ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
			e2epv.SkipIfNoDefaultStorageClass(c)
			*(ss.Spec.Replicas) = 2
			e2estatefulset.PauseNewPods(ss)

			_, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			e2estatefulset.WaitForRunning(c, 1, 0, ss)

			ginkgo.By("Resuming stateful pod at index 0.")
			e2estatefulset.ResumeNextPod(c, ss)

			ginkgo.By("Waiting for stateful pod at index 1 to enter running.")
			e2estatefulset.WaitForRunning(c, 2, 1, ss)

			// Now we have 1 healthy and 1 unhealthy stateful pod. Deleting the healthy stateful pod should *not*
			// create a new stateful pod till the remaining stateful pod becomes healthy, which won't happen till
			// we set the healthy bit.

			ginkgo.By("Deleting healthy stateful pod at index 0.")
			deleteStatefulPodAtIndex(c, 0, ss)

			ginkgo.By("Confirming stateful pod at index 0 is recreated.")
			e2estatefulset.WaitForRunning(c, 2, 1, ss)

			ginkgo.By("Resuming stateful pod at index 1.")
			e2estatefulset.ResumeNextPod(c, ss)

			ginkgo.By("Confirming all stateful pods in statefulset are created.")
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)
		})

		// This can't be Conformance yet because it depends on a default
		// StorageClass and a dynamic provisioner.
		ginkgo.It("should perform rolling updates and roll backs of template modifications with PVCs", func() {
			ginkgo.By("Creating a new StatefulSet with PVCs")
			e2epv.SkipIfNoDefaultStorageClass(c)
			*(ss.Spec.Replicas) = 3
			rollbackTest(c, ns, ss)
		})

		/*
		   Release: v1.9
		   Testname: StatefulSet, Rolling Update
		   Description: StatefulSet MUST support the RollingUpdate strategy to automatically replace Pods one at a time when the Pod template changes. The StatefulSet's status MUST indicate the CurrentRevision and UpdateRevision. If the template is changed to match a prior revision, StatefulSet MUST detect this as a rollback instead of creating a new revision. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("should perform rolling updates and roll backs of template modifications", func() {
			ginkgo.By("Creating a new StatefulSet")
			ss := e2estatefulset.NewStatefulSet("ss2", ns, headlessSvcName, 3, nil, nil, labels)
			rollbackTest(c, ns, ss)
		})

		/*
		   Release: v1.9
		   Testname: StatefulSet, Rolling Update with Partition
		   Description: StatefulSet's RollingUpdate strategy MUST support the Partition parameter for canaries and phased rollouts. If a Pod is deleted while a rolling update is in progress, StatefulSet MUST restore the Pod without violating the Partition. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("should perform canary updates and phased rolling updates of template modifications", func() {
			ginkgo.By("Creating a new StatefulSet")
			ss := e2estatefulset.NewStatefulSet("ss2", ns, headlessSvcName, 3, nil, nil, labels)
			setHTTPProbe(ss)
			ss.Spec.UpdateStrategy = appsv1.StatefulSetUpdateStrategy{
				Type: appsv1.RollingUpdateStatefulSetStrategyType,
				RollingUpdate: func() *appsv1.RollingUpdateStatefulSetStrategy {
					return &appsv1.RollingUpdateStatefulSetStrategy{
						Partition: func() *int32 {
							i := int32(3)
							return &i
						}()}
				}(),
			}
			ss, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)
			ss = waitForStatus(c, ss)
			currentRevision, updateRevision := ss.Status.CurrentRevision, ss.Status.UpdateRevision
			framework.ExpectEqual(currentRevision, updateRevision, fmt.Sprintf("StatefulSet %s/%s created with update revision %s not equal to current revision %s",
				ss.Namespace, ss.Name, updateRevision, currentRevision))
			pods := e2estatefulset.GetPodList(c, ss)
			for i := range pods.Items {
				framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], currentRevision, fmt.Sprintf("Pod %s/%s revision %s is not equal to currentRevision %s",
					pods.Items[i].Namespace,
					pods.Items[i].Name,
					pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
					currentRevision))
			}
			newImage := NewWebserverImage
			oldImage := ss.Spec.Template.Spec.Containers[0].Image

			ginkgo.By(fmt.Sprintf("Updating stateful set template: update image from %s to %s", oldImage, newImage))
			framework.ExpectNotEqual(oldImage, newImage, "Incorrect test setup: should update to a different image")
			ss, err = updateStatefulSetWithRetries(c, ns, ss.Name, func(update *appsv1.StatefulSet) {
				update.Spec.Template.Spec.Containers[0].Image = newImage
			})
			framework.ExpectNoError(err)

			ginkgo.By("Creating a new revision")
			ss = waitForStatus(c, ss)
			currentRevision, updateRevision = ss.Status.CurrentRevision, ss.Status.UpdateRevision
			framework.ExpectNotEqual(currentRevision, updateRevision, "Current revision should not equal update revision during rolling update")

			ginkgo.By("Not applying an update when the partition is greater than the number of replicas")
			for i := range pods.Items {
				framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, oldImage, fmt.Sprintf("Pod %s/%s has image %s not equal to current image %s",
					pods.Items[i].Namespace,
					pods.Items[i].Name,
					pods.Items[i].Spec.Containers[0].Image,
					oldImage))
				framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], currentRevision, fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
					pods.Items[i].Namespace,
					pods.Items[i].Name,
					pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
					currentRevision))
			}

			ginkgo.By("Performing a canary update")
			ss.Spec.UpdateStrategy = appsv1.StatefulSetUpdateStrategy{
				Type: appsv1.RollingUpdateStatefulSetStrategyType,
				RollingUpdate: func() *appsv1.RollingUpdateStatefulSetStrategy {
					return &appsv1.RollingUpdateStatefulSetStrategy{
						Partition: func() *int32 {
							i := int32(2)
							return &i
						}()}
				}(),
			}
			ss, err = updateStatefulSetWithRetries(c, ns, ss.Name, func(update *appsv1.StatefulSet) {
				update.Spec.UpdateStrategy = appsv1.StatefulSetUpdateStrategy{
					Type: appsv1.RollingUpdateStatefulSetStrategyType,
					RollingUpdate: func() *appsv1.RollingUpdateStatefulSetStrategy {
						return &appsv1.RollingUpdateStatefulSetStrategy{
							Partition: func() *int32 {
								i := int32(2)
								return &i
							}()}
					}(),
				}
			})
			framework.ExpectNoError(err)
			ss, pods = waitForPartitionedRollingUpdate(c, ss)
			for i := range pods.Items {
				if i < int(*ss.Spec.UpdateStrategy.RollingUpdate.Partition) {
					framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, oldImage, fmt.Sprintf("Pod %s/%s has image %s not equal to current image %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Spec.Containers[0].Image,
						oldImage))
					framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], currentRevision, fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
						currentRevision))
				} else {
					framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, newImage, fmt.Sprintf("Pod %s/%s has image %s not equal to new image  %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Spec.Containers[0].Image,
						newImage))
					framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], updateRevision, fmt.Sprintf("Pod %s/%s has revision %s not equal to new revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
						updateRevision))
				}
			}

			ginkgo.By("Restoring Pods to the correct revision when they are deleted")
			deleteStatefulPodAtIndex(c, 0, ss)
			deleteStatefulPodAtIndex(c, 2, ss)
			e2estatefulset.WaitForRunningAndReady(c, 3, ss)
			ss = getStatefulSet(c, ss.Namespace, ss.Name)
			pods = e2estatefulset.GetPodList(c, ss)
			for i := range pods.Items {
				if i < int(*ss.Spec.UpdateStrategy.RollingUpdate.Partition) {
					framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, oldImage, fmt.Sprintf("Pod %s/%s has image %s not equal to current image %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Spec.Containers[0].Image,
						oldImage))
					framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], currentRevision, fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
						currentRevision))
				} else {
					framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, newImage, fmt.Sprintf("Pod %s/%s has image %s not equal to new image  %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Spec.Containers[0].Image,
						newImage))
					framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], updateRevision, fmt.Sprintf("Pod %s/%s has revision %s not equal to new revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
						updateRevision))
				}
			}

			ginkgo.By("Performing a phased rolling update")
			for i := int(*ss.Spec.UpdateStrategy.RollingUpdate.Partition) - 1; i >= 0; i-- {
				ss, err = updateStatefulSetWithRetries(c, ns, ss.Name, func(update *appsv1.StatefulSet) {
					update.Spec.UpdateStrategy = appsv1.StatefulSetUpdateStrategy{
						Type: appsv1.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: func() *appsv1.RollingUpdateStatefulSetStrategy {
							j := int32(i)
							return &appsv1.RollingUpdateStatefulSetStrategy{
								Partition: &j,
							}
						}(),
					}
				})
				framework.ExpectNoError(err)
				ss, pods = waitForPartitionedRollingUpdate(c, ss)
				for i := range pods.Items {
					if i < int(*ss.Spec.UpdateStrategy.RollingUpdate.Partition) {
						framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, oldImage, fmt.Sprintf("Pod %s/%s has image %s not equal to current image %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Spec.Containers[0].Image,
							oldImage))
						framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], currentRevision, fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
							currentRevision))
					} else {
						framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, newImage, fmt.Sprintf("Pod %s/%s has image %s not equal to new image  %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Spec.Containers[0].Image,
							newImage))
						framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], updateRevision, fmt.Sprintf("Pod %s/%s has revision %s not equal to new revision %s",
							pods.Items[i].Namespace,
							pods.Items[i].Name,
							pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
							updateRevision))
					}
				}
			}
			framework.ExpectEqual(ss.Status.CurrentRevision, updateRevision, fmt.Sprintf("StatefulSet %s/%s current revision %s does not equal update revision %s on update completion",
				ss.Namespace,
				ss.Name,
				ss.Status.CurrentRevision,
				updateRevision))

		})

		// Do not mark this as Conformance.
		// The legacy OnDelete strategy only exists for backward compatibility with pre-v1 APIs.
		ginkgo.It("should implement legacy replacement when the update strategy is OnDelete", func() {
			ginkgo.By("Creating a new StatefulSet")
			ss := e2estatefulset.NewStatefulSet("ss2", ns, headlessSvcName, 3, nil, nil, labels)
			setHTTPProbe(ss)
			ss.Spec.UpdateStrategy = appsv1.StatefulSetUpdateStrategy{
				Type: appsv1.OnDeleteStatefulSetStrategyType,
			}
			ss, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)
			ss = waitForStatus(c, ss)
			currentRevision, updateRevision := ss.Status.CurrentRevision, ss.Status.UpdateRevision
			framework.ExpectEqual(currentRevision, updateRevision, fmt.Sprintf("StatefulSet %s/%s created with update revision %s not equal to current revision %s",
				ss.Namespace, ss.Name, updateRevision, currentRevision))
			pods := e2estatefulset.GetPodList(c, ss)
			for i := range pods.Items {
				framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], currentRevision, fmt.Sprintf("Pod %s/%s revision %s is not equal to current revision %s",
					pods.Items[i].Namespace,
					pods.Items[i].Name,
					pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
					currentRevision))
			}

			ginkgo.By("Restoring Pods to the current revision")
			deleteStatefulPodAtIndex(c, 0, ss)
			deleteStatefulPodAtIndex(c, 1, ss)
			deleteStatefulPodAtIndex(c, 2, ss)
			e2estatefulset.WaitForRunningAndReady(c, 3, ss)
			ss = getStatefulSet(c, ss.Namespace, ss.Name)
			pods = e2estatefulset.GetPodList(c, ss)
			for i := range pods.Items {
				framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], currentRevision, fmt.Sprintf("Pod %s/%s revision %s is not equal to current revision %s",
					pods.Items[i].Namespace,
					pods.Items[i].Name,
					pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
					currentRevision))
			}
			newImage := NewWebserverImage
			oldImage := ss.Spec.Template.Spec.Containers[0].Image

			ginkgo.By(fmt.Sprintf("Updating stateful set template: update image from %s to %s", oldImage, newImage))
			framework.ExpectNotEqual(oldImage, newImage, "Incorrect test setup: should update to a different image")
			ss, err = updateStatefulSetWithRetries(c, ns, ss.Name, func(update *appsv1.StatefulSet) {
				update.Spec.Template.Spec.Containers[0].Image = newImage
			})
			framework.ExpectNoError(err)

			ginkgo.By("Creating a new revision")
			ss = waitForStatus(c, ss)
			currentRevision, updateRevision = ss.Status.CurrentRevision, ss.Status.UpdateRevision
			framework.ExpectNotEqual(currentRevision, updateRevision, "Current revision should not equal update revision during rolling update")

			ginkgo.By("Recreating Pods at the new revision")
			deleteStatefulPodAtIndex(c, 0, ss)
			deleteStatefulPodAtIndex(c, 1, ss)
			deleteStatefulPodAtIndex(c, 2, ss)
			e2estatefulset.WaitForRunningAndReady(c, 3, ss)
			ss = getStatefulSet(c, ss.Namespace, ss.Name)
			pods = e2estatefulset.GetPodList(c, ss)
			for i := range pods.Items {
				framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, newImage, fmt.Sprintf("Pod %s/%s has image %s not equal to new image %s",
					pods.Items[i].Namespace,
					pods.Items[i].Name,
					pods.Items[i].Spec.Containers[0].Image,
					newImage))
				framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], updateRevision, fmt.Sprintf("Pod %s/%s has revision %s not equal to current revision %s",
					pods.Items[i].Namespace,
					pods.Items[i].Name,
					pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
					updateRevision))
			}
		})

		/*
		   Release: v1.9
		   Testname: StatefulSet, Scaling
		   Description: StatefulSet MUST create Pods in ascending order by ordinal index when scaling up, and delete Pods in descending order when scaling down. Scaling up or down MUST pause if any Pods belonging to the StatefulSet are unhealthy. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("Scaling should happen in predictable order and halt if any stateful pod is unhealthy [Slow]", func() {
			psLabels := klabels.Set(labels)
			w := &cache.ListWatch{
				WatchFunc: func(options metav1.ListOptions) (i watch.Interface, e error) {
					options.LabelSelector = psLabels.AsSelector().String()
					return f.ClientSet.CoreV1().Pods(ns).Watch(context.TODO(), options)
				},
			}
			ginkgo.By("Initializing watcher for selector " + psLabels.String())
			pl, err := f.ClientSet.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{
				LabelSelector: psLabels.AsSelector().String(),
			})
			framework.ExpectNoError(err)

			// Verify that statuful set will be scaled up in order.
			wg := sync.WaitGroup{}
			var orderErr error
			wg.Add(1)
			go func() {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()

				expectedOrder := []string{ssName + "-0", ssName + "-1", ssName + "-2"}
				ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), statefulSetTimeout)
				defer cancel()

				_, orderErr = watchtools.Until(ctx, pl.ResourceVersion, w, func(event watch.Event) (bool, error) {
					if event.Type != watch.Added {
						return false, nil
					}
					pod := event.Object.(*v1.Pod)
					if pod.Name == expectedOrder[0] {
						expectedOrder = expectedOrder[1:]
					}
					return len(expectedOrder) == 0, nil
				})
			}()

			ginkgo.By("Creating stateful set " + ssName + " in namespace " + ns)
			ss := e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, psLabels)
			setHTTPProbe(ss)
			ss, err = c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Waiting until all stateful set " + ssName + " replicas will be running in namespace " + ns)
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)

			ginkgo.By("Confirming that stateful set scale up will halt with unhealthy stateful pod")
			breakHTTPProbe(c, ss)
			waitForRunningAndNotReady(c, *ss.Spec.Replicas, ss)
			e2estatefulset.WaitForStatusReadyReplicas(c, ss, 0)
			e2estatefulset.UpdateReplicas(c, ss, 3)
			confirmStatefulPodCount(c, 1, ss, 10*time.Second, true)

			ginkgo.By("Scaling up stateful set " + ssName + " to 3 replicas and waiting until all of them will be running in namespace " + ns)
			restoreHTTPProbe(c, ss)
			e2estatefulset.WaitForRunningAndReady(c, 3, ss)

			ginkgo.By("Verifying that stateful set " + ssName + " was scaled up in order")
			wg.Wait()
			framework.ExpectNoError(orderErr)

			ginkgo.By("Scale down will halt with unhealthy stateful pod")
			pl, err = f.ClientSet.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{
				LabelSelector: psLabels.AsSelector().String(),
			})
			framework.ExpectNoError(err)

			// Verify that statuful set will be scaled down in order.
			wg.Add(1)
			go func() {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()

				expectedOrder := []string{ssName + "-2", ssName + "-1", ssName + "-0"}
				ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), statefulSetTimeout)
				defer cancel()

				_, orderErr = watchtools.Until(ctx, pl.ResourceVersion, w, func(event watch.Event) (bool, error) {
					if event.Type != watch.Deleted {
						return false, nil
					}
					pod := event.Object.(*v1.Pod)
					if pod.Name == expectedOrder[0] {
						expectedOrder = expectedOrder[1:]
					}
					return len(expectedOrder) == 0, nil
				})
			}()

			breakHTTPProbe(c, ss)
			e2estatefulset.WaitForStatusReadyReplicas(c, ss, 0)
			waitForRunningAndNotReady(c, 3, ss)
			e2estatefulset.UpdateReplicas(c, ss, 0)
			confirmStatefulPodCount(c, 3, ss, 10*time.Second, true)

			ginkgo.By("Scaling down stateful set " + ssName + " to 0 replicas and waiting until none of pods will run in namespace" + ns)
			restoreHTTPProbe(c, ss)
			e2estatefulset.Scale(c, ss, 0)

			ginkgo.By("Verifying that stateful set " + ssName + " was scaled down in reverse order")
			wg.Wait()
			framework.ExpectNoError(orderErr)
		})

		/*
		   Release: v1.9
		   Testname: StatefulSet, Burst Scaling
		   Description: StatefulSet MUST support the Parallel PodManagementPolicy for burst scaling. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("Burst scaling should run to completion even with unhealthy pods [Slow]", func() {
			psLabels := klabels.Set(labels)

			ginkgo.By("Creating stateful set " + ssName + " in namespace " + ns)
			ss := e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, psLabels)
			ss.Spec.PodManagementPolicy = appsv1.ParallelPodManagement
			setHTTPProbe(ss)
			ss, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Waiting until all stateful set " + ssName + " replicas will be running in namespace " + ns)
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)

			ginkgo.By("Confirming that stateful set scale up will not halt with unhealthy stateful pod")
			breakHTTPProbe(c, ss)
			waitForRunningAndNotReady(c, *ss.Spec.Replicas, ss)
			e2estatefulset.WaitForStatusReadyReplicas(c, ss, 0)
			e2estatefulset.UpdateReplicas(c, ss, 3)
			confirmStatefulPodCount(c, 3, ss, 10*time.Second, false)

			ginkgo.By("Scaling up stateful set " + ssName + " to 3 replicas and waiting until all of them will be running in namespace " + ns)
			restoreHTTPProbe(c, ss)
			e2estatefulset.WaitForRunningAndReady(c, 3, ss)

			ginkgo.By("Scale down will not halt with unhealthy stateful pod")
			breakHTTPProbe(c, ss)
			e2estatefulset.WaitForStatusReadyReplicas(c, ss, 0)
			waitForRunningAndNotReady(c, 3, ss)
			e2estatefulset.UpdateReplicas(c, ss, 0)
			confirmStatefulPodCount(c, 0, ss, 10*time.Second, false)

			ginkgo.By("Scaling down stateful set " + ssName + " to 0 replicas and waiting until none of pods will run in namespace" + ns)
			restoreHTTPProbe(c, ss)
			e2estatefulset.Scale(c, ss, 0)
			e2estatefulset.WaitForStatusReplicas(c, ss, 0)
		})

		/*
		   Release: v1.9
		   Testname: StatefulSet, Recreate Failed Pod
		   Description: StatefulSet MUST delete and recreate Pods it owns that go into a Failed state, such as when they are rejected or evicted by a Node. This test does not depend on a preexisting default StorageClass or a dynamic provisioner.
		*/
		framework.ConformanceIt("Should recreate evicted statefulset", func() {
			podName := "test-pod"
			statefulPodName := ssName + "-0"
			ginkgo.By("Looking for a node to schedule stateful set and pod")
			node, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
			framework.ExpectNoError(err)

			ginkgo.By("Creating pod with conflicting port in namespace " + f.Namespace.Name)
			conflictingPort := v1.ContainerPort{HostPort: 21017, ContainerPort: 21017, Name: "conflict"}
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "webserver",
							Image: imageutils.GetE2EImage(imageutils.Httpd),
							Ports: []v1.ContainerPort{conflictingPort},
						},
					},
					NodeName: node.Name,
				},
			}
			pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			ginkgo.By("Waiting until pod " + podName + " will start running in namespace " + f.Namespace.Name)
			if err := e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, podName, f.Namespace.Name); err != nil {
				framework.Failf("Pod %v did not start running: %v", podName, err)
			}

			ginkgo.By("Creating statefulset with conflicting port in namespace " + f.Namespace.Name)
			ss := e2estatefulset.NewStatefulSet(ssName, f.Namespace.Name, headlessSvcName, 1, nil, nil, labels)
			statefulPodContainer := &ss.Spec.Template.Spec.Containers[0]
			statefulPodContainer.Ports = append(statefulPodContainer.Ports, conflictingPort)
			ss.Spec.Template.Spec.NodeName = node.Name
			_, err = f.ClientSet.AppsV1().StatefulSets(f.Namespace.Name).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			var initialStatefulPodUID types.UID
			ginkgo.By("Waiting until stateful pod " + statefulPodName + " will be recreated and deleted at least once in namespace " + f.Namespace.Name)

			fieldSelector := fields.OneTermEqualSelector("metadata.name", statefulPodName).String()
			pl, err := f.ClientSet.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{
				FieldSelector: fieldSelector,
			})
			framework.ExpectNoError(err)
			if len(pl.Items) > 0 {
				pod := pl.Items[0]
				framework.Logf("Observed stateful pod in namespace: %v, name: %v, uid: %v, status phase: %v. Waiting for statefulset controller to delete.",
					pod.Namespace, pod.Name, pod.UID, pod.Status.Phase)
				initialStatefulPodUID = pod.UID
			}

			lw := &cache.ListWatch{
				WatchFunc: func(options metav1.ListOptions) (i watch.Interface, e error) {
					options.FieldSelector = fieldSelector
					return f.ClientSet.CoreV1().Pods(f.Namespace.Name).Watch(context.TODO(), options)
				},
			}
			ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), statefulPodTimeout)
			defer cancel()
			// we need to get UID from pod in any state and wait until stateful set controller will remove pod at least once
			_, err = watchtools.Until(ctx, pl.ResourceVersion, lw, func(event watch.Event) (bool, error) {
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

			ginkgo.By("Removing pod with conflicting port in namespace " + f.Namespace.Name)
			err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)

			ginkgo.By("Waiting when stateful pod " + statefulPodName + " will be recreated in namespace " + f.Namespace.Name + " and will be in running state")
			// we may catch delete event, that's why we are waiting for running phase like this, and not with watchtools.UntilWithoutRetry
			gomega.Eventually(func() error {
				statefulPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), statefulPodName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				if statefulPod.Status.Phase != v1.PodRunning {
					return fmt.Errorf("pod %v is not in running phase: %v", statefulPod.Name, statefulPod.Status.Phase)
				} else if statefulPod.UID == initialStatefulPodUID {
					return fmt.Errorf("pod %v wasn't recreated: %v == %v", statefulPod.Name, statefulPod.UID, initialStatefulPodUID)
				}
				return nil
			}, statefulPodTimeout, 2*time.Second).Should(gomega.BeNil())
		})

		/*
			Release: v1.16, v1.21
			Testname: StatefulSet resource Replica scaling
			Description: Create a StatefulSet resource.
			Newly created StatefulSet resource MUST have a scale of one.
			Bring the scale of the StatefulSet resource up to two. StatefulSet scale MUST be at two replicas.
		*/
		framework.ConformanceIt("should have a working scale subresource", func() {
			ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
			ss := e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, labels)
			setHTTPProbe(ss)
			ss, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)
			waitForStatus(c, ss)

			ginkgo.By("getting scale subresource")
			scale, err := c.AppsV1().StatefulSets(ns).GetScale(context.TODO(), ssName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("Failed to get scale subresource: %v", err)
			}
			framework.ExpectEqual(scale.Spec.Replicas, int32(1))
			framework.ExpectEqual(scale.Status.Replicas, int32(1))

			ginkgo.By("updating a scale subresource")
			scale.ResourceVersion = "" // indicate the scale update should be unconditional
			scale.Spec.Replicas = 2
			scaleResult, err := c.AppsV1().StatefulSets(ns).UpdateScale(context.TODO(), ssName, scale, metav1.UpdateOptions{})
			if err != nil {
				framework.Failf("Failed to put scale subresource: %v", err)
			}
			framework.ExpectEqual(scaleResult.Spec.Replicas, int32(2))

			ginkgo.By("verifying the statefulset Spec.Replicas was modified")
			ss, err = c.AppsV1().StatefulSets(ns).Get(context.TODO(), ssName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("Failed to get statefulset resource: %v", err)
			}
			framework.ExpectEqual(*(ss.Spec.Replicas), int32(2))

			ginkgo.By("Patch a scale subresource")
			scale.ResourceVersion = "" // indicate the scale update should be unconditional
			scale.Spec.Replicas = 4    // should be 2 after "UpdateScale" operation, now Patch to 4
			ssScalePatchPayload, err := json.Marshal(autoscalingv1.Scale{
				Spec: autoscalingv1.ScaleSpec{
					Replicas: scale.Spec.Replicas,
				},
			})
			framework.ExpectNoError(err, "Could not Marshal JSON for patch payload")

			_, err = c.AppsV1().StatefulSets(ns).Patch(context.TODO(), ssName, types.StrategicMergePatchType, []byte(ssScalePatchPayload), metav1.PatchOptions{}, "scale")
			framework.ExpectNoError(err, "Failed to patch stateful set: %v", err)

			ginkgo.By("verifying the statefulset Spec.Replicas was modified")
			ss, err = c.AppsV1().StatefulSets(ns).Get(context.TODO(), ssName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to get statefulset resource: %v", err)
			framework.ExpectEqual(*(ss.Spec.Replicas), int32(4), "statefulset should have 4 replicas")
		})

		/*
			Release: v1.22
			Testname: StatefulSet, list, patch and delete a collection of StatefulSets
			Description: When a StatefulSet is created it MUST succeed. It
			MUST succeed when listing StatefulSets via a label selector. It
			MUST succeed when patching a StatefulSet. It MUST succeed when
			deleting the StatefulSet via deleteCollection.
		*/
		framework.ConformanceIt("should list, patch and delete a collection of StatefulSets", func() {

			ssPatchReplicas := int32(2)
			ssPatchImage := imageutils.GetE2EImage(imageutils.Pause)
			one := int64(1)
			ssName := "test-ss"

			// Define StatefulSet Labels
			ssPodLabels := map[string]string{
				"name": "sample-pod",
				"pod":  WebserverImageName,
			}
			ss := e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, ssPodLabels)
			setHTTPProbe(ss)
			ss, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)
			waitForStatus(c, ss)

			ginkgo.By("patching the StatefulSet")
			ssPatch, err := json.Marshal(map[string]interface{}{
				"metadata": map[string]interface{}{
					"labels": map[string]string{"test-ss": "patched"},
				},
				"spec": map[string]interface{}{
					"replicas": ssPatchReplicas,
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"TerminationGracePeriodSeconds": &one,
							"containers": [1]map[string]interface{}{{
								"name":  ssName,
								"image": ssPatchImage,
							}},
						},
					},
				},
			})
			framework.ExpectNoError(err, "failed to Marshal StatefulSet JSON patch")
			_, err = f.ClientSet.AppsV1().StatefulSets(ns).Patch(context.TODO(), ssName, types.StrategicMergePatchType, []byte(ssPatch), metav1.PatchOptions{})
			framework.ExpectNoError(err, "failed to patch Set")
			ss, err = c.AppsV1().StatefulSets(ns).Get(context.TODO(), ssName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to get statefulset resource: %v", err)
			framework.ExpectEqual(*(ss.Spec.Replicas), ssPatchReplicas, "statefulset should have 2 replicas")
			framework.ExpectEqual(ss.Spec.Template.Spec.Containers[0].Image, ssPatchImage, "statefulset not using ssPatchImage. Is using %v", ss.Spec.Template.Spec.Containers[0].Image)
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)
			waitForStatus(c, ss)

			ginkgo.By("Listing all StatefulSets")
			ssList, err := c.AppsV1().StatefulSets("").List(context.TODO(), metav1.ListOptions{LabelSelector: "test-ss=patched"})
			framework.ExpectNoError(err, "failed to list StatefulSets")
			framework.ExpectEqual(len(ssList.Items), 1, "filtered list wasn't found")

			ginkgo.By("Delete all of the StatefulSets")
			err = c.AppsV1().StatefulSets(ns).DeleteCollection(context.TODO(), metav1.DeleteOptions{GracePeriodSeconds: &one}, metav1.ListOptions{LabelSelector: "test-ss=patched"})
			framework.ExpectNoError(err, "failed to delete StatefulSets")

			ginkgo.By("Verify that StatefulSets have been deleted")
			ssList, err = c.AppsV1().StatefulSets("").List(context.TODO(), metav1.ListOptions{LabelSelector: "test-ss=patched"})
			framework.ExpectNoError(err, "failed to list StatefulSets")
			framework.ExpectEqual(len(ssList.Items), 0, "filtered list should have no Statefulsets")
		})

		/*
			Release: v1.22
			Testname: StatefulSet, status sub-resource
			Description: When a StatefulSet is created it MUST succeed.
			Attempt to read, update and patch its status sub-resource; all
			mutating sub-resource operations MUST be visible to subsequent reads.
		*/
		framework.ConformanceIt("should validate Statefulset Status endpoints", func() {
			ssClient := c.AppsV1().StatefulSets(ns)
			labelSelector := "e2e=testing"

			w := &cache.ListWatch{
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					options.LabelSelector = labelSelector
					return ssClient.Watch(context.TODO(), options)
				},
			}
			ssList, err := c.AppsV1().StatefulSets("").List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector})
			framework.ExpectNoError(err, "failed to list StatefulSets")

			ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
			ss := e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, labels)
			setHTTPProbe(ss)
			ss, err = c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)
			waitForStatus(c, ss)

			ginkgo.By("Patch Statefulset to include a label")
			payload := []byte(`{"metadata":{"labels":{"e2e":"testing"}}}`)
			ss, err = ssClient.Patch(context.TODO(), ssName, types.StrategicMergePatchType, payload, metav1.PatchOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Getting /status")
			ssResource := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "statefulsets"}
			ssStatusUnstructured, err := f.DynamicClient.Resource(ssResource).Namespace(ns).Get(context.TODO(), ssName, metav1.GetOptions{}, "status")
			framework.ExpectNoError(err, "Failed to fetch the status of replica set %s in namespace %s", ssName, ns)
			ssStatusBytes, err := json.Marshal(ssStatusUnstructured)
			framework.ExpectNoError(err, "Failed to marshal unstructured response. %v", err)

			var ssStatus appsv1.StatefulSet
			err = json.Unmarshal(ssStatusBytes, &ssStatus)
			framework.ExpectNoError(err, "Failed to unmarshal JSON bytes to a Statefulset object type")
			framework.Logf("StatefulSet %s has Conditions: %#v", ssName, ssStatus.Status.Conditions)

			ginkgo.By("updating the StatefulSet Status")
			var statusToUpdate, updatedStatus *appsv1.StatefulSet

			err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
				statusToUpdate, err = ssClient.Get(context.TODO(), ssName, metav1.GetOptions{})
				framework.ExpectNoError(err, "Unable to retrieve statefulset %s", ssName)

				statusToUpdate.Status.Conditions = append(statusToUpdate.Status.Conditions, appsv1.StatefulSetCondition{
					Type:    "StatusUpdate",
					Status:  "True",
					Reason:  "E2E",
					Message: "Set from e2e test",
				})

				updatedStatus, err = ssClient.UpdateStatus(context.TODO(), statusToUpdate, metav1.UpdateOptions{})
				return err
			})
			framework.ExpectNoError(err, "Failed to update status. %v", err)
			framework.Logf("updatedStatus.Conditions: %#v", updatedStatus.Status.Conditions)

			ginkgo.By("watching for the statefulset status to be updated")

			ctx, cancel := context.WithTimeout(context.Background(), statefulSetTimeout)
			defer cancel()

			_, err = watchtools.Until(ctx, ssList.ResourceVersion, w, func(event watch.Event) (bool, error) {

				if e, ok := event.Object.(*appsv1.StatefulSet); ok {
					found := e.ObjectMeta.Name == ss.ObjectMeta.Name &&
						e.ObjectMeta.Namespace == ss.ObjectMeta.Namespace &&
						e.ObjectMeta.Labels["e2e"] == ss.ObjectMeta.Labels["e2e"]
					if !found {
						framework.Logf("Observed Statefulset %v in namespace %v with annotations: %v & Conditions: %v", ss.ObjectMeta.Name, ss.ObjectMeta.Namespace, ss.Annotations, ss.Status.Conditions)
						return false, nil
					}
					for _, cond := range e.Status.Conditions {
						if cond.Type == "StatusUpdate" &&
							cond.Reason == "E2E" &&
							cond.Message == "Set from e2e test" {
							framework.Logf("Found Statefulset %v in namespace %v with labels: %v annotations: %v & Conditions: %v", ss.ObjectMeta.Name, ss.ObjectMeta.Namespace, ss.ObjectMeta.Labels, ss.Annotations, cond)
							return found, nil
						}
						framework.Logf("Observed Statefulset %v in namespace %v with annotations: %v & Conditions: %v", ss.ObjectMeta.Name, ss.ObjectMeta.Namespace, ss.Annotations, cond)
					}
				}
				object := strings.Split(fmt.Sprintf("%v", event.Object), "{")[0]
				framework.Logf("Observed %v event: %+v", object, event.Type)
				return false, nil
			})
			framework.ExpectNoError(err, "failed to locate Statefulset %v in namespace %v", ss.ObjectMeta.Name, ns)
			framework.Logf("Statefulset %s has an updated status", ssName)

			ginkgo.By("patching the Statefulset Status")
			payload = []byte(`{"status":{"conditions":[{"type":"StatusPatched","status":"True"}]}}`)
			framework.Logf("Patch payload: %v", string(payload))

			patchedStatefulSet, err := ssClient.Patch(context.TODO(), ssName, types.MergePatchType, payload, metav1.PatchOptions{}, "status")
			framework.ExpectNoError(err, "Failed to patch status. %v", err)
			framework.Logf("Patched status conditions: %#v", patchedStatefulSet.Status.Conditions)

			ginkgo.By("watching for the Statefulset status to be patched")
			ctx, cancel = context.WithTimeout(context.Background(), statefulSetTimeout)

			_, err = watchtools.Until(ctx, ssList.ResourceVersion, w, func(event watch.Event) (bool, error) {

				defer cancel()
				if e, ok := event.Object.(*appsv1.StatefulSet); ok {
					found := e.ObjectMeta.Name == ss.ObjectMeta.Name &&
						e.ObjectMeta.Namespace == ss.ObjectMeta.Namespace &&
						e.ObjectMeta.Labels["e2e"] == ss.ObjectMeta.Labels["e2e"]
					if !found {
						framework.Logf("Observed Statefulset %v in namespace %v with annotations: %v & Conditions: %v", ss.ObjectMeta.Name, ss.ObjectMeta.Namespace, ss.Annotations, ss.Status.Conditions)
						return false, nil
					}
					for _, cond := range e.Status.Conditions {
						if cond.Type == "StatusPatched" {
							framework.Logf("Found Statefulset %v in namespace %v with labels: %v annotations: %v & Conditions: %v", ss.ObjectMeta.Name, ss.ObjectMeta.Namespace, ss.ObjectMeta.Labels, ss.Annotations, cond)
							return found, nil
						}
						framework.Logf("Observed Statefulset %v in namespace %v with annotations: %v & Conditions: %v", ss.ObjectMeta.Name, ss.ObjectMeta.Namespace, ss.Annotations, cond)
					}
				}
				object := strings.Split(fmt.Sprintf("%v", event.Object), "{")[0]
				framework.Logf("Observed %v event: %+v", object, event.Type)
				return false, nil
			})
		})
	})

	ginkgo.Describe("Deploy clustered applications [Feature:StatefulSet] [Slow]", func() {
		var appTester *clusterAppTester

		ginkgo.BeforeEach(func() {
			appTester = &clusterAppTester{client: c, ns: ns}
		})

		ginkgo.AfterEach(func() {
			if ginkgo.CurrentSpecReport().Failed() {
				framework.DumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all statefulset in ns %v", ns)
			e2estatefulset.DeleteAllStatefulSets(c, ns)
		})

		// Do not mark this as Conformance.
		// StatefulSet Conformance should not be dependent on specific applications.
		ginkgo.It("should creating a working zookeeper cluster", func() {
			e2epv.SkipIfNoDefaultStorageClass(c)
			appTester.statefulPod = &zookeeperTester{client: c}
			appTester.run()
		})

		// Do not mark this as Conformance.
		// StatefulSet Conformance should not be dependent on specific applications.
		ginkgo.It("should creating a working redis cluster", func() {
			e2epv.SkipIfNoDefaultStorageClass(c)
			appTester.statefulPod = &redisTester{client: c}
			appTester.run()
		})

		// Do not mark this as Conformance.
		// StatefulSet Conformance should not be dependent on specific applications.
		ginkgo.It("should creating a working mysql cluster", func() {
			e2epv.SkipIfNoDefaultStorageClass(c)
			appTester.statefulPod = &mysqlGaleraTester{client: c}
			appTester.run()
		})

		// Do not mark this as Conformance.
		// StatefulSet Conformance should not be dependent on specific applications.
		ginkgo.It("should creating a working CockroachDB cluster", func() {
			e2epv.SkipIfNoDefaultStorageClass(c)
			appTester.statefulPod = &cockroachDBTester{client: c}
			appTester.run()
		})
	})

	// Make sure minReadySeconds is honored
	// Don't mark it as conformance yet
	ginkgo.It("MinReadySeconds should be honored when enabled", func() {
		ssName := "test-ss"
		headlessSvcName := "test"
		// Define StatefulSet Labels
		ssPodLabels := map[string]string{
			"name": "sample-pod",
			"pod":  WebserverImageName,
		}
		ss := e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, ssPodLabels)
		setHTTPProbe(ss)
		ss, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		e2estatefulset.WaitForStatusAvailableReplicas(c, ss, 1)
	})

	ginkgo.It("AvailableReplicas should get updated accordingly when MinReadySeconds is enabled", func() {
		ssName := "test-ss"
		headlessSvcName := "test"
		// Define StatefulSet Labels
		ssPodLabels := map[string]string{
			"name": "sample-pod",
			"pod":  WebserverImageName,
		}
		ss := e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 2, nil, nil, ssPodLabels)
		ss.Spec.MinReadySeconds = 30
		setHTTPProbe(ss)
		ss, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		e2estatefulset.WaitForStatusAvailableReplicas(c, ss, 0)
		// let's check that the availableReplicas have still not updated
		time.Sleep(5 * time.Second)
		ss, err = c.AppsV1().StatefulSets(ns).Get(context.TODO(), ss.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		if ss.Status.AvailableReplicas != 0 {
			framework.Failf("invalid number of availableReplicas: expected=%v received=%v", 0, ss.Status.AvailableReplicas)
		}
		e2estatefulset.WaitForStatusAvailableReplicas(c, ss, 2)

		ss, err = updateStatefulSetWithRetries(c, ns, ss.Name, func(update *appsv1.StatefulSet) {
			update.Spec.MinReadySeconds = 3600
		})
		framework.ExpectNoError(err)
		// We don't expect replicas to be updated till 1 hour, so the availableReplicas should be 0
		e2estatefulset.WaitForStatusAvailableReplicas(c, ss, 0)

		ss, err = updateStatefulSetWithRetries(c, ns, ss.Name, func(update *appsv1.StatefulSet) {
			update.Spec.MinReadySeconds = 0
		})
		framework.ExpectNoError(err)
		e2estatefulset.WaitForStatusAvailableReplicas(c, ss, 2)

		ginkgo.By("check availableReplicas are shown in status")
		out, err := framework.RunKubectl(ns, "get", "statefulset", ss.Name, "-o=yaml")
		framework.ExpectNoError(err)
		if !strings.Contains(out, "availableReplicas: 2") {
			framework.Failf("invalid number of availableReplicas: expected=%v received=%v", 2, out)
		}
	})

	ginkgo.Describe("Non-retain StatefulSetPersistentVolumeClaimPolicy [Feature:StatefulSetAutoDeletePVC]", func() {
		ssName := "ss"
		labels := map[string]string{
			"foo": "bar",
			"baz": "blah",
		}
		headlessSvcName := "test"
		var statefulPodMounts, podMounts []v1.VolumeMount
		var ss *appsv1.StatefulSet

		ginkgo.BeforeEach(func() {
			statefulPodMounts = []v1.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
			podMounts = []v1.VolumeMount{{Name: "home", MountPath: "/home"}}
			ss = e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 2, statefulPodMounts, podMounts, labels)

			ginkgo.By("Creating service " + headlessSvcName + " in namespace " + ns)
			headlessService := e2eservice.CreateServiceSpec(headlessSvcName, "", true, labels)
			_, err := c.CoreV1().Services(ns).Create(context.TODO(), headlessService, metav1.CreateOptions{})
			framework.ExpectNoError(err)
		})

		ginkgo.AfterEach(func() {
			if ginkgo.CurrentSpecReport().Failed() {
				framework.DumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all statefulset in ns %v", ns)
			e2estatefulset.DeleteAllStatefulSets(c, ns)
		})

		ginkgo.It("should delete PVCs with a WhenDeleted policy", func() {
			e2epv.SkipIfNoDefaultStorageClass(c)
			ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 3
			ss.Spec.PersistentVolumeClaimRetentionPolicy = &appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
			}
			_, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Confirming all 3 PVCs exist with their owner refs")
			err = verifyStatefulSetPVCsExistWithOwnerRefs(c, ss, []int{0, 1, 2}, true, false)
			framework.ExpectNoError(err)

			ginkgo.By("Deleting stateful set " + ss.Name)
			err = c.AppsV1().StatefulSets(ns).Delete(context.TODO(), ss.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Verifying PVCs deleted")
			err = verifyStatefulSetPVCsExist(c, ss, []int{})
			framework.ExpectNoError(err)
		})

		ginkgo.It("should delete PVCs with a OnScaledown policy", func() {
			e2epv.SkipIfNoDefaultStorageClass(c)
			ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 3
			ss.Spec.PersistentVolumeClaimRetentionPolicy = &appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenScaled: appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
			}
			_, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Confirming all 3 PVCs exist")
			err = verifyStatefulSetPVCsExist(c, ss, []int{0, 1, 2})
			framework.ExpectNoError(err)

			ginkgo.By("Scaling stateful set " + ss.Name + " to one replica")
			ss, err = e2estatefulset.Scale(c, ss, 1)
			framework.ExpectNoError(err)

			ginkgo.By("Verifying all but one PVC deleted")
			err = verifyStatefulSetPVCsExist(c, ss, []int{0})
			framework.ExpectNoError(err)
		})

		ginkgo.It("should delete PVCs after adopting pod (WhenDeleted)", func() {
			e2epv.SkipIfNoDefaultStorageClass(c)
			ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 3
			ss.Spec.PersistentVolumeClaimRetentionPolicy = &appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
			}
			_, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Confirming all 3 PVCs exist with their owner refs")
			err = verifyStatefulSetPVCsExistWithOwnerRefs(c, ss, []int{0, 1, 2}, true, false)
			framework.ExpectNoError(err)

			ginkgo.By("Orphaning the 3rd pod")
			patch, err := json.Marshal(metav1.ObjectMeta{
				OwnerReferences: []metav1.OwnerReference{},
			})
			framework.ExpectNoError(err, "Could not Marshal JSON for patch payload")
			_, err = c.CoreV1().Pods(ns).Patch(context.TODO(), fmt.Sprintf("%s-2", ss.Name), types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{}, "")
			framework.ExpectNoError(err, "Could not patch payload")

			ginkgo.By("Deleting stateful set " + ss.Name)
			err = c.AppsV1().StatefulSets(ns).Delete(context.TODO(), ss.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Verifying PVCs deleted")
			err = verifyStatefulSetPVCsExist(c, ss, []int{})
			framework.ExpectNoError(err)
		})

		ginkgo.It("should delete PVCs after adopting pod (WhenScaled) [Feature:StatefulSetAutoDeletePVC]", func() {
			e2epv.SkipIfNoDefaultStorageClass(c)
			ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 3
			ss.Spec.PersistentVolumeClaimRetentionPolicy = &appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenScaled: appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
			}
			_, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Confirming all 3 PVCs exist")
			err = verifyStatefulSetPVCsExist(c, ss, []int{0, 1, 2})
			framework.ExpectNoError(err)

			ginkgo.By("Orphaning the 3rd pod")
			patch, err := json.Marshal(metav1.ObjectMeta{
				OwnerReferences: []metav1.OwnerReference{},
			})
			framework.ExpectNoError(err, "Could not Marshal JSON for patch payload")
			_, err = c.CoreV1().Pods(ns).Patch(context.TODO(), fmt.Sprintf("%s-2", ss.Name), types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{}, "")
			framework.ExpectNoError(err, "Could not patch payload")

			ginkgo.By("Scaling stateful set " + ss.Name + " to one replica")
			ss, err = e2estatefulset.Scale(c, ss, 1)
			framework.ExpectNoError(err)

			ginkgo.By("Verifying all but one PVC deleted")
			err = verifyStatefulSetPVCsExist(c, ss, []int{0})
			framework.ExpectNoError(err)
		})
	})
})

func kubectlExecWithRetries(ns string, args ...string) (out string) {
	var err error
	for i := 0; i < 3; i++ {
		if out, err = framework.RunKubectl(ns, args...); err == nil {
			return
		}
		framework.Logf("Retrying %v:\nerror %v\nstdout %v", args, err, out)
	}
	framework.Failf("Failed to execute \"%v\" with retries: %v", args, err)
	return
}

type statefulPodTester interface {
	deploy(ns string) *appsv1.StatefulSet
	write(statefulPodIndex int, kv map[string]string)
	read(statefulPodIndex int, key string) string
	name() string
}

type clusterAppTester struct {
	ns          string
	statefulPod statefulPodTester
	client      clientset.Interface
}

func (c *clusterAppTester) run() {
	ginkgo.By("Deploying " + c.statefulPod.name())
	ss := c.statefulPod.deploy(c.ns)

	ginkgo.By("Creating foo:bar in member with index 0")
	c.statefulPod.write(0, map[string]string{"foo": "bar"})

	switch c.statefulPod.(type) {
	case *mysqlGaleraTester:
		// Don't restart MySQL cluster since it doesn't handle restarts well
	default:
		if restartCluster {
			ginkgo.By("Restarting stateful set " + ss.Name)
			e2estatefulset.Restart(c.client, ss)
			e2estatefulset.WaitForRunningAndReady(c.client, *ss.Spec.Replicas, ss)
		}
	}

	ginkgo.By("Reading value under foo from member with index 2")
	if err := pollReadWithTimeout(c.statefulPod, 2, "foo", "bar"); err != nil {
		framework.Failf("%v", err)
	}
}

type zookeeperTester struct {
	ss     *appsv1.StatefulSet
	client clientset.Interface
}

func (z *zookeeperTester) name() string {
	return "zookeeper"
}

func (z *zookeeperTester) deploy(ns string) *appsv1.StatefulSet {
	z.ss = e2estatefulset.CreateStatefulSet(z.client, zookeeperManifestPath, ns)
	return z.ss
}

func (z *zookeeperTester) write(statefulPodIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", z.ss.Name, statefulPodIndex)
	for k, v := range kv {
		cmd := fmt.Sprintf("/opt/zookeeper/bin/zkCli.sh create /%v %v", k, v)
		framework.Logf(framework.RunKubectlOrDie(z.ss.Namespace, "exec", name, "--", "/bin/sh", "-c", cmd))
	}
}

func (z *zookeeperTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", z.ss.Name, statefulPodIndex)
	cmd := fmt.Sprintf("/opt/zookeeper/bin/zkCli.sh get /%v", key)
	return lastLine(framework.RunKubectlOrDie(z.ss.Namespace, "exec", name, "--", "/bin/sh", "-c", cmd))
}

type mysqlGaleraTester struct {
	ss     *appsv1.StatefulSet
	client clientset.Interface
}

func (m *mysqlGaleraTester) name() string {
	return "mysql: galera"
}

func (m *mysqlGaleraTester) mysqlExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/usr/bin/mysql -u root -B -e '%v'", cmd)
	// TODO: Find a readiness probe for mysql that guarantees writes will
	// succeed and ditch retries. Current probe only reads, so there's a window
	// for a race.
	return kubectlExecWithRetries(ns, "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (m *mysqlGaleraTester) deploy(ns string) *appsv1.StatefulSet {
	m.ss = e2estatefulset.CreateStatefulSet(m.client, mysqlGaleraManifestPath, ns)

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
		cmd := fmt.Sprintf("use statefulset; insert into foo (k, v) values (\"%v\", \"%v\");", k, v)
		framework.Logf(m.mysqlExec(cmd, m.ss.Namespace, name))
	}
}

func (m *mysqlGaleraTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", m.ss.Name, statefulPodIndex)
	return lastLine(m.mysqlExec(fmt.Sprintf("use statefulset; select v from foo where k=\"%v\";", key), m.ss.Namespace, name))
}

type redisTester struct {
	ss     *appsv1.StatefulSet
	client clientset.Interface
}

func (m *redisTester) name() string {
	return "redis: master/slave"
}

func (m *redisTester) redisExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/opt/redis/redis-cli -h %v %v", podName, cmd)
	return framework.RunKubectlOrDie(ns, "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (m *redisTester) deploy(ns string) *appsv1.StatefulSet {
	m.ss = e2estatefulset.CreateStatefulSet(m.client, redisManifestPath, ns)
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
	ss     *appsv1.StatefulSet
	client clientset.Interface
}

func (c *cockroachDBTester) name() string {
	return "CockroachDB"
}

func (c *cockroachDBTester) cockroachDBExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/cockroach/cockroach sql --insecure --host %s.cockroachdb -e \"%v\"", podName, cmd)
	return framework.RunKubectlOrDie(ns, "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (c *cockroachDBTester) deploy(ns string) *appsv1.StatefulSet {
	c.ss = e2estatefulset.CreateStatefulSet(c.client, cockroachDBManifestPath, ns)
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
func rollbackTest(c clientset.Interface, ns string, ss *appsv1.StatefulSet) {
	setHTTPProbe(ss)
	ss, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ss, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	e2estatefulset.WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)
	ss = waitForStatus(c, ss)
	currentRevision, updateRevision := ss.Status.CurrentRevision, ss.Status.UpdateRevision
	framework.ExpectEqual(currentRevision, updateRevision, fmt.Sprintf("StatefulSet %s/%s created with update revision %s not equal to current revision %s",
		ss.Namespace, ss.Name, updateRevision, currentRevision))
	pods := e2estatefulset.GetPodList(c, ss)
	for i := range pods.Items {
		framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], currentRevision, fmt.Sprintf("Pod %s/%s revision %s is not equal to current revision %s",
			pods.Items[i].Namespace,
			pods.Items[i].Name,
			pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
			currentRevision))
	}
	e2estatefulset.SortStatefulPods(pods)
	err = breakPodHTTPProbe(ss, &pods.Items[1])
	framework.ExpectNoError(err)
	ss, _ = waitForPodNotReady(c, ss, pods.Items[1].Name)
	newImage := NewWebserverImage
	oldImage := ss.Spec.Template.Spec.Containers[0].Image

	ginkgo.By(fmt.Sprintf("Updating StatefulSet template: update image from %s to %s", oldImage, newImage))
	framework.ExpectNotEqual(oldImage, newImage, "Incorrect test setup: should update to a different image")
	ss, err = updateStatefulSetWithRetries(c, ns, ss.Name, func(update *appsv1.StatefulSet) {
		update.Spec.Template.Spec.Containers[0].Image = newImage
	})
	framework.ExpectNoError(err)

	ginkgo.By("Creating a new revision")
	ss = waitForStatus(c, ss)
	currentRevision, updateRevision = ss.Status.CurrentRevision, ss.Status.UpdateRevision
	framework.ExpectNotEqual(currentRevision, updateRevision, "Current revision should not equal update revision during rolling update")

	ginkgo.By("Updating Pods in reverse ordinal order")
	pods = e2estatefulset.GetPodList(c, ss)
	e2estatefulset.SortStatefulPods(pods)
	err = restorePodHTTPProbe(ss, &pods.Items[1])
	framework.ExpectNoError(err)
	ss, _ = e2estatefulset.WaitForPodReady(c, ss, pods.Items[1].Name)
	ss, pods = waitForRollingUpdate(c, ss)
	framework.ExpectEqual(ss.Status.CurrentRevision, updateRevision, fmt.Sprintf("StatefulSet %s/%s current revision %s does not equal update revision %s on update completion",
		ss.Namespace,
		ss.Name,
		ss.Status.CurrentRevision,
		updateRevision))
	for i := range pods.Items {
		framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, newImage, fmt.Sprintf(" Pod %s/%s has image %s not have new image %s",
			pods.Items[i].Namespace,
			pods.Items[i].Name,
			pods.Items[i].Spec.Containers[0].Image,
			newImage))
		framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], updateRevision, fmt.Sprintf("Pod %s/%s revision %s is not equal to update revision %s",
			pods.Items[i].Namespace,
			pods.Items[i].Name,
			pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
			updateRevision))
	}

	ginkgo.By("Rolling back to a previous revision")
	err = breakPodHTTPProbe(ss, &pods.Items[1])
	framework.ExpectNoError(err)
	ss, _ = waitForPodNotReady(c, ss, pods.Items[1].Name)
	priorRevision := currentRevision
	ss, err = updateStatefulSetWithRetries(c, ns, ss.Name, func(update *appsv1.StatefulSet) {
		update.Spec.Template.Spec.Containers[0].Image = oldImage
	})
	framework.ExpectNoError(err)
	ss = waitForStatus(c, ss)
	currentRevision, updateRevision = ss.Status.CurrentRevision, ss.Status.UpdateRevision
	framework.ExpectEqual(priorRevision, updateRevision, "Prior revision should equal update revision during roll back")
	framework.ExpectNotEqual(currentRevision, updateRevision, "Current revision should not equal update revision during roll back")

	ginkgo.By("Rolling back update in reverse ordinal order")
	pods = e2estatefulset.GetPodList(c, ss)
	e2estatefulset.SortStatefulPods(pods)
	restorePodHTTPProbe(ss, &pods.Items[1])
	ss, _ = e2estatefulset.WaitForPodReady(c, ss, pods.Items[1].Name)
	ss, pods = waitForRollingUpdate(c, ss)
	framework.ExpectEqual(ss.Status.CurrentRevision, priorRevision, fmt.Sprintf("StatefulSet %s/%s current revision %s does not equal prior revision %s on rollback completion",
		ss.Namespace,
		ss.Name,
		ss.Status.CurrentRevision,
		updateRevision))

	for i := range pods.Items {
		framework.ExpectEqual(pods.Items[i].Spec.Containers[0].Image, oldImage, fmt.Sprintf("Pod %s/%s has image %s not equal to previous image %s",
			pods.Items[i].Namespace,
			pods.Items[i].Name,
			pods.Items[i].Spec.Containers[0].Image,
			oldImage))
		framework.ExpectEqual(pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel], priorRevision, fmt.Sprintf("Pod %s/%s revision %s is not equal to prior revision %s",
			pods.Items[i].Namespace,
			pods.Items[i].Name,
			pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel],
			priorRevision))
	}
}

// confirmStatefulPodCount asserts that the current number of Pods in ss is count, waiting up to timeout for ss to
// to scale to count.
func confirmStatefulPodCount(c clientset.Interface, count int, ss *appsv1.StatefulSet, timeout time.Duration, hard bool) {
	start := time.Now()
	deadline := start.Add(timeout)
	for t := time.Now(); t.Before(deadline); t = time.Now() {
		podList := e2estatefulset.GetPodList(c, ss)
		statefulPodCount := len(podList.Items)
		if statefulPodCount != count {
			e2epod.LogPodStates(podList.Items)
			if hard {
				framework.Failf("StatefulSet %v scaled unexpectedly scaled to %d -> %d replicas", ss.Name, count, len(podList.Items))
			} else {
				framework.Logf("StatefulSet %v has not reached scale %d, at %d", ss.Name, count, statefulPodCount)
			}
			time.Sleep(1 * time.Second)
			continue
		}
		framework.Logf("Verifying statefulset %v doesn't scale past %d for another %+v", ss.Name, count, deadline.Sub(t))
		time.Sleep(1 * time.Second)
	}
}

// setHTTPProbe sets the pod template's ReadinessProbe for Webserver StatefulSet containers.
// This probe can then be controlled with BreakHTTPProbe() and RestoreHTTPProbe().
// Note that this cannot be used together with PauseNewPods().
func setHTTPProbe(ss *appsv1.StatefulSet) {
	ss.Spec.Template.Spec.Containers[0].ReadinessProbe = httpProbe
}

// breakHTTPProbe breaks the readiness probe for Nginx StatefulSet containers in ss.
func breakHTTPProbe(c clientset.Interface, ss *appsv1.StatefulSet) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /usr/local/apache2/htdocs%v /tmp/ || true", path)
	return e2estatefulset.ExecInStatefulPods(c, ss, cmd)
}

// breakPodHTTPProbe breaks the readiness probe for Nginx StatefulSet containers in one pod.
func breakPodHTTPProbe(ss *appsv1.StatefulSet, pod *v1.Pod) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /usr/local/apache2/htdocs%v /tmp/ || true", path)
	stdout, err := framework.RunHostCmdWithRetries(pod.Namespace, pod.Name, cmd, statefulSetPoll, statefulPodTimeout)
	framework.Logf("stdout of %v on %v: %v", cmd, pod.Name, stdout)
	return err
}

// restoreHTTPProbe restores the readiness probe for Nginx StatefulSet containers in ss.
func restoreHTTPProbe(c clientset.Interface, ss *appsv1.StatefulSet) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /tmp%v /usr/local/apache2/htdocs/ || true", path)
	return e2estatefulset.ExecInStatefulPods(c, ss, cmd)
}

// restorePodHTTPProbe restores the readiness probe for Nginx StatefulSet containers in pod.
func restorePodHTTPProbe(ss *appsv1.StatefulSet, pod *v1.Pod) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /tmp%v /usr/local/apache2/htdocs/ || true", path)
	stdout, err := framework.RunHostCmdWithRetries(pod.Namespace, pod.Name, cmd, statefulSetPoll, statefulPodTimeout)
	framework.Logf("stdout of %v on %v: %v", cmd, pod.Name, stdout)
	return err
}

// deleteStatefulPodAtIndex deletes the Pod with ordinal index in ss.
func deleteStatefulPodAtIndex(c clientset.Interface, index int, ss *appsv1.StatefulSet) {
	name := getStatefulSetPodNameAtIndex(index, ss)
	noGrace := int64(0)
	if err := c.CoreV1().Pods(ss.Namespace).Delete(context.TODO(), name, metav1.DeleteOptions{GracePeriodSeconds: &noGrace}); err != nil {
		framework.Failf("Failed to delete stateful pod %v for StatefulSet %v/%v: %v", name, ss.Namespace, ss.Name, err)
	}
}

// getStatefulSetPodNameAtIndex gets formatted pod name given index.
func getStatefulSetPodNameAtIndex(index int, ss *appsv1.StatefulSet) string {
	// TODO: we won't use "-index" as the name strategy forever,
	// pull the name out from an identity mapper.
	return fmt.Sprintf("%v-%v", ss.Name, index)
}

type updateStatefulSetFunc func(*appsv1.StatefulSet)

// updateStatefulSetWithRetries updates statfulset template with retries.
func updateStatefulSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateStatefulSetFunc) (statefulSet *appsv1.StatefulSet, err error) {
	statefulSets := c.AppsV1().StatefulSets(namespace)
	var updateErr error
	pollErr := wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		if statefulSet, err = statefulSets.Get(context.TODO(), name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(statefulSet)
		if statefulSet, err = statefulSets.Update(context.TODO(), statefulSet, metav1.UpdateOptions{}); err == nil {
			framework.Logf("Updating stateful set %s", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("couldn't apply the provided updated to stateful set %q: %v", name, updateErr)
	}
	return statefulSet, pollErr
}

// getStatefulSet gets the StatefulSet named name in namespace.
func getStatefulSet(c clientset.Interface, namespace, name string) *appsv1.StatefulSet {
	ss, err := c.AppsV1().StatefulSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		framework.Failf("Failed to get StatefulSet %s/%s: %v", namespace, name, err)
	}
	return ss
}

// verifyStatefulSetPVCsExist confirms that exactly the PVCs for ss with the specified ids exist. This polls until the situation occurs, an error happens, or until timeout (in the latter case an error is also returned). Beware that this cannot tell if a PVC will be deleted at some point in the future, so if used to confirm that no PVCs are deleted, the caller should wait for some event giving the PVCs a reasonable chance to be deleted, before calling this function.
func verifyStatefulSetPVCsExist(c clientset.Interface, ss *appsv1.StatefulSet, claimIds []int) error {
	idSet := map[int]struct{}{}
	for _, id := range claimIds {
		idSet[id] = struct{}{}
	}
	return wait.PollImmediate(e2estatefulset.StatefulSetPoll, e2estatefulset.StatefulSetTimeout, func() (bool, error) {
		pvcList, err := c.CoreV1().PersistentVolumeClaims(ss.Namespace).List(context.TODO(), metav1.ListOptions{LabelSelector: klabels.Everything().String()})
		if err != nil {
			framework.Logf("WARNING: Failed to list pvcs for verification, retrying: %v", err)
			return false, nil
		}
		for _, claim := range ss.Spec.VolumeClaimTemplates {
			pvcNameRE := regexp.MustCompile(fmt.Sprintf("^%s-%s-([0-9]+)$", claim.Name, ss.Name))
			seenPVCs := map[int]struct{}{}
			for _, pvc := range pvcList.Items {
				matches := pvcNameRE.FindStringSubmatch(pvc.Name)
				if len(matches) != 2 {
					continue
				}
				ordinal, err := strconv.ParseInt(matches[1], 10, 32)
				if err != nil {
					framework.Logf("ERROR: bad pvc name %s (%v)", pvc.Name, err)
					return false, err
				}
				if _, found := idSet[int(ordinal)]; !found {
					return false, nil // Retry until the PVCs are consistent.
				} else {
					seenPVCs[int(ordinal)] = struct{}{}
				}
			}
			if len(seenPVCs) != len(idSet) {
				framework.Logf("Found %d of %d PVCs", len(seenPVCs), len(idSet))
				return false, nil // Retry until the PVCs are consistent.
			}
		}
		return true, nil
	})
}

// verifyStatefulSetPVCsExistWithOwnerRefs works as verifyStatefulSetPVCsExist, but also waits for the ownerRefs to match.
func verifyStatefulSetPVCsExistWithOwnerRefs(c clientset.Interface, ss *appsv1.StatefulSet, claimIndicies []int, wantSetRef, wantPodRef bool) error {
	indexSet := map[int]struct{}{}
	for _, id := range claimIndicies {
		indexSet[id] = struct{}{}
	}
	set := getStatefulSet(c, ss.Namespace, ss.Name)
	setUID := set.GetUID()
	if setUID == "" {
		framework.Failf("Statefulset %s mising UID", ss.Name)
	}
	return wait.PollImmediate(e2estatefulset.StatefulSetPoll, e2estatefulset.StatefulSetTimeout, func() (bool, error) {
		pvcList, err := c.CoreV1().PersistentVolumeClaims(ss.Namespace).List(context.TODO(), metav1.ListOptions{LabelSelector: klabels.Everything().String()})
		if err != nil {
			framework.Logf("WARNING: Failed to list pvcs for verification, retrying: %v", err)
			return false, nil
		}
		for _, claim := range ss.Spec.VolumeClaimTemplates {
			pvcNameRE := regexp.MustCompile(fmt.Sprintf("^%s-%s-([0-9]+)$", claim.Name, ss.Name))
			seenPVCs := map[int]struct{}{}
			for _, pvc := range pvcList.Items {
				matches := pvcNameRE.FindStringSubmatch(pvc.Name)
				if len(matches) != 2 {
					continue
				}
				ordinal, err := strconv.ParseInt(matches[1], 10, 32)
				if err != nil {
					framework.Logf("ERROR: bad pvc name %s (%v)", pvc.Name, err)
					return false, err
				}
				if _, found := indexSet[int(ordinal)]; !found {
					framework.Logf("Unexpected, retrying")
					return false, nil // Retry until the PVCs are consistent.
				}
				var foundSetRef, foundPodRef bool
				for _, ref := range pvc.GetOwnerReferences() {
					if ref.Kind == "StatefulSet" && ref.UID == setUID {
						foundSetRef = true
					}
					if ref.Kind == "Pod" {
						podName := fmt.Sprintf("%s-%d", ss.Name, ordinal)
						pod, err := c.CoreV1().Pods(ss.Namespace).Get(context.TODO(), podName, metav1.GetOptions{})
						if err != nil {
							framework.Logf("Pod %s not found, retrying (%v)", podName, err)
							return false, nil
						}
						podUID := pod.GetUID()
						if podUID == "" {
							framework.Failf("Pod %s is missing UID", pod.Name)
						}
						if ref.UID == podUID {
							foundPodRef = true
						}
					}
				}
				if foundSetRef == wantSetRef && foundPodRef == wantPodRef {
					seenPVCs[int(ordinal)] = struct{}{}
				}
			}
			if len(seenPVCs) != len(indexSet) {
				framework.Logf("Only %d PVCs, retrying", len(seenPVCs))
				return false, nil // Retry until the PVCs are consistent.
			}
		}
		return true, nil
	})
}
