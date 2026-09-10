/*
Copyright 2020 The Kubernetes Authors.

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
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	typeConfigMap = "ConfigMap"
	typeSecret    = "Secret"
)

var _ = SIGDescribe("Pod conditions managed by Kubelet", func() {
	f := framework.NewDefaultFramework("pod-conditions")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	f.Context("including PodReadyToStartContainers condition", f.WithSerial(), feature.PodReadyToStartContainersCondition, func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
		})
		ginkgo.It("a pod without init containers should report all conditions set in expected order after the pod is up", runPodReadyConditionsTest(f, false, true))
		ginkgo.It("a pod with init containers should report all conditions set in expected order after the pod is up", runPodReadyConditionsTest(f, true, true))
		ginkgo.It("a pod failing to mount volumes (ConfigMap) and without init containers should report scheduled and initialized conditions set", runPodFailingConditionsTest(f, false, true, typeConfigMap))
		ginkgo.It("a pod failing to mount volumes (ConfigMap) and with init containers should report just the scheduled condition set", runPodFailingConditionsTest(f, true, true, typeConfigMap))
		ginkgo.It("a pod failing to mount volumes (Secret) and without init containers should report scheduled and initialized conditions set", runPodFailingConditionsTest(f, false, true, typeSecret))
		ginkgo.It("a pod failing to mount volumes (Secret) and with init containers should report just the scheduled condition set", runPodFailingConditionsTest(f, true, true, typeSecret))
		addAfterEachForCleaningUpPods(f)
	})

	f.Context("without PodReadyToStartContainersCondition condition", f.WithNodeConformance(), func() {
		ginkgo.It("a pod without init containers should report all conditions set in expected order after the pod is up", runPodReadyConditionsTest(f, false, false))
		ginkgo.It("a pod with init containers should report all conditions set in expected order after the pod is up", runPodReadyConditionsTest(f, true, false))
		ginkgo.It("a pod failing to mount volumes (ConfigMap) and without init containers should report scheduled and initialized conditions set", runPodFailingConditionsTest(f, false, false, typeConfigMap))
		ginkgo.It("a pod failing to mount volumes (ConfigMap) and with init containers should report just the scheduled condition set", runPodFailingConditionsTest(f, true, false, typeConfigMap))
		ginkgo.It("a pod failing to mount volumes (Secret) and without init containers should report scheduled and initialized conditions set", runPodFailingConditionsTest(f, false, false, typeSecret))
		ginkgo.It("a pod failing to mount volumes (Secret) and with init containers should report just the scheduled condition set", runPodFailingConditionsTest(f, true, false, typeSecret))
		addAfterEachForCleaningUpPods(f)
	})

	f.Context("including InsecureUserID and InsecureGroupID conditions", f.WithSerial(), framework.WithFeatureGate(features.InsecurePodWarnings), func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
			initialConfig.FeatureGates[string(features.InsecurePodWarnings)] = true
		})

		ginkgo.It("a pod without runAsUser/runAsGroup should get both conditions set to true and a combined event", func(ctx context.Context) {
			p := e2epod.NewPodClient(f).Create(ctx, insecureIDTestPod("implicitly-insecure", nil, nil))

			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureUserID, true)
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureGroupID, true)

			eventSelector := fields.Set{
				"involvedObject.kind": "Pod",
				"involvedObject.name": p.Name,
				"reason":              "ImplicitlyInsecureUserAndGroupID",
			}.AsSelector().String()
			framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(ctx, f.ClientSet, f.Namespace.Name, eventSelector, "", framework.PodEventTimeout))
		})

		ginkgo.It("a pod with only runAsGroup explicitly set to 0 should get only the InsecureUserID condition set to true", func(ctx context.Context) {
			zero := int64(0)
			p := e2epod.NewPodClient(f).Create(ctx, insecureIDTestPod("implicitly-insecure-uid-only", nil, &zero))

			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureUserID, true)
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureGroupID, false)

			eventSelector := fields.Set{
				"involvedObject.kind": "Pod",
				"involvedObject.name": p.Name,
				"reason":              "ImplicitlyInsecureUserID",
			}.AsSelector().String()
			framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(ctx, f.ClientSet, f.Namespace.Name, eventSelector, "", framework.PodEventTimeout))
		})

		ginkgo.It("a pod with only runAsUser explicitly set to 0 should get only the InsecureGroupID condition set to true", func(ctx context.Context) {
			zero := int64(0)
			p := e2epod.NewPodClient(f).Create(ctx, insecureIDTestPod("implicitly-insecure-gid-only", &zero, nil))

			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureUserID, false)
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureGroupID, true)

			eventSelector := fields.Set{
				"involvedObject.kind": "Pod",
				"involvedObject.name": p.Name,
				"reason":              "ImplicitlyInsecureGroupID",
			}.AsSelector().String()
			framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(ctx, f.ClientSet, f.Namespace.Name, eventSelector, "", framework.PodEventTimeout))
		})

		ginkgo.It("a pod with runAsUser/runAsGroup explicitly set to 0 should get both conditions set to false", func(ctx context.Context) {
			zero := int64(0)
			p := e2epod.NewPodClient(f).Create(ctx, insecureIDTestPod("explicitly-insecure", &zero, &zero))

			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureUserID, false)
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureGroupID, false)
		})

		ginkgo.It("a pod running as non-root should get both conditions set to false", func(ctx context.Context) {
			nonRoot := int64(1000)
			p := e2epod.NewPodClient(f).Create(ctx, insecureIDTestPod("secure-pod", &nonRoot, &nonRoot))

			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureUserID, false)
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureGroupID, false)
		})

		ginkgo.It("a pod with hostUsers false should get both conditions set to false even when running as root", func(ctx context.Context) {
			if !supportsUserNS(ctx, f) {
				e2eskipper.Skipf("runtime does not support user namespaces")
			}

			hostUsersFalse := false
			pod := insecureIDTestPod("root-in-userns", nil, nil)
			pod.Spec.HostUsers = &hostUsersFalse
			p := e2epod.NewPodClient(f).Create(ctx, pod)

			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureUserID, false)
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureGroupID, false)
		})

		ginkgo.It("a pod with an implicitly-insecure init container should get both conditions set to true even if the main container is secure", func(ctx context.Context) {
			nonRoot := int64(1000)
			pod := insecureIDTestPod("implicitly-insecure-init-container", &nonRoot, &nonRoot)
			pod.Spec.InitContainers = []v1.Container{
				{
					Name:    "init",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "true"},
				},
			}
			p := e2epod.NewPodClient(f).Create(ctx, pod)

			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureUserID, true)
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureGroupID, true)
		})

		ginkgo.It("a pod with an implicitly-insecure ephemeral container should get both conditions set to true even if the main container is secure", func(ctx context.Context) {
			nonRoot := int64(1000)
			pod := insecureIDTestPod("implicitly-insecure-ephemeral-container", &nonRoot, &nonRoot)
			p := e2epod.NewPodClient(f).Create(ctx, pod)
			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureUserID, false)
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureGroupID, false)

			ec := &v1.EphemeralContainer{
				EphemeralContainerCommon: v1.EphemeralContainerCommon{
					Name:    "debug",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sleep", "3600"},
				},
			}
			framework.ExpectNoError(e2epod.NewPodClient(f).AddEphemeralContainerSync(ctx, p, ec, framework.PodStartTimeout))
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureUserID, true)
			waitForPodConditionWithStatus(ctx, f, p.Name, v1.InsecureGroupID, true)
		})

		ginkgo.It("should report implicitly and explicitly insecure pod counts in kubelet metrics", func(ctx context.Context) {
			zero := int64(0)
			implicit := e2epod.NewPodClient(f).Create(ctx, insecureIDTestPod("metrics-implicitly-insecure", nil, nil))
			explicit := e2epod.NewPodClient(f).Create(ctx, insecureIDTestPod("metrics-explicitly-insecure", &zero, &zero))

			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, implicit.Name, f.Namespace.Name, framework.PodStartTimeout))
			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, explicit.Name, f.Namespace.Name, framework.PodStartTimeout))

			gomega.Eventually(ctx, func(ctx context.Context) error {
				m, err := getKubeletMetrics(ctx)
				if err != nil {
					return err
				}
				for _, tc := range []struct{ declaration, idType string }{
					{"implicit", "uid"},
					{"implicit", "gid"},
					{"explicit", "uid"},
					{"explicit", "gid"},
				} {
					value, err := getCounterMetricValue(m, "kubelet_insecure_pods", map[string]string{"declaration": tc.declaration, "id_type": tc.idType})
					if err != nil {
						return err
					}
					if value < 1 {
						return fmt.Errorf("expected kubelet_insecure_pods{declaration=%q,id_type=%q} to be >= 1, got %v", tc.declaration, tc.idType, value)
					}
				}
				return nil
			}, framework.PodStartTimeout).Should(gomega.Succeed())
		})
		addAfterEachForCleaningUpPods(f)
	})
})

// insecureIDTestPod returns a pod spec running busybox with the given runAsUser/runAsGroup
// (nil leaves the field unset).
func insecureIDTestPod(name string, runAsUser, runAsGroup *int64) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "c",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sleep", "3600"},
					SecurityContext: &v1.SecurityContext{
						RunAsUser:  runAsUser,
						RunAsGroup: runAsGroup,
					},
				},
			},
		},
	}
}

func runPodFailingConditionsTest(f *framework.Framework, hasInitContainers, checkPodReadyToStart bool, volumeSource string) func(ctx context.Context) {
	return func(ctx context.Context) {
		ginkgo.By("creating a pod whose sandbox creation is blocked due to a missing volume")

		p := webserverPodSpec("pod-"+string(uuid.NewUUID()), "web1", "init1", hasInitContainers)

		switch volumeSource {
		case typeConfigMap:
			p.Spec.Volumes = []v1.Volume{
				{
					Name: "cm",
					VolumeSource: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{Name: "cm-that-unblock-pod-condition"},
						},
					},
				},
			}
			p.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{
				{
					Name:      "cm",
					MountPath: "/config",
				},
			}
		case typeSecret:
			p.Spec.Volumes = []v1.Volume{
				{
					Name: "secret",
					VolumeSource: v1.VolumeSource{
						Secret: &v1.SecretVolumeSource{
							SecretName: "secret-that-unblock-pod-condition",
						},
					},
				},
			}
			p.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{
				{
					Name:      "secret",
					MountPath: "/secret",
				},
			}
		default:
			framework.Failf("unsupported volumeSource: %s", volumeSource)
		}

		p = e2epod.NewPodClient(f).Create(ctx, p)

		ginkgo.By("waiting until kubelet has started trying to set up the pod and started to fail")

		eventSelector := fields.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.name":      p.Name,
			"involvedObject.namespace": f.Namespace.Name,
			"reason":                   events.FailedMountVolume,
		}.AsSelector().String()
		framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(ctx, f.ClientSet, f.Namespace.Name, eventSelector, "MountVolume.SetUp failed for volume", framework.PodEventTimeout))

		p, err := e2epod.NewPodClient(f).Get(ctx, p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("checking pod condition for a pod whose sandbox creation is blocked")

		scheduledTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodScheduled, true)
		framework.ExpectNoError(err)

		// Verify PodReadyToStartContainers is not set (since sandboxcreation is blocked)
		if checkPodReadyToStart {
			_, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodReadyToStartContainers, false)
			framework.ExpectNoError(err)
		}

		if hasInitContainers {
			// Verify PodInitialized is not set if init containers are present (since sandboxcreation is blocked)
			_, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodInitialized, false)
			framework.ExpectNoError(err)
		} else {
			// Verify PodInitialized is set if init containers are not present (since without init containers, it gets set very early)
			initializedTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodInitialized, true)
			framework.ExpectNoError(err)
			gomega.Expect(initializedTime.Before(scheduledTime)).NotTo(gomega.BeTrueBecause("pod without init containers is initialized at: %v which is before pod scheduled at: %v", initializedTime, scheduledTime))
		}

		// Verify ContainersReady is not set (since sandboxcreation is blocked)
		_, err = getTransitionTimeForPodConditionWithStatus(p, v1.ContainersReady, false)
		framework.ExpectNoError(err)
		// Verify PodReady is not set (since sandboxcreation is blocked)
		_, err = getTransitionTimeForPodConditionWithStatus(p, v1.PodReady, false)
		framework.ExpectNoError(err)

		// this testcase is creating the missing volume that unblock the pod above,
		// and check PodReadyToStartContainer is setting correctly.
		ginkgo.By(fmt.Sprintf("creating the missing volume source (%s) to unblock the pod", volumeSource))

		switch volumeSource {
		case typeConfigMap:
			configmap := v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "cm-that-unblock-pod-condition",
				},
				Data: map[string]string{
					"key": "value",
				},
				BinaryData: map[string][]byte{
					"binaryKey": []byte("value"),
				},
			}
			_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, &configmap, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func() {
				err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Delete(ctx, "cm-that-unblock-pod-condition", metav1.DeleteOptions{})
				framework.ExpectNoError(err, "unable to delete configmap")
			}()
		case typeSecret:
			secret := v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "secret-that-unblock-pod-condition",
				},
				Data: map[string][]byte{
					"key": []byte("value"),
				},
			}
			_, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, &secret, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			defer func() {
				err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Delete(ctx, "secret-that-unblock-pod-condition", metav1.DeleteOptions{})
				framework.ExpectNoError(err, "unable to delete secret")
			}()
		}

		ginkgo.By("waiting for the pod to become ready after the volume source is created")
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, p.Namespace, framework.PodStartTimeout))

		p, err = e2epod.NewPodClient(f).Get(ctx, p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		_, err = getTransitionTimeForPodConditionWithStatus(p, v1.PodScheduled, true)
		framework.ExpectNoError(err)

		_, err = getTransitionTimeForPodConditionWithStatus(p, v1.PodInitialized, true)
		framework.ExpectNoError(err)

		// Verify PodReadyToStartContainers is set (since sandboxcreation is unblocked)
		if checkPodReadyToStart {
			_, err = getTransitionTimeForPodConditionWithStatus(p, v1.PodReadyToStartContainers, true)
			framework.ExpectNoError(err)
		}
	}
}

func runPodReadyConditionsTest(f *framework.Framework, hasInitContainers, checkPodReadyToStart bool) func(ctx context.Context) {
	return func(ctx context.Context) {
		ginkgo.By("creating a pod that successfully comes up in a ready/running state")

		p := e2epod.NewPodClient(f).Create(ctx, webserverPodSpec("pod-"+string(uuid.NewUUID()), "web1", "init1", hasInitContainers))
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))

		p, err := e2epod.NewPodClient(f).Get(ctx, p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		isReady, err := testutils.PodRunningReady(p)
		framework.ExpectNoError(err)
		if !isReady {
			framework.Failf("pod %q should be ready", p.Name)
		}

		ginkgo.By("checking order of pod condition transitions for a pod with no container/sandbox restarts")

		scheduledTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodScheduled, true)
		framework.ExpectNoError(err)
		initializedTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodInitialized, true)
		framework.ExpectNoError(err)

		condBeforeContainersReadyTransitionTime := initializedTime
		errSubstrIfContainersReadyTooEarly := "is initialized"
		if checkPodReadyToStart {
			readyToStartContainersTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodReadyToStartContainers, true)
			framework.ExpectNoError(err)

			if hasInitContainers {
				// With init containers, verify the sequence of conditions is: Scheduled => PodReadyToStartContainers => Initialized
				gomega.Expect(readyToStartContainersTime.Before(scheduledTime)).ToNot(gomega.BeTrueBecause("pod with init containers is initialized at: %v which is before pod has ready to start at: %v", initializedTime, readyToStartContainersTime))
				gomega.Expect(initializedTime.Before(readyToStartContainersTime)).ToNot(gomega.BeTrueBecause("pod with init containers is initialized at: %v which is before pod has ready to start at: %v", initializedTime, readyToStartContainersTime))
			} else {
				// Without init containers, verify the sequence of conditions is: Scheduled => Initialized => PodReadyToStartContainers
				condBeforeContainersReadyTransitionTime = readyToStartContainersTime
				errSubstrIfContainersReadyTooEarly = "ready to start"
				gomega.Expect(initializedTime.Before(scheduledTime)).NotTo(gomega.BeTrueBecause("pod without init containers initialized at: %v which is before pod scheduled at: %v", initializedTime, scheduledTime))
				gomega.Expect(readyToStartContainersTime.Before(initializedTime)).NotTo(gomega.BeTrueBecause("pod without init containers has ready to start at: %v which is before pod is initialized at: %v", readyToStartContainersTime, initializedTime))
			}
		} else {
			// In the absence of PodHasReadyToStartContainers feature disabled, verify the sequence is: Scheduled => Initialized
			gomega.Expect(initializedTime.Before(scheduledTime)).NotTo(gomega.BeTrueBecause("pod initialized at: %v which is before pod scheduled at: %v", initializedTime, scheduledTime))
		}
		// Verify the next condition to get set is ContainersReady
		containersReadyTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.ContainersReady, true)
		framework.ExpectNoError(err)
		gomega.Expect(containersReadyTime.Before(condBeforeContainersReadyTransitionTime)).NotTo(gomega.BeTrueBecause("containers ready at: %v which is before pod %s: %v", containersReadyTime, errSubstrIfContainersReadyTooEarly, initializedTime))

		// Verify ContainersReady => PodReady
		podReadyTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodReady, true)
		framework.ExpectNoError(err)
		gomega.Expect(podReadyTime.Before(containersReadyTime)).NotTo(gomega.BeTrueBecause("pod ready at: %v which is before pod containers ready at: %v", podReadyTime, containersReadyTime))
	}
}

func getTransitionTimeForPodConditionWithStatus(pod *v1.Pod, condType v1.PodConditionType, expectedStatus bool) (time.Time, error) {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == condType {
			if strings.EqualFold(string(cond.Status), strconv.FormatBool(expectedStatus)) {
				return cond.LastTransitionTime.Time, nil
			}
			return time.Time{}, fmt.Errorf("condition: %s found for pod but status: %s did not match expected status: %s", condType, cond.Status, strconv.FormatBool(expectedStatus))
		}
	}
	return time.Time{}, fmt.Errorf("condition: %s not found for pod", condType)
}

// waitForPodConditionWithStatus polls until the pod's condition reaches the expected status.
func waitForPodConditionWithStatus(ctx context.Context, f *framework.Framework, podName string, condType v1.PodConditionType, expectedStatus bool) {
	gomega.Eventually(ctx, func(ctx context.Context) error {
		p, err := e2epod.NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		_, err = getTransitionTimeForPodConditionWithStatus(p, condType, expectedStatus)
		return err
	}, framework.PodStartTimeout).Should(gomega.Succeed())
}

func webserverPodSpec(podName, containerName, initContainerName string, addInitContainer bool) *v1.Pod {
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  containerName,
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"test-webserver"},
				},
			},
		},
	}
	if addInitContainer {
		p.Spec.InitContainers = []v1.Container{
			{
				Name:    initContainerName,
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"sh", "-c", "sleep 5s"},
			},
		}
	}
	return p
}
