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
	"k8s.io/kubernetes/pkg/kubelet/events"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("Pod conditions managed by Kubelet", func() {
	f := framework.NewDefaultFramework("pod-conditions")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.Context("including PodHasNetwork condition [Serial] [Feature:PodHasNetwork]", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodHasNetworkCondition): true,
			}
		})
		ginkgo.It("a pod without init containers should report all conditions set in expected order after the pod is up", runPodReadyConditionsTest(f, false, true))
		ginkgo.It("a pod with init containers should report all conditions set in expected order after the pod is up", runPodReadyConditionsTest(f, true, true))
		ginkgo.It("a pod failing to mount volumes and without init containers should report scheduled and initialized conditions set", runPodFailingConditionsTest(f, false, true))
		ginkgo.It("a pod failing to mount volumes and with init containers should report just the scheduled condition set", runPodFailingConditionsTest(f, true, true))
	})

	ginkgo.Context("without PodHasNetwork condition", func() {
		ginkgo.It("a pod without init containers should report all conditions set in expected order after the pod is up", runPodReadyConditionsTest(f, false, false))
		ginkgo.It("a pod with init containers should report all conditions set in expected order after the pod is up", runPodReadyConditionsTest(f, true, false))
		ginkgo.It("a pod failing to mount volumes and without init containers should report scheduled and initialized conditions set", runPodFailingConditionsTest(f, false, false))
		ginkgo.It("a pod failing to mount volumes and with init containers should report just the scheduled condition set", runPodFailingConditionsTest(f, true, false))
	})
})

func runPodFailingConditionsTest(f *framework.Framework, hasInitContainers, checkPodHasNetwork bool) func(ctx context.Context) {
	return func(ctx context.Context) {
		ginkgo.By("creating a pod whose sandbox creation is blocked due to a missing volume")

		p := webserverPodSpec("pod-"+string(uuid.NewUUID()), "web1", "init1", hasInitContainers)
		p.Spec.Volumes = []v1.Volume{
			{
				Name: "cm",
				VolumeSource: v1.VolumeSource{
					ConfigMap: &v1.ConfigMapVolumeSource{
						LocalObjectReference: v1.LocalObjectReference{Name: "does-not-exist"},
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

		// Verify PodHasNetwork is not set (since sandboxcreation is blocked)
		if checkPodHasNetwork {
			_, err := getTransitionTimeForPodConditionWithStatus(p, kubetypes.PodHasNetwork, false)
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
			framework.ExpectNotEqual(initializedTime.Before(scheduledTime), true, fmt.Sprintf("pod without init containers is initialized at: %v which is before pod scheduled at: %v", initializedTime, scheduledTime))
		}

		// Verify ContainersReady is not set (since sandboxcreation is blocked)
		_, err = getTransitionTimeForPodConditionWithStatus(p, v1.ContainersReady, false)
		framework.ExpectNoError(err)
		// Verify PodReady is not set (since sandboxcreation is blocked)
		_, err = getTransitionTimeForPodConditionWithStatus(p, v1.PodReady, false)
		framework.ExpectNoError(err)
	}
}

func runPodReadyConditionsTest(f *framework.Framework, hasInitContainers, checkPodHasNetwork bool) func(ctx context.Context) {
	return func(ctx context.Context) {
		ginkgo.By("creating a pod that successfully comes up in a ready/running state")

		p := e2epod.NewPodClient(f).Create(ctx, webserverPodSpec("pod-"+string(uuid.NewUUID()), "web1", "init1", hasInitContainers))
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name, framework.PodStartTimeout))

		p, err := e2epod.NewPodClient(f).Get(ctx, p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		isReady, err := testutils.PodRunningReady(p)
		framework.ExpectNoError(err)
		framework.ExpectEqual(isReady, true, "pod should be ready")

		ginkgo.By("checking order of pod condition transitions for a pod with no container/sandbox restarts")

		scheduledTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodScheduled, true)
		framework.ExpectNoError(err)
		initializedTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodInitialized, true)
		framework.ExpectNoError(err)

		condBeforeContainersReadyTransitionTime := initializedTime
		errSubstrIfContainersReadyTooEarly := "is initialized"
		if checkPodHasNetwork {
			hasNetworkTime, err := getTransitionTimeForPodConditionWithStatus(p, kubetypes.PodHasNetwork, true)
			framework.ExpectNoError(err)

			if hasInitContainers {
				// With init containers, verify the sequence of conditions is: Scheduled => HasNetwork => Initialized
				framework.ExpectNotEqual(hasNetworkTime.Before(scheduledTime), true, fmt.Sprintf("pod with init containers is initialized at: %v which is before pod has network at: %v", initializedTime, hasNetworkTime))
				framework.ExpectNotEqual(initializedTime.Before(hasNetworkTime), true, fmt.Sprintf("pod with init containers is initialized at: %v which is before pod has network at: %v", initializedTime, hasNetworkTime))
			} else {
				// Without init containers, verify the sequence of conditions is: Scheduled => Initialized => HasNetwork
				condBeforeContainersReadyTransitionTime = hasNetworkTime
				errSubstrIfContainersReadyTooEarly = "has network"
				framework.ExpectNotEqual(initializedTime.Before(scheduledTime), true, fmt.Sprintf("pod without init containers initialized at: %v which is before pod scheduled at: %v", initializedTime, scheduledTime))
				framework.ExpectNotEqual(hasNetworkTime.Before(initializedTime), true, fmt.Sprintf("pod without init containers has network at: %v which is before pod is initialized at: %v", hasNetworkTime, initializedTime))
			}
		} else {
			// In the absence of HasNetwork feature disabled, verify the sequence is: Scheduled => Initialized
			framework.ExpectNotEqual(initializedTime.Before(scheduledTime), true, fmt.Sprintf("pod initialized at: %v which is before pod scheduled at: %v", initializedTime, scheduledTime))
		}
		// Verify the next condition to get set is ContainersReady
		containersReadyTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.ContainersReady, true)
		framework.ExpectNoError(err)
		framework.ExpectNotEqual(containersReadyTime.Before(condBeforeContainersReadyTransitionTime), true, fmt.Sprintf("containers ready at: %v which is before pod %s: %v", containersReadyTime, errSubstrIfContainersReadyTooEarly, initializedTime))

		// Verify ContainersReady => PodReady
		podReadyTime, err := getTransitionTimeForPodConditionWithStatus(p, v1.PodReady, true)
		framework.ExpectNoError(err)
		framework.ExpectNotEqual(podReadyTime.Before(containersReadyTime), true, fmt.Sprintf("pod ready at: %v which is before pod containers ready at: %v", podReadyTime, containersReadyTime))
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
