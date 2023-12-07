/*
Copyright 2022 The Kubernetes Authors.

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

package csi_mock

import (
	"context"
	"fmt"
	"sort"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock selinux on mount", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-selinux")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	f.Context("SELinuxMount [LinuxOnly]", feature.SELinux, func() {
		// Make sure all options are set so system specific defaults are not used.
		seLinuxOpts1 := v1.SELinuxOptions{
			User:  "system_u",
			Role:  "system_r",
			Type:  "container_t",
			Level: "s0:c0,c1",
		}
		seLinuxMountOption1 := "context=\"system_u:object_r:container_file_t:s0:c0,c1\""
		seLinuxOpts2 := v1.SELinuxOptions{
			User:  "system_u",
			Role:  "system_r",
			Type:  "container_t",
			Level: "s0:c98,c99",
		}
		seLinuxMountOption2 := "context=\"system_u:object_r:container_file_t:s0:c98,c99\""

		tests := []struct {
			name                       string
			csiDriverSELinuxEnabled    bool
			firstPodSELinuxOpts        *v1.SELinuxOptions
			startSecondPod             bool
			secondPodSELinuxOpts       *v1.SELinuxOptions
			mountOptions               []string
			volumeMode                 v1.PersistentVolumeAccessMode
			expectedFirstMountOptions  []string
			expectedSecondMountOptions []string
			expectedUnstage            bool
		}{
			// Start just a single pod and check its volume is mounted correctly
			{
				name:                      "should pass SELinux mount option for RWOP volume and Pod with SELinux context set",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                v1.ReadWriteOncePod,
				expectedFirstMountOptions: []string{seLinuxMountOption1},
			},
			{
				name:                      "should add SELinux mount option to existing mount options",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				mountOptions:              []string{"noexec", "noatime"},
				volumeMode:                v1.ReadWriteOncePod,
				expectedFirstMountOptions: []string{"noexec", "noatime", seLinuxMountOption1},
			},
			{
				name:                      "should not pass SELinux mount option for RWO volume",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                v1.ReadWriteOnce,
				expectedFirstMountOptions: nil,
			},
			{
				name:                      "should not pass SELinux mount option for Pod without SELinux context",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       nil,
				volumeMode:                v1.ReadWriteOncePod,
				expectedFirstMountOptions: nil,
			},
			{
				name:                      "should not pass SELinux mount option for CSI driver that does not support SELinux mount",
				csiDriverSELinuxEnabled:   false,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                v1.ReadWriteOncePod,
				expectedFirstMountOptions: nil,
			},
			// Start two pods in a sequence and check their volume is / is not unmounted in between
			{
				name:                       "should not unstage volume when starting a second pod with the same SELinux context",
				csiDriverSELinuxEnabled:    true,
				firstPodSELinuxOpts:        &seLinuxOpts1,
				startSecondPod:             true,
				secondPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                 v1.ReadWriteOncePod,
				expectedFirstMountOptions:  []string{seLinuxMountOption1},
				expectedSecondMountOptions: []string{seLinuxMountOption1},
				expectedUnstage:            false,
			},
			{
				name:                       "should unstage volume when starting a second pod with different SELinux context",
				csiDriverSELinuxEnabled:    true,
				firstPodSELinuxOpts:        &seLinuxOpts1,
				startSecondPod:             true,
				secondPodSELinuxOpts:       &seLinuxOpts2,
				volumeMode:                 v1.ReadWriteOncePod,
				expectedFirstMountOptions:  []string{seLinuxMountOption1},
				expectedSecondMountOptions: []string{seLinuxMountOption2},
				expectedUnstage:            true,
			},
		}
		for _, t := range tests {
			t := t
			ginkgo.It(t.name, func(ctx context.Context) {
				if framework.NodeOSDistroIs("windows") {
					e2eskipper.Skipf("SELinuxMount is only applied on linux nodes -- skipping")
				}
				var nodeStageMountOpts, nodePublishMountOpts []string
				var unstageCalls, stageCalls, unpublishCalls, publishCalls atomic.Int32
				m.init(ctx, testParameters{
					disableAttach:      true,
					registerDriver:     true,
					enableSELinuxMount: &t.csiDriverSELinuxEnabled,
					hooks:              createSELinuxMountPreHook(&nodeStageMountOpts, &nodePublishMountOpts, &stageCalls, &unstageCalls, &publishCalls, &unpublishCalls),
				})
				ginkgo.DeferCleanup(m.cleanup)

				// Act
				ginkgo.By("Starting the initial pod")
				accessModes := []v1.PersistentVolumeAccessMode{t.volumeMode}
				_, claim, pod := m.createPodWithSELinux(ctx, accessModes, t.mountOptions, t.firstPodSELinuxOpts)
				err := e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "starting the initial pod")

				// Assert
				ginkgo.By("Checking the initial pod mount options")
				gomega.Expect(nodeStageMountOpts).To(gomega.Equal(t.expectedFirstMountOptions), "NodeStage MountFlags for the initial pod")
				gomega.Expect(nodePublishMountOpts).To(gomega.Equal(t.expectedFirstMountOptions), "NodePublish MountFlags for the initial pod")

				ginkgo.By("Checking the CSI driver calls for the initial pod")
				gomega.Expect(unstageCalls.Load()).To(gomega.BeNumerically("==", 0), "NodeUnstage call count for the initial pod")
				gomega.Expect(unpublishCalls.Load()).To(gomega.BeNumerically("==", 0), "NodeUnpublish call count for the initial pod")
				gomega.Expect(stageCalls.Load()).To(gomega.BeNumerically(">", 0), "NodeStage for the initial pod")
				gomega.Expect(publishCalls.Load()).To(gomega.BeNumerically(">", 0), "NodePublish for the initial pod")

				if !t.startSecondPod {
					return
				}

				// Arrange 2nd part of the test
				ginkgo.By("Starting the second pod to check if a volume used by the initial pod is / is not unmounted based on SELinux context")

				// Skip scheduler, it would block scheduling the second pod with ReadWriteOncePod PV.
				pod, err = m.cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "getting the initial pod")
				nodeSelection := e2epod.NodeSelection{Name: pod.Spec.NodeName}
				pod2, err := startPausePodWithSELinuxOptions(f.ClientSet, claim, nodeSelection, f.Namespace.Name, t.secondPodSELinuxOpts)
				framework.ExpectNoError(err, "creating second pod with SELinux context %s", t.secondPodSELinuxOpts)
				m.pods = append(m.pods, pod2)

				// Delete the initial pod only after kubelet processes the second pod and adds its volumes to
				// DesiredStateOfWorld.
				// In this state, any volume UnPublish / UnStage must be done because of SELinux contexts and not
				// because of random races because volumes of the second pod are not in DesiredStateOfWorld yet.
				ginkgo.By("Waiting for the second pod to fail to start because of ReadWriteOncePod.")
				eventSelector := fields.Set{
					"involvedObject.kind":      "Pod",
					"involvedObject.name":      pod2.Name,
					"involvedObject.namespace": pod2.Namespace,
					"reason":                   events.FailedMountVolume,
				}.AsSelector().String()
				var msg string
				if t.expectedUnstage {
					// This message is emitted before kubelet checks for ReadWriteOncePod
					msg = "conflicting SELinux labels of volume"
				} else {
					msg = "volume uses the ReadWriteOncePod access mode and is already in use by another pod"
				}
				err = e2eevents.WaitTimeoutForEvent(ctx, m.cs, pod2.Namespace, eventSelector, msg, f.Timeouts.PodStart)
				framework.ExpectNoError(err, "waiting for event %q in the second test pod", msg)

				// count fresh CSI driver calls between the first and the second pod
				nodeStageMountOpts = nil
				nodePublishMountOpts = nil
				unstageCalls.Store(0)
				unpublishCalls.Store(0)
				stageCalls.Store(0)
				publishCalls.Store(0)

				// Act 2nd part of the test
				ginkgo.By("Deleting the initial pod")
				err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
				framework.ExpectNoError(err, "deleting the initial pod")

				// Assert 2nd part of the test
				ginkgo.By("Waiting for the second pod to start")
				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod2.Name, pod2.Namespace)
				framework.ExpectNoError(err, "starting the second pod")

				ginkgo.By("Checking CSI driver calls for the second pod")
				if t.expectedUnstage {
					// Volume should be fully unstaged between the first and the second pod
					gomega.Expect(unstageCalls.Load()).To(gomega.BeNumerically(">", 0), "NodeUnstage calls after the first pod is deleted")
					gomega.Expect(stageCalls.Load()).To(gomega.BeNumerically(">", 0), "NodeStage calls for the second pod")
					// The second pod got the right mount option
					gomega.Expect(nodeStageMountOpts).To(gomega.Equal(t.expectedSecondMountOptions), "NodeStage MountFlags for the second pod")
				} else {
					// Volume should not be fully unstaged between the first and the second pod
					gomega.Expect(unstageCalls.Load()).To(gomega.BeNumerically("==", 0), "NodeUnstage calls after the first pod is deleted")
					gomega.Expect(stageCalls.Load()).To(gomega.BeNumerically("==", 0), "NodeStage calls for the second pod")
				}
				// In both cases, Unublish and Publish is called, with the right mount opts
				gomega.Expect(unpublishCalls.Load()).To(gomega.BeNumerically(">", 0), "NodeUnpublish calls after the first pod is deleted")
				gomega.Expect(publishCalls.Load()).To(gomega.BeNumerically(">", 0), "NodePublish calls for the second pod")
				gomega.Expect(nodePublishMountOpts).To(gomega.Equal(t.expectedSecondMountOptions), "NodePublish MountFlags for the second pod")
			})
		}
	})
})

var _ = utils.SIGDescribe("CSI Mock selinux on mount metrics", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-selinux-metrics")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	// [Serial]: the tests read global kube-controller-manager metrics, so no other test changes them in parallel.
	f.Context("SELinuxMount metrics [LinuxOnly]", feature.SELinux, feature.SELinuxMountReadWriteOncePod, f.WithSerial(), func() {

		// All SELinux metrics. Unless explicitly mentioned in test.expectIncreases, these metrics must not grow during
		// a test.
		allMetrics := sets.NewString(
			"volume_manager_selinux_container_errors_total",
			"volume_manager_selinux_container_warnings_total",
			"volume_manager_selinux_pod_context_mismatch_errors_total",
			"volume_manager_selinux_pod_context_mismatch_warnings_total",
			"volume_manager_selinux_volume_context_mismatch_errors_total",
			"volume_manager_selinux_volume_context_mismatch_warnings_total",
			"volume_manager_selinux_volumes_admitted_total",
		)

		// Make sure all options are set so system specific defaults are not used.
		seLinuxOpts1 := v1.SELinuxOptions{
			User:  "system_u",
			Role:  "system_r",
			Type:  "container_t",
			Level: "s0:c0,c1",
		}
		seLinuxOpts2 := v1.SELinuxOptions{
			User:  "system_u",
			Role:  "system_r",
			Type:  "container_t",
			Level: "s0:c98,c99",
		}

		tests := []struct {
			name                    string
			csiDriverSELinuxEnabled bool
			firstPodSELinuxOpts     *v1.SELinuxOptions
			secondPodSELinuxOpts    *v1.SELinuxOptions
			volumeMode              v1.PersistentVolumeAccessMode
			waitForSecondPodStart   bool
			secondPodFailureEvent   string
			expectIncreases         sets.String
		}{
			{
				name:                    "warning is not bumped on two Pods with the same context on RWO volume",
				csiDriverSELinuxEnabled: true,
				firstPodSELinuxOpts:     &seLinuxOpts1,
				secondPodSELinuxOpts:    &seLinuxOpts1,
				volumeMode:              v1.ReadWriteOnce,
				waitForSecondPodStart:   true,
				expectIncreases:         sets.NewString( /* no metric is increased, admitted_total was already increased when the first pod started */ ),
			},
			{
				name:                    "warning is bumped on two Pods with a different context on RWO volume",
				csiDriverSELinuxEnabled: true,
				firstPodSELinuxOpts:     &seLinuxOpts1,
				secondPodSELinuxOpts:    &seLinuxOpts2,
				volumeMode:              v1.ReadWriteOnce,
				waitForSecondPodStart:   true,
				expectIncreases:         sets.NewString("volume_manager_selinux_volume_context_mismatch_warnings_total"),
			},
			{
				name:                    "error is bumped on two Pods with a different context on RWOP volume",
				csiDriverSELinuxEnabled: true,
				firstPodSELinuxOpts:     &seLinuxOpts1,
				secondPodSELinuxOpts:    &seLinuxOpts2,
				secondPodFailureEvent:   "conflicting SELinux labels of volume",
				volumeMode:              v1.ReadWriteOncePod,
				waitForSecondPodStart:   false,
				expectIncreases:         sets.NewString("volume_manager_selinux_volume_context_mismatch_errors_total"),
			},
		}
		for _, t := range tests {
			t := t
			ginkgo.It(t.name, func(ctx context.Context) {
				if framework.NodeOSDistroIs("windows") {
					e2eskipper.Skipf("SELinuxMount is only applied on linux nodes -- skipping")
				}
				grabber, err := e2emetrics.NewMetricsGrabber(ctx, f.ClientSet, nil, f.ClientConfig(), true, false, false, false, false, false)
				framework.ExpectNoError(err, "creating the metrics grabber")

				var nodeStageMountOpts, nodePublishMountOpts []string
				var unstageCalls, stageCalls, unpublishCalls, publishCalls atomic.Int32
				m.init(ctx, testParameters{
					disableAttach:      true,
					registerDriver:     true,
					enableSELinuxMount: &t.csiDriverSELinuxEnabled,
					hooks:              createSELinuxMountPreHook(&nodeStageMountOpts, &nodePublishMountOpts, &stageCalls, &unstageCalls, &publishCalls, &unpublishCalls),
				})
				ginkgo.DeferCleanup(m.cleanup)

				ginkgo.By("Starting the first pod")
				accessModes := []v1.PersistentVolumeAccessMode{t.volumeMode}
				_, claim, pod := m.createPodWithSELinux(ctx, accessModes, []string{}, t.firstPodSELinuxOpts)
				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "starting the initial pod")

				ginkgo.By("Grabbing initial metrics")
				pod, err = m.cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "getting the initial pod")
				metrics, err := grabMetrics(ctx, grabber, pod.Spec.NodeName, allMetrics)
				framework.ExpectNoError(err, "collecting the initial metrics")
				dumpMetrics(metrics)

				// Act
				ginkgo.By("Starting the second pod")
				// Skip scheduler, it would block scheduling the second pod with ReadWriteOncePod PV.
				nodeSelection := e2epod.NodeSelection{Name: pod.Spec.NodeName}
				pod2, err := startPausePodWithSELinuxOptions(f.ClientSet, claim, nodeSelection, f.Namespace.Name, t.secondPodSELinuxOpts)
				framework.ExpectNoError(err, "creating second pod with SELinux context %s", t.secondPodSELinuxOpts)
				m.pods = append(m.pods, pod2)

				if t.waitForSecondPodStart {
					err := e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod2.Name, pod2.Namespace)
					framework.ExpectNoError(err, "starting the second pod")
				} else {
					ginkgo.By("Waiting for the second pod to fail to start")
					eventSelector := fields.Set{
						"involvedObject.kind":      "Pod",
						"involvedObject.name":      pod2.Name,
						"involvedObject.namespace": pod2.Namespace,
						"reason":                   events.FailedMountVolume,
					}.AsSelector().String()
					err = e2eevents.WaitTimeoutForEvent(ctx, m.cs, pod2.Namespace, eventSelector, t.secondPodFailureEvent, f.Timeouts.PodStart)
					framework.ExpectNoError(err, "waiting for event %q in the second test pod", t.secondPodFailureEvent)
				}

				// Assert: count the metrics
				ginkgo.By("Waiting for expected metric changes")
				err = waitForMetricIncrease(ctx, grabber, pod.Spec.NodeName, allMetrics, t.expectIncreases, metrics, framework.PodStartShortTimeout)
				framework.ExpectNoError(err, "waiting for metrics %s to increase", t.expectIncreases)
			})
		}
	})
})

func grabMetrics(ctx context.Context, grabber *e2emetrics.Grabber, nodeName string, metricNames sets.String) (map[string]float64, error) {
	response, err := grabber.GrabFromKubelet(ctx, nodeName)
	framework.ExpectNoError(err)

	metrics := map[string]float64{}
	for method, samples := range response {
		if metricNames.Has(method) {
			if len(samples) == 0 {
				return nil, fmt.Errorf("metric %s has no samples", method)
			}
			lastSample := samples[len(samples)-1]
			metrics[method] = float64(lastSample.Value)
		}
	}

	// Ensure all metrics were provided
	for name := range metricNames {
		if _, found := metrics[name]; !found {
			return nil, fmt.Errorf("metric %s not found", name)
		}
	}

	return metrics, nil
}

func waitForMetricIncrease(ctx context.Context, grabber *e2emetrics.Grabber, nodeName string, allMetricNames, expectedIncreaseNames sets.String, initialValues map[string]float64, timeout time.Duration) error {
	var noIncreaseMetrics sets.String
	var metrics map[string]float64

	err := wait.Poll(time.Second, timeout, func() (bool, error) {
		var err error
		metrics, err = grabMetrics(ctx, grabber, nodeName, allMetricNames)
		if err != nil {
			return false, err
		}

		noIncreaseMetrics = sets.NewString()
		// Always evaluate all SELinux metrics to check that the other metrics are not unexpectedly increased.
		for name := range allMetricNames {
			if expectedIncreaseNames.Has(name) {
				if metrics[name] <= initialValues[name] {
					noIncreaseMetrics.Insert(name)
				}
			} else {
				if initialValues[name] != metrics[name] {
					return false, fmt.Errorf("metric %s unexpectedly increased to %v", name, metrics[name])
				}
			}
		}
		return noIncreaseMetrics.Len() == 0, nil
	})

	ginkgo.By("Dumping final metrics")
	dumpMetrics(metrics)

	if err == context.DeadlineExceeded {
		return fmt.Errorf("timed out waiting for metrics %v", noIncreaseMetrics.List())
	}
	return err
}

func dumpMetrics(metrics map[string]float64) {
	// Print the metrics sorted by metric name for better readability
	keys := make([]string, 0, len(metrics))
	for key := range metrics {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		framework.Logf("Metric %s: %v", key, metrics[key])
	}
}
