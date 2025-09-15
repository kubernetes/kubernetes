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

package csimock

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
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

// Tests for SELinuxMount feature.
// The tests have explicit [Feature:SELinux] and are skipped unless explicitly requested.
// All tests in this file exptect the SELinux is enabled on all worker nodes.
// Supported node operating systems passed as --node-os-distro are "debian", "ubuntu" and "custom".
// "custom" expects a Fedora Linux derivative (RHEL, CentOS, Rocky, Alma, ...)
// (Patches for more distros are welcome, the author cannot test SELinux on "gci" - is it even supported?)
//
// KEP: https://github.com/kubernetes/enhancements/tree/master/keps/sig-storage/1710-selinux-relabeling
// There are three feature gates: SELinuxMountReadWriteOncePod, SELinuxChangePolicy and SELinuxMount.
// These tags are used in the tests:
//
// [FeatureGate:SELinuxMountReadWriteOncePod]
//   - The test requires SELinuxMountReadWriteOncePod enabled.
//
// [FeatureGate:SELinuxMountReadWriteOncePod] [Feature:SELinuxMountReadWriteOncePodOnly]
//   - The test requires SELinuxMountReadWriteOncePod enabled and SELinuxMount disabled. This checks metrics that are emitted only when SELinuxMount is disabled.
//
// [FeatureGate:SELinuxMountReadWriteOncePod] [Feature:SELinuxMountReadWriteOncePodOnly] [Feature:SELinuxChangePolicy]
//   - The test requires SELinuxMountReadWriteOncePod and SELinuxChangePolicy enabled and SELinuxMount disabled. This checks metrics that are emitted only when SELinuxMount is disabled.
//
// [FeatureGate:SELinuxMountReadWriteOncePod] [Feature:SELinuxChangePolicy] [FeatureGate:SELinuxMount]
//   - The test requires SELinuxMountReadWriteOncePod, Feature:SELinuxChangePolicy and SELinuxMount enabled.
//
// All other feature gate combinations should be invalid.

const (
	controllerSELinuxMetricName = "selinux_warning_controller_selinux_volume_conflict"
)

var (
	defaultSELinuxLabels = map[string]struct{ defaultProcessLabel, defaultFileLabel string }{
		"debian": {"svirt_lxc_net_t", "svirt_lxc_file_t"},
		"ubuntu": {"svirt_lxc_net_t", "svirt_lxc_file_t"},
		// Assume "custom" means Fedora and derivates. `e2e.test --node-os-distro=` does not have "fedora" or "rhel".
		"custom": {"container_t", "container_file_t"},
	}
)

var _ = utils.SIGDescribe("CSI Mock selinux on mount", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-selinux")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)
	recursive := v1.SELinuxChangePolicyRecursive
	mount := v1.SELinuxChangePolicyMountOption
	f.Context("SELinuxMount [LinuxOnly]", feature.SELinux, func() {
		processLabel, fileLabel := getDefaultContainerSELinuxLabels()
		// Make sure all options are set so system specific defaults are not used.
		seLinuxOpts1 := v1.SELinuxOptions{
			User:  "system_u",
			Role:  "system_r",
			Type:  processLabel,
			Level: "s0:c0,c1",
		}
		seLinuxMountOption1 := fmt.Sprintf("context=\"system_u:object_r:%s:s0:c0,c1\"", fileLabel)
		seLinuxOpts2 := v1.SELinuxOptions{
			User:  "system_u",
			Role:  "system_r",
			Type:  processLabel,
			Level: "s0:c98,c99",
		}
		seLinuxMountOption2 := fmt.Sprintf("context=\"system_u:object_r:%s:s0:c98,c99\"", fileLabel)

		tests := []struct {
			name                       string
			csiDriverSELinuxEnabled    bool
			firstPodSELinuxOpts        *v1.SELinuxOptions
			firstPodChangePolicy       *v1.PodSELinuxChangePolicy
			startSecondPod             bool
			secondPodSELinuxOpts       *v1.SELinuxOptions
			secondPodChangePolicy      *v1.PodSELinuxChangePolicy
			mountOptions               []string
			volumeMode                 v1.PersistentVolumeAccessMode
			expectedFirstMountOptions  []string
			expectedSecondMountOptions []string
			expectedUnstage            bool
			testTags                   []interface{}
		}{
			// Start just a single pod and check its volume is mounted correctly
			{
				name:                      "should pass SELinux mount option for RWOP volume and Pod with SELinux context set",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                v1.ReadWriteOncePod,
				expectedFirstMountOptions: []string{seLinuxMountOption1},
				testTags:                  []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod)},
			},
			{
				name:                      "should add SELinux mount option to existing mount options",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				mountOptions:              []string{"noexec", "noatime"},
				volumeMode:                v1.ReadWriteOncePod,
				expectedFirstMountOptions: []string{"noexec", "noatime", seLinuxMountOption1},
				testTags:                  []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod)},
			},
			{
				name:                      "should not pass SELinux mount option for RWO volume with SELinuxMount disabled",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                v1.ReadWriteOnce,
				expectedFirstMountOptions: nil,
				testTags:                  []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod), feature.SELinuxMountReadWriteOncePodOnly},
			},
			{
				name:                      "should not pass SELinux mount option for RWO volume with only SELinuxChangePolicy enabled",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                v1.ReadWriteOnce,
				expectedFirstMountOptions: nil,
				testTags:                  []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod), feature.SELinuxMountReadWriteOncePodOnly, framework.WithFeatureGate(features.SELinuxChangePolicy)},
			},
			{
				name:                      "should pass SELinux mount option for RWO volume with SELinuxMount enabled and nil policy",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                v1.ReadWriteOnce,
				expectedFirstMountOptions: []string{seLinuxMountOption1},
				testTags:                  []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod), framework.WithFeatureGate(features.SELinuxChangePolicy), framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                      "should pass SELinux mount option for RWO volume with SELinuxMount enabled and MountOption policy",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				firstPodChangePolicy:      &mount,
				volumeMode:                v1.ReadWriteOnce,
				expectedFirstMountOptions: []string{seLinuxMountOption1},
				testTags:                  []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod), framework.WithFeatureGate(features.SELinuxChangePolicy), framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                      "should not pass SELinux mount option for RWO volume with SELinuxMount disabled and Recursive policy",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				firstPodChangePolicy:      &recursive,
				volumeMode:                v1.ReadWriteOnce,
				expectedFirstMountOptions: nil,
				testTags:                  []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod), framework.WithFeatureGate(features.SELinuxChangePolicy), framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                      "should not pass SELinux mount option for Pod without SELinux context",
				csiDriverSELinuxEnabled:   true,
				firstPodSELinuxOpts:       nil,
				volumeMode:                v1.ReadWriteOncePod,
				expectedFirstMountOptions: nil,
				testTags:                  []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod)},
			},
			{
				name:                      "should not pass SELinux mount option for CSI driver that does not support SELinux mount",
				csiDriverSELinuxEnabled:   false,
				firstPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                v1.ReadWriteOncePod,
				expectedFirstMountOptions: nil,
				testTags:                  []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod)},
			},
			// Start two pods in a sequence and check their volume is / is not unmounted in between
			{
				name:                       "should not unstage RWOP volume when starting a second pod with the same SELinux context",
				csiDriverSELinuxEnabled:    true,
				firstPodSELinuxOpts:        &seLinuxOpts1,
				startSecondPod:             true,
				secondPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                 v1.ReadWriteOncePod,
				expectedFirstMountOptions:  []string{seLinuxMountOption1},
				expectedSecondMountOptions: []string{seLinuxMountOption1},
				expectedUnstage:            false,
				testTags:                   []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod)},
			},
			{
				name:                       "should unstage RWOP volume when starting a second pod with different SELinux context",
				csiDriverSELinuxEnabled:    true,
				firstPodSELinuxOpts:        &seLinuxOpts1,
				startSecondPod:             true,
				secondPodSELinuxOpts:       &seLinuxOpts2,
				volumeMode:                 v1.ReadWriteOncePod,
				expectedFirstMountOptions:  []string{seLinuxMountOption1},
				expectedSecondMountOptions: []string{seLinuxMountOption2},
				expectedUnstage:            true,
				testTags:                   []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod)},
			},
			{
				name:                       "should not unstage RWO volume when starting a second pod with the same SELinux context",
				csiDriverSELinuxEnabled:    true,
				firstPodSELinuxOpts:        &seLinuxOpts1,
				startSecondPod:             true,
				secondPodSELinuxOpts:       &seLinuxOpts1,
				volumeMode:                 v1.ReadWriteOnce,
				expectedFirstMountOptions:  []string{seLinuxMountOption1},
				expectedSecondMountOptions: []string{seLinuxMountOption1},
				expectedUnstage:            false,
				testTags:                   []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod), framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                       "should unstage RWO volume when starting a second pod with different SELinux context",
				csiDriverSELinuxEnabled:    true,
				firstPodSELinuxOpts:        &seLinuxOpts1,
				startSecondPod:             true,
				secondPodSELinuxOpts:       &seLinuxOpts2,
				volumeMode:                 v1.ReadWriteOnce,
				expectedFirstMountOptions:  []string{seLinuxMountOption1},
				expectedSecondMountOptions: []string{seLinuxMountOption2},
				expectedUnstage:            true,
				testTags:                   []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod), framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                       "should unstage RWO volume when starting a second pod with different policy (MountOption -> Recursive)",
				csiDriverSELinuxEnabled:    true,
				firstPodSELinuxOpts:        &seLinuxOpts1,
				firstPodChangePolicy:       &mount,
				startSecondPod:             true,
				secondPodSELinuxOpts:       &seLinuxOpts2,
				secondPodChangePolicy:      &recursive,
				volumeMode:                 v1.ReadWriteOnce,
				expectedFirstMountOptions:  []string{seLinuxMountOption1},
				expectedSecondMountOptions: nil,
				expectedUnstage:            true,
				testTags:                   []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod), framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                       "should unstage RWO volume when starting a second pod with different policy (Recursive -> MountOption)",
				csiDriverSELinuxEnabled:    true,
				firstPodSELinuxOpts:        &seLinuxOpts1,
				firstPodChangePolicy:       &recursive,
				startSecondPod:             true,
				secondPodSELinuxOpts:       &seLinuxOpts2,
				secondPodChangePolicy:      &mount,
				volumeMode:                 v1.ReadWriteOnce,
				expectedFirstMountOptions:  nil,
				expectedSecondMountOptions: []string{seLinuxMountOption2},
				expectedUnstage:            true,
				testTags:                   []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod), framework.WithFeatureGate(features.SELinuxMount)},
			},
		}
		for _, t := range tests {
			t := t
			testFunc := func(ctx context.Context) {
				if processLabel == "" {
					e2eskipper.Skipf("SELinux tests are supported only on %+v", getSupportedSELinuxDistros())
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
				_, claim, pod := m.createPodWithSELinux(ctx, accessModes, t.mountOptions, t.firstPodSELinuxOpts, t.firstPodChangePolicy)
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
				// count fresh CSI driver calls between the first and the second pod
				nodeStageMountOpts = nil
				nodePublishMountOpts = nil
				unstageCalls.Store(0)
				unpublishCalls.Store(0)
				stageCalls.Store(0)
				publishCalls.Store(0)

				// Skip scheduler, it would block scheduling the second pod with ReadWriteOncePod PV.
				pod, err = m.cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "getting the initial pod")
				nodeSelection := e2epod.NodeSelection{Name: pod.Spec.NodeName}
				pod2, err := startPausePodWithSELinuxOptions(f.ClientSet, claim, nodeSelection, f.Namespace.Name, t.secondPodSELinuxOpts, t.secondPodChangePolicy)
				framework.ExpectNoError(err, "creating second pod with SELinux context %s", t.secondPodSELinuxOpts)
				m.pods = append(m.pods, pod2)

				// Delete the initial pod only after kubelet processes the second pod and adds its volumes to
				// DesiredStateOfWorld.
				// In this state, any volume UnPublish / UnStage must be done because of SELinux contexts and not
				// because of random races because volumes of the second pod are not in DesiredStateOfWorld yet.
				ginkgo.By("Waiting for the second pod to start (or fail to start because of ReadWriteOncePod).")
				reason := events.FailedMountVolume
				var msg string
				if t.expectedUnstage {
					// This message is emitted before kubelet checks for ReadWriteOncePod
					msg = "conflicting SELinux labels of volume"
				} else {
					// Kubelet should re-use staged volume.
					if t.volumeMode == v1.ReadWriteOncePod {
						// Wait for the second pod to get stuck because of RWOP.
						msg = "volume uses the ReadWriteOncePod access mode and is already in use by another pod"
					} else {
						// There is nothing blocking the second pod from starting, wait for the second pod to fullly start.
						reason = string(events.StartedContainer)
						msg = "Started container"
					}
				}
				eventSelector := fields.Set{
					"involvedObject.kind":      "Pod",
					"involvedObject.name":      pod2.Name,
					"involvedObject.namespace": pod2.Namespace,
					"reason":                   reason,
				}.AsSelector().String()
				err = e2eevents.WaitTimeoutForEvent(ctx, m.cs, pod2.Namespace, eventSelector, msg, f.Timeouts.PodStart)
				framework.ExpectNoError(err, "waiting for event %q in the second test pod", msg)

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
			}
			// t.testTags is array and it's not possible to use It("name", func(){}, t.testTags...)
			// Compose It() arguments separately.
			args := []interface{}{
				t.name,
				testFunc,
			}
			args = append(args, t.testTags...)
			framework.It(args...)
		}
	})
})

var (
	// SELinux metrics that have volume_plugin and access_mode labels
	metricsWithVolumePluginLabel = sets.New[string](
		"volume_manager_selinux_volume_context_mismatch_errors_total",
		"volume_manager_selinux_volume_context_mismatch_warnings_total",
		"volume_manager_selinux_volumes_admitted_total",
	)
	// SELinuxMetrics that have only access_mode label
	metricsWithoutVolumePluginLabel = sets.New[string](
		"volume_manager_selinux_container_errors_total",
		"volume_manager_selinux_container_warnings_total",
		"volume_manager_selinux_pod_context_mismatch_errors_total",
		"volume_manager_selinux_pod_context_mismatch_warnings_total",
	)
	// All SELinux metrics
	allSELinuxMetrics = metricsWithoutVolumePluginLabel.Union(metricsWithVolumePluginLabel)
)

// While kubelet VolumeManager and KCM SELinuxWarningController are quite different components,
// their tests would have exactly the same setup, so we test both here.
var _ = utils.SIGDescribe("CSI Mock selinux on mount metrics and SELinuxWarningController", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-selinux-metrics")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	// [Serial]: the tests read global node metrics, so no other test changes them in parallel.
	f.Context("SELinuxMount metrics [LinuxOnly]", feature.SELinux, f.WithSerial(), func() {
		processLabel, _ := getDefaultContainerSELinuxLabels()
		// Make sure all options are set so system specific defaults are not used.
		seLinuxOpts1 := v1.SELinuxOptions{
			User:  "system_u",
			Role:  "system_r",
			Type:  processLabel,
			Level: "s0:c0,c1",
		}
		seLinuxOpts2 := v1.SELinuxOptions{
			User:  "system_u",
			Role:  "system_r",
			Type:  processLabel,
			Level: "s0:c98,c99",
		}
		recursive := v1.SELinuxChangePolicyRecursive
		mount := v1.SELinuxChangePolicyMountOption

		tests := []struct {
			name                             string
			csiDriverSELinuxEnabled          bool
			firstPodSELinuxOpts              *v1.SELinuxOptions
			firstPodChangePolicy             *v1.PodSELinuxChangePolicy
			secondPodSELinuxOpts             *v1.SELinuxOptions
			secondPodChangePolicy            *v1.PodSELinuxChangePolicy
			volumeMode                       v1.PersistentVolumeAccessMode
			waitForSecondPodStart            bool
			secondPodFailureEvent            string
			expectNodeIncreases              sets.Set[string] // For testing kubelet metrics
			expectControllerConflictProperty string           // For testing  SELinuxWarningController metrics + events
			testTags                         []interface{}    // SELinuxMountReadWriteOncePod and SELinuxChangePolicy are always added automatically
		}{
			{
				name:                    "warning is not bumped on two Pods with the same context on RWO volume",
				csiDriverSELinuxEnabled: true,
				firstPodSELinuxOpts:     &seLinuxOpts1,
				secondPodSELinuxOpts:    &seLinuxOpts1,
				volumeMode:              v1.ReadWriteOnce,
				waitForSecondPodStart:   true,
				expectNodeIncreases:     sets.New[string]( /* no metric is increased, admitted_total was already increased when the first pod started */ ),
				testTags:                []interface{}{feature.SELinuxMountReadWriteOncePodOnly},
			},
			{
				name:                             "warning is bumped on two Pods with a different context on RWO volume",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				secondPodSELinuxOpts:             &seLinuxOpts2,
				volumeMode:                       v1.ReadWriteOnce,
				waitForSecondPodStart:            true,
				expectNodeIncreases:              sets.New[string]("volume_manager_selinux_volume_context_mismatch_warnings_total"),
				expectControllerConflictProperty: "SELinuxLabel",
				testTags:                         []interface{}{feature.SELinuxMountReadWriteOncePodOnly},
			},
			{
				name:                             "warning is bumped on two Pods with different policies on RWO volume",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				firstPodChangePolicy:             nil,
				secondPodSELinuxOpts:             &seLinuxOpts1,
				secondPodChangePolicy:            &recursive,
				volumeMode:                       v1.ReadWriteOnce,
				waitForSecondPodStart:            true,
				expectNodeIncreases:              sets.New[string]("volume_manager_selinux_volume_context_mismatch_warnings_total"),
				expectControllerConflictProperty: "SELinuxChangePolicy",
				testTags:                         []interface{}{feature.SELinuxMountReadWriteOncePodOnly},
			},
			{
				name:                             "warning is not bumped on two Pods with Recursive policy and a different context on RWO volume",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				firstPodChangePolicy:             &recursive,
				secondPodSELinuxOpts:             &seLinuxOpts2,
				secondPodChangePolicy:            &recursive,
				volumeMode:                       v1.ReadWriteOnce,
				waitForSecondPodStart:            true,
				expectNodeIncreases:              sets.New[string]( /* no metric is increased, admitted_total was already increased when the first pod started */ ),
				expectControllerConflictProperty: "", /* SELinuxController does not emit any warning either */
				testTags:                         []interface{}{framework.WithFeatureGate(features.SELinuxChangePolicy), feature.SELinuxMountReadWriteOncePodOnly},
			},
			{
				name:                    "error is not bumped on two Pods with the same context on RWO volume and SELinuxMount enabled",
				csiDriverSELinuxEnabled: true,
				firstPodSELinuxOpts:     &seLinuxOpts1,
				secondPodSELinuxOpts:    &seLinuxOpts1,
				volumeMode:              v1.ReadWriteOnce,
				waitForSecondPodStart:   true,
				expectNodeIncreases:     sets.New[string]( /* no metric is increased, admitted_total was already increased when the first pod started */ ),
				testTags:                []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                             "error is bumped on two Pods with a different context on RWO volume and SELinuxMount enabled",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				secondPodSELinuxOpts:             &seLinuxOpts2,
				secondPodFailureEvent:            "conflicting SELinux labels of volume",
				volumeMode:                       v1.ReadWriteOnce,
				waitForSecondPodStart:            false,
				expectNodeIncreases:              sets.New[string]("volume_manager_selinux_volume_context_mismatch_errors_total"),
				expectControllerConflictProperty: "SELinuxLabel",
				testTags:                         []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                             "error is bumped on two Pods with a different policy on RWO volume and SELinuxMount enabled (nil + Recursive)",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				firstPodChangePolicy:             nil,
				secondPodSELinuxOpts:             &seLinuxOpts1,
				secondPodChangePolicy:            &recursive,
				secondPodFailureEvent:            "conflicting SELinux labels of volume",
				volumeMode:                       v1.ReadWriteOnce,
				waitForSecondPodStart:            false,
				expectNodeIncreases:              sets.New[string]("volume_manager_selinux_volume_context_mismatch_errors_total"),
				expectControllerConflictProperty: "SELinuxChangePolicy",
				testTags:                         []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                             "error is bumped on two Pods with a different policy on RWO volume and SELinuxMount enabled (Recursive + nil)",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				firstPodChangePolicy:             &recursive,
				secondPodSELinuxOpts:             &seLinuxOpts1,
				secondPodChangePolicy:            nil,
				secondPodFailureEvent:            "conflicting SELinux labels of volume",
				volumeMode:                       v1.ReadWriteOnce,
				waitForSecondPodStart:            false,
				expectNodeIncreases:              sets.New[string]("volume_manager_selinux_volume_context_mismatch_errors_total"),
				expectControllerConflictProperty: "SELinuxChangePolicy",
				testTags:                         []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                             "error is bumped on two Pods with a different policy on RWO volume and SELinuxMount enabled (Recursive + MountOption)",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				firstPodChangePolicy:             &recursive,
				secondPodSELinuxOpts:             &seLinuxOpts1,
				secondPodChangePolicy:            &mount,
				secondPodFailureEvent:            "conflicting SELinux labels of volume",
				volumeMode:                       v1.ReadWriteOnce,
				waitForSecondPodStart:            false,
				expectNodeIncreases:              sets.New[string]("volume_manager_selinux_volume_context_mismatch_errors_total"),
				expectControllerConflictProperty: "SELinuxChangePolicy",
				testTags:                         []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                             "error is bumped on two Pods with a different context on RWX volume and SELinuxMount enabled",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				secondPodSELinuxOpts:             &seLinuxOpts2,
				secondPodFailureEvent:            "conflicting SELinux labels of volume",
				volumeMode:                       v1.ReadWriteMany,
				waitForSecondPodStart:            false,
				expectNodeIncreases:              sets.New[string]("volume_manager_selinux_volume_context_mismatch_errors_total"),
				expectControllerConflictProperty: "SELinuxLabel",
				testTags:                         []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                             "error is not bumped on two Pods with Recursive policy and a different context on RWX volume",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				firstPodChangePolicy:             &recursive,
				secondPodSELinuxOpts:             &seLinuxOpts2,
				secondPodChangePolicy:            &recursive,
				volumeMode:                       v1.ReadWriteMany,
				waitForSecondPodStart:            true,
				expectNodeIncreases:              sets.New[string]( /* no metric is increased, admitted_total was already increased when the first pod started */ ),
				expectControllerConflictProperty: "", /* SELinuxController does not emit any warning either */
				testTags:                         []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                    "error is not bumped on two Pods with a different policy RWX volume (nil + MountOption)",
				csiDriverSELinuxEnabled: true,
				firstPodSELinuxOpts:     &seLinuxOpts1,
				firstPodChangePolicy:    &mount,
				secondPodSELinuxOpts:    &seLinuxOpts1,
				secondPodChangePolicy:   nil,
				volumeMode:              v1.ReadWriteMany,
				waitForSecondPodStart:   true,
				expectNodeIncreases:     sets.New[string]( /* no metric is increased, admitted_total was already increased when the first pod started */ ),
				testTags:                []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                    "error is not bumped on two Pods with a different policy RWX volume (MountOption + MountOption)",
				csiDriverSELinuxEnabled: true,
				firstPodSELinuxOpts:     &seLinuxOpts1,
				firstPodChangePolicy:    &mount,
				secondPodSELinuxOpts:    &seLinuxOpts1,
				secondPodChangePolicy:   &mount,
				volumeMode:              v1.ReadWriteMany,
				waitForSecondPodStart:   true,
				expectNodeIncreases:     sets.New[string]( /* no metric is increased, admitted_total was already increased when the first pod started */ ),
				testTags:                []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
			{
				name:                             "error is bumped on two Pods with a different context on RWOP volume",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				secondPodSELinuxOpts:             &seLinuxOpts2,
				secondPodFailureEvent:            "conflicting SELinux labels of volume",
				volumeMode:                       v1.ReadWriteOncePod,
				waitForSecondPodStart:            false,
				expectNodeIncreases:              sets.New[string]("volume_manager_selinux_volume_context_mismatch_errors_total"),
				expectControllerConflictProperty: "SELinuxLabel",
				testTags:                         []interface{}{framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod)},
			},
			{
				name:                             "error is bumped on two Pods with MountOption policy and a different context on RWOP volume",
				csiDriverSELinuxEnabled:          true,
				firstPodSELinuxOpts:              &seLinuxOpts1,
				firstPodChangePolicy:             &mount,
				secondPodSELinuxOpts:             &seLinuxOpts2,
				secondPodChangePolicy:            &mount,
				secondPodFailureEvent:            "conflicting SELinux labels of volume",
				volumeMode:                       v1.ReadWriteOncePod,
				waitForSecondPodStart:            false,
				expectNodeIncreases:              sets.New[string]("volume_manager_selinux_volume_context_mismatch_errors_total"),
				expectControllerConflictProperty: "SELinuxLabel",
				testTags:                         []interface{}{framework.WithFeatureGate(features.SELinuxMount)},
			},
		}
		for _, t := range tests {
			t := t
			testFunc := func(ctx context.Context) {
				if processLabel == "" {
					e2eskipper.Skipf("SELinux tests are supported only on %+v", getSupportedSELinuxDistros())
				}

				// Some metrics use CSI driver name as a label, which is "csi-mock-" + the namespace name.
				volumePluginLabel := "volume_plugin=\"kubernetes.io/csi/csi-mock-" + f.Namespace.Name + "\""
				grabber, err := e2emetrics.NewMetricsGrabber(ctx, f.ClientSet, nil, f.ClientConfig(), true /*kubelet*/, false /*scheduler*/, true /*controllers*/, false /*apiserver*/, false /*autoscaler*/, false /*snapshotController*/)
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
				_, claim, pod := m.createPodWithSELinux(ctx, accessModes, []string{}, t.firstPodSELinuxOpts, t.firstPodChangePolicy)
				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "starting the initial pod")

				ginkgo.By("Grabbing initial metrics")
				pod, err = m.cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "getting the initial pod")
				metrics, err := grabNodeMetrics(ctx, grabber, pod.Spec.NodeName, allSELinuxMetrics, volumePluginLabel)
				framework.ExpectNoError(err, "collecting the initial metrics")
				dumpMetrics(metrics)

				// Act
				ginkgo.By("Starting the second pod")
				// Skip scheduler, it would block scheduling the second pod with ReadWriteOncePod PV.
				nodeSelection := e2epod.NodeSelection{Name: pod.Spec.NodeName}
				pod2, err := startPausePodWithSELinuxOptions(f.ClientSet, claim, nodeSelection, f.Namespace.Name, t.secondPodSELinuxOpts, t.secondPodChangePolicy)
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

				// Assert: count the kubelet metrics
				expectIncreaseWithLabels := addLabels(t.expectNodeIncreases, volumePluginLabel, t.volumeMode)
				framework.Logf("Waiting for changes of metrics %+v", expectIncreaseWithLabels)
				err = waitForNodeMetricIncrease(ctx, grabber, pod.Spec.NodeName, volumePluginLabel, allSELinuxMetrics, expectIncreaseWithLabels, metrics, framework.PodStartShortTimeout)
				framework.ExpectNoError(err, "waiting for metrics %s to increase", t.expectNodeIncreases)
				if t.expectControllerConflictProperty != "" {
					// Assert: count the KCM metrics + events
					// We don't need to compare the initial and final KCM metrics,
					// KCM metrics report exact pod namespaces+names as labels and the metric value is always "1".
					err = waitForControllerMetric(ctx, grabber, f.Namespace.Name, pod.Name, pod2.Name, t.expectControllerConflictProperty, framework.PodStartShortTimeout)
					framework.ExpectNoError(err, "while waiting for metrics from KCM")
					// Check the controler generated a conflict event on the first pod
					err = waitForConflictEvent(ctx, m.cs, pod, pod2, t.expectControllerConflictProperty, f.Timeouts.PodStart)
					framework.ExpectNoError(err, "while waiting for an event on the first pod")
					// Check the controler generated event on the second pod
					err = waitForConflictEvent(ctx, m.cs, pod2, pod, t.expectControllerConflictProperty, f.Timeouts.PodStart)
					framework.ExpectNoError(err, "while waiting for an event on the second pod")
				}
			}
			// t.testTags is array and it's not possible to use It("name", func(){xxx}, t.testTags...)
			// Compose It() arguments separately.
			args := []interface{}{
				t.name,
				testFunc,
				framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod),
				framework.WithFeatureGate(features.SELinuxChangePolicy),
			}
			args = append(args, t.testTags...)
			framework.It(args...)
		}
	})
})

func grabNodeMetrics(ctx context.Context, grabber *e2emetrics.Grabber, nodeName string, metricNames sets.Set[string], volumePluginLabel string) (map[string]float64, error) {
	response, err := grabber.GrabFromKubelet(ctx, nodeName)
	framework.ExpectNoError(err)

	metrics := map[string]float64{}
	for _, samples := range response {
		if len(samples) == 0 {
			continue
		}
		// For each metric + label combination, remember the last sample
		for i := range samples {
			// E.g. "volume_manager_selinux_pod_context_mismatch_errors_total"
			metricName := samples[i].Metric[testutil.MetricNameLabel]
			if metricNames.Has(string(metricName)) {
				// E.g. "volume_manager_selinux_pod_context_mismatch_errors_total{access_mode="RWOP",volume_plugin="kubernetes.io/csi/csi-mock-ns"}
				metricNameWithLabels := samples[i].Metric.String()
				// Filter out metrics of any other volume plugin
				if strings.Contains(metricNameWithLabels, "volume_plugin=") && !strings.Contains(metricNameWithLabels, volumePluginLabel) {
					continue
				}
				// Overwrite any previous value, so only the last one is stored.
				metrics[metricNameWithLabels] = float64(samples[i].Value)
			}
		}
	}

	return metrics, nil
}

func grabKCMSELinuxMetrics(ctx context.Context, grabber *e2emetrics.Grabber, namespace string) (map[string]float64, error) {
	response, err := grabber.GrabFromControllerManager(ctx)
	if err != nil {
		return nil, err
	}

	metrics := map[string]float64{}
	for _, samples := range response {
		if len(samples) == 0 {
			continue
		}
		// For each metric + label combination, remember the last sample
		for i := range samples {
			// E.g. "selinux_warning_controller_selinux_volume_conflict"
			metricName := samples[i].Metric[testutil.MetricNameLabel]
			if metricName != controllerSELinuxMetricName {
				continue
			}

			metricNamespace := samples[0].Metric["pod1_namespace"]
			if string(metricNamespace) != namespace {
				continue
			}
			// E.g. selinux_warning_controller_selinux_volume_conflict{pod1_name="testpod-c1",pod1_namespace="default",pod1_value="system_u:object_r:container_file_t:s0:c0,c1",pod2_name="testpod-c2",pod2_namespace="default",pod2_value="system_u:object_r:container_file_t:s0:c0,c2",property="SELinuxLabel"} 1
			metricNameWithLabels := samples[i].Metric.String()
			// Overwrite any previous value, so only the last one is stored.
			metrics[metricNameWithLabels] = float64(samples[i].Value)
		}
	}
	framework.Logf("KCM metrics")
	dumpMetrics(metrics)

	return metrics, nil
}

func waitForNodeMetricIncrease(ctx context.Context, grabber *e2emetrics.Grabber, nodeName string, volumePluginLabel string, allMetricNames, expectedIncreaseNames sets.Set[string], initialValues map[string]float64, timeout time.Duration) error {
	var noIncreaseMetrics sets.Set[string]
	var metrics map[string]float64

	err := wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		var err error
		metrics, err = grabNodeMetrics(ctx, grabber, nodeName, allMetricNames, volumePluginLabel)
		if err != nil {
			return false, err
		}

		noIncreaseMetrics = sets.New[string]()
		// Always evaluate all SELinux metrics to check that the other metrics are not unexpectedly increased.
		for name := range metrics {
			if expectedIncreaseNames.Has(name) {
				if metrics[name] <= initialValues[name] {
					noIncreaseMetrics.Insert(name)
				}
			} else {
				// Expect the metric to be stable
				if initialValues[name] != metrics[name] {
					return false, fmt.Errorf("metric %s unexpectedly increased to %v", name, metrics[name])
				}
			}
		}
		return noIncreaseMetrics.Len() == 0, nil
	})

	ginkgo.By("Dumping final node metrics")
	dumpMetrics(metrics)

	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("timed out waiting for node metrics %v", noIncreaseMetrics.UnsortedList())
	}
	return err
}

func waitForControllerMetric(ctx context.Context, grabber *e2emetrics.Grabber, namespace, pod1Name, pod2Name, propertyName string, timeout time.Duration) error {
	var metrics map[string]float64

	expectLabels := []string{
		fmt.Sprintf("pod1_name=%q", pod1Name),
		fmt.Sprintf("pod2_name=%q", pod2Name),
		fmt.Sprintf("pod1_namespace=%q", namespace),
		fmt.Sprintf("pod2_namespace=%q", namespace),
		fmt.Sprintf("property=%q", propertyName),
	}
	framework.Logf("Waiting for KCM metric %s{%+v}", controllerSELinuxMetricName, expectLabels)

	err := wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		var err error
		metrics, err = grabKCMSELinuxMetrics(ctx, grabber, namespace)
		if err != nil {
			return false, err
		}

		foundMatch := false
		for metric := range metrics {
			allLabelsMatched := true
			for _, expectedLabel := range expectLabels {
				if !strings.Contains(metric, expectedLabel) {
					allLabelsMatched = false
				}
			}
			if allLabelsMatched {
				foundMatch = true
			}
		}

		return foundMatch, nil
	})
	if errors.Is(err, e2emetrics.MetricsGrabbingDisabledError) {
		ginkgo.By("Cannot grab metrics from kube-controller-manager in this e2e job, skipping metrics checks")
		return nil
	}

	ginkgo.By("Dumping final KCM metrics")
	dumpMetrics(metrics)

	if err != nil {
		return fmt.Errorf("error waiting for KCM metrics %s{%+v}: %w", controllerSELinuxMetricName, expectLabels, err)
	}
	return err
}

func waitForConflictEvent(ctx context.Context, cs clientset.Interface, pod, otherPod *v1.Pod, expectControllerConflictProperty string, timeout time.Duration) error {
	eventSelector := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      pod.Name,
		"involvedObject.namespace": pod.Namespace,
		"reason":                   expectControllerConflictProperty + "Conflict",
	}.AsSelector().String()
	// msg is a substring of the full event message that does not contain the actual SELinux label (too long, too variable)
	// Full event: SELinuxLabel "system_u:system_r:container_t:s0:c0,c1" conflicts with pod pvc-volume-tester-djqqd that uses the same volume as this pod with SELinuxLabel "system_u:system_r:container_t:s0:c98,c99". If both pods land on the same node, only one of them may access the volume.
	msg := fmt.Sprintf("conflicts with pod %s that uses the same volume as this pod with %s", otherPod.Name, expectControllerConflictProperty)
	ginkgo.By(fmt.Sprintf("Waiting for the SELinux controller event on pod %q: %q", pod.Name, msg))
	return e2eevents.WaitTimeoutForEvent(ctx, cs, pod.Namespace, eventSelector, msg, timeout)
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

// Add labels to the metric name based on the current test case
func addLabels(metricNames sets.Set[string], volumePluginLabel string, accessMode v1.PersistentVolumeAccessMode) sets.Set[string] {
	ret := sets.New[string]()
	accessModeShortString := helper.GetAccessModesAsString([]v1.PersistentVolumeAccessMode{accessMode})

	for metricName := range metricNames {
		var metricWithLabels string
		if metricsWithVolumePluginLabel.Has(metricName) {
			metricWithLabels = fmt.Sprintf("%s{access_mode=\"%s\", %s}", metricName, accessModeShortString, volumePluginLabel)
		} else {
			metricWithLabels = fmt.Sprintf("%s{access_mode=\"%s\"}", metricName, accessModeShortString)
		}

		ret.Insert(metricWithLabels)
	}

	return ret
}

func getDefaultContainerSELinuxLabels() (processLabel string, fileLabel string) {
	defaultLabels := defaultSELinuxLabels[framework.TestContext.NodeOSDistro]
	// This function can return "" for unknown distros!
	// SELinux tests should be skipped on those in their ginkgo.It().
	return defaultLabels.defaultProcessLabel, defaultLabels.defaultFileLabel
}

func getSupportedSELinuxDistros() []string {
	distros := make([]string, 0, len(defaultSELinuxLabels))
	for distro := range defaultSELinuxLabels {
		distros = append(distros, distro)
	}
	return distros
}
