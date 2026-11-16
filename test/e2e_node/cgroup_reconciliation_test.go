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
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

type cgroupTestParam struct {
	filename                  string
	manualValue               string
	altManualValue            string
	expectReconcileOnRestart  bool
	expectReconcilePeriodic   bool
	expectReconcileOnPodChurn bool
	// expectReconcileOnPodCountTransition covers the node going from N pods
	// to 0, which is distinct from steady-state churn (see getCgroupParams).
	expectReconcileOnPodCountTransition bool
	comment                             string
}

// reconcileTrigger describes how a batch of manually-set cgroup values is
// pushed back toward Kubelet's desired state, and which reconciliation
// expectation applies to each property under that trigger.
type reconcileTrigger struct {
	label string
	// zeroPodBaseline skips the long-lived sentinel pods, so pod churn moves
	// the node between 0 and N pods. Real nodes always run system pods, so
	// only the transition test wants this.
	zeroPodBaseline bool
	// expectSubtreeControlChange is whether the trigger is expected to change
	// the pod root's enabled controller set (cgroup.subtree_control).
	expectSubtreeControlChange bool
	// setup runs before the manual values are written. Used to bring pods up
	// so that their later deletion is what fires reconciliation. May be nil.
	setup func(ctx context.Context)
	// fire performs the action that (potentially) reconciles the values.
	fire func(ctx context.Context)
	// shouldReconcile reports whether the given property is expected to be
	// reset away from its manual value by this trigger.
	shouldReconcile func(p cgroupTestParam) bool
}

var _ = SIGDescribe("Cgroup reconciliation", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("cgroup-reconciliation")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func(ctx context.Context) {
		if !IsCgroup2UnifiedMode() {
			ginkgo.Skip("This test requires cgroups v2")
		}
		// Pod root cgroup reconciliation is only exercised meaningfully under the
		// systemd driver, where UpdateQOSCgroups -> SetUnitProperties resets the
		// slice's unspecified properties. On cgroupfs the equivalent update writes
		// no files, so the reconciliation expectations encoded here do not hold.
		if driver := getCgroupDriver(); driver != "systemd" {
			ginkgo.Skip(fmt.Sprintf("This test requires the systemd cgroup driver (detected %q)", driver))
		}
	})

	testCgroups := func(ctx context.Context, cgroupPath string, params []cgroupTestParam, trigger reconcileTrigger) {
		driver := getCgroupDriver()
		ginkgo.By(fmt.Sprintf("Running batch test for %d cgroups in %s (driver=%s, trigger=%s)", len(params), cgroupPath, driver, trigger.label))

		// 0. Ensure Kubelet is started and has performed its initial reconciliation
		ginkgo.By("Ensuring Kubelet is started before capturing baseline")
		waitForKubeletToStart(ctx, f)

		// 0a. Ensure the node is quiescent: no pod cgroups from previous tests may
		// linger anywhere under the pod hierarchy. A pod finishing its teardown
		// causes systemd to re-realize the (ancestor) root slice and reset its
		// values, which would corrupt the reconciliation windows below. DeleteSync
		// in other tests only waits for the API server, not for the kubelet to
		// finish on-node cleanup, so we must poll the cgroup paths directly.
		ginkgo.By("Waiting for all pod cgroups from other tests to be removed")
		gomega.Eventually(ctx, func() ([]string, error) {
			return podCgroupDirs(cgroupPath)
		}).WithTimeout(f.Timeouts.PodDelete).WithPolling(2*time.Second).Should(gomega.BeEmpty(),
			"pod cgroups from other tests must be gone before verifying root cgroup reconciliation")

		// 0b. Real nodes always run system pods, so the pod root's enabled
		// controller set is stable under churn. Keep one pod per QoS class
		// running for the whole test so we measure that steady state rather
		// than the 0 <-> N transition, which changes cgroup.subtree_control
		// and makes systemd re-realize (and rewrite) the slice.
		if !trigger.zeroPodBaseline {
			sentinels := createQoSPods(ctx, f, "cgroup-sentinel")
			ginkgo.DeferCleanup(deleteQoSPods, f, sentinels)

			sentinelUIDs := sets.New[string]()
			for _, p := range sentinels {
				sentinelUIDs.Insert(string(p.UID))
			}
			ginkgo.By("Waiting for exactly the sentinel pod cgroups to be present")
			gomega.Eventually(ctx, func() ([]string, error) {
				uids, err := podCgroupUIDs(cgroupPath)
				return sets.List(uids), err
			}).WithTimeout(f.Timeouts.PodStart).WithPolling(2*time.Second).Should(gomega.Equal(sets.List(sentinelUIDs)),
				"only the sentinel pods may have cgroups under the pod root before capturing the baseline")
		}

		// 0c. Run any trigger-specific setup (e.g. bring pods up so that a later
		// deletion is what fires reconciliation) before capturing the baseline.
		if trigger.setup != nil {
			trigger.setup(ctx)
		}

		subtreeControlPath := filepath.Join(cgroupPath, "cgroup.subtree_control")
		readSubtreeControl := func() (string, error) {
			b, err := os.ReadFile(subtreeControlPath)
			return strings.TrimSpace(string(b)), err
		}
		subtreeControlBefore, err := readSubtreeControl()
		framework.ExpectNoError(err, "reading %s", subtreeControlPath)
		framework.Logf("%s before trigger: %q", subtreeControlPath, subtreeControlBefore)

		type testState struct {
			filename        string
			filePath        string
			defaultValue    string
			effectiveManual string
			shouldReconcile bool
		}

		var states []testState
		for _, p := range params {
			filePath := filepath.Join(cgroupPath, p.filename)
			if _, err := os.Stat(filePath); os.IsNotExist(err) {
				framework.Logf("Skipping %s: file not found", p.filename)
				continue
			}

			// 1. Read and log initial value (stabilized by Kubelet)
			initialValueBytes, err := os.ReadFile(filePath)
			framework.ExpectNoError(err)
			defaultValue := strings.TrimSpace(string(initialValueBytes))
			framework.Logf("Initial (stabilized) value of %s: %s (%s)", filePath, defaultValue, p.comment)

			shouldReconcile := trigger.shouldReconcile(p)
			framework.Logf("Cgroup %s expectation: shouldReconcile=%v (trigger=%s, periodic=%v, restart=%v, podChurn=%v)",
				p.filename, shouldReconcile, trigger.label, p.expectReconcilePeriodic, p.expectReconcileOnRestart, p.expectReconcileOnPodChurn)

			// 2. Determine manual value. It MUST be different from the current state to observe reconciliation.
			effectiveManual := p.manualValue
			// Use exact match to avoid false positives with substrings (e.g. "0-23" contains "0")
			if defaultValue == effectiveManual {
				effectiveManual = p.altManualValue
				framework.Logf("Manual value %s matches current state for %s; using alternative %s", p.manualValue, p.filename, effectiveManual)
			}

			states = append(states, testState{
				filename:        p.filename,
				filePath:        filePath,
				defaultValue:    defaultValue,
				effectiveManual: effectiveManual,
				shouldReconcile: shouldReconcile,
			})

			// 3. Set manual value
			ginkgo.By(fmt.Sprintf("Setting manual value %s for %s", effectiveManual, filePath))
			err = os.WriteFile(filePath, []byte(effectiveManual), 0644)
			framework.ExpectNoError(err)
		}

		if len(states) == 0 {
			return
		}

		// Cleanup: Restore values for ALL files in the batch to original state.
		defer func() {
			for _, s := range states {
				currentBytes, err := os.ReadFile(s.filePath)
				if err == nil && strings.TrimSpace(string(currentBytes)) != s.defaultValue {
					ginkgo.By(fmt.Sprintf("Cleanup: Restoring %s to original value %s (currently %s)", s.filePath, s.defaultValue, strings.TrimSpace(string(currentBytes))))
					if err := os.WriteFile(s.filePath, []byte(s.defaultValue), 0644); err != nil {
						framework.Logf("Warning: failed to restore %s: %v", s.filePath, err)
					}
				}
			}
		}()

		// 4. Trigger reconciliation (once per batch)
		ginkgo.By(fmt.Sprintf("Triggering reconciliation via %s", trigger.label))
		trigger.fire(ctx)

		// 5. Verify - Group by expectation for aggregate reporting
		var reconciledStates []testState
		var nonReconciledStates []testState
		for _, s := range states {
			if s.shouldReconcile {
				reconciledStates = append(reconciledStates, s)
			} else {
				nonReconciledStates = append(nonReconciledStates, s)
			}
		}

		// 5a. Verify those that should reconcile (aggregate Eventually)
		if len(reconciledStates) > 0 {
			ginkgo.By(fmt.Sprintf("Polling for %d cgroups to be reconciled back", len(reconciledStates)))
			gomega.Eventually(ctx, func() []string {
				var failures []string
				for _, s := range reconciledStates {
					currentValueBytes, _ := os.ReadFile(s.filePath)
					currentValue := strings.TrimSpace(string(currentValueBytes))
					if currentValue == s.effectiveManual {
						failures = append(failures, fmt.Sprintf("%s (still at manual %s)", s.filename, s.effectiveManual))
					} else if !strings.Contains(currentValue, s.defaultValue) {
						framework.Logf("Notice: %s reconciled to %s (captured baseline: %s)", s.filename, currentValue, s.defaultValue)
					}
				}
				return failures
			}).WithTimeout(75*time.Second).WithPolling(2*time.Second).Should(gomega.BeEmpty(),
				"The following cgroups failed to reconcile back from manual values")
		}

		// 5b. Verify those that should NOT reconcile (aggregate Consistently)
		if len(nonReconciledStates) > 0 {
			ginkgo.By(fmt.Sprintf("Verifying %d unmanaged cgroups stay at manual values (70s wait)", len(nonReconciledStates)))
			gomega.Consistently(ctx, func() []string {
				var failures []string
				for _, s := range nonReconciledStates {
					currentValueBytes, err := os.ReadFile(s.filePath)
					if err != nil {
						failures = append(failures, fmt.Sprintf("%s (read error: %v)", s.filename, err))
						continue
					}
					currentValue := strings.TrimSpace(string(currentValueBytes))
					// Exact match: a reset cpuset.cpus of "0-7" contains a manual "0".
					if currentValue != s.effectiveManual {
						failures = append(failures, fmt.Sprintf("%s (was reset! got %s, expected %s)", s.filename, currentValue, s.effectiveManual))
					}
				}
				return failures
			}).WithTimeout(70*time.Second).WithPolling(2*time.Second).Should(gomega.BeEmpty(),
				"The following cgroups were unexpectedly reconciled (reset to defaults)")
		}

		// 6. Guard against measuring the wrong mechanism: a controller set
		// change re-realizes the slice regardless of what the trigger did.
		if trigger.expectSubtreeControlChange {
			gomega.Eventually(ctx, readSubtreeControl).WithTimeout(f.Timeouts.PodDelete).WithPolling(2*time.Second).ShouldNot(gomega.Equal(subtreeControlBefore),
				"%s was expected to change under trigger %q; without it this test no longer exercises the 0 <-> N pod transition", subtreeControlPath, trigger.label)
		} else {
			subtreeControlAfter, err := readSubtreeControl()
			framework.ExpectNoError(err, "reading %s", subtreeControlPath)
			gomega.Expect(subtreeControlAfter).To(gomega.Equal(subtreeControlBefore),
				"%s changed under trigger %q; systemd re-realized the slice, so reconciliation results above do not reflect the trigger itself", subtreeControlPath, trigger.label)
		}
	}

	ginkgo.It("should follow reconciliation rules (Periodic)", func(ctx context.Context) {
		// No explicit trigger: the passage of time inside the verification
		// windows exercises Kubelet's periodic QoS cgroup update loop.
		trigger := reconcileTrigger{
			label:           "periodic",
			fire:            func(ctx context.Context) {},
			shouldReconcile: func(p cgroupTestParam) bool { return p.expectReconcilePeriodic },
		}
		testCgroups(ctx, getPodRootCgroupPath(), getCgroupParams(), trigger)
	})

	ginkgo.It("should follow reconciliation rules (Restart)", func(ctx context.Context) {
		trigger := reconcileTrigger{
			label: "kubelet restart",
			fire: func(ctx context.Context) {
				ginkgo.By("Restarting Kubelet")
				restartKubelet(ctx, true)
				waitForKubeletToStart(ctx, f)
			},
			shouldReconcile: func(p cgroupTestParam) bool { return p.expectReconcileOnRestart },
		}
		testCgroups(ctx, getPodRootCgroupPath(), getCgroupParams(), trigger)
	})

	// Pod lifecycle events funnel through UpdateQOSCgroups, which calls
	// cgroupManager.Update on the pod root cgroup. With the sentinel pods
	// keeping the node at >= 1 pod (as on any real node), the pod root's
	// controller set never changes, systemd never re-realizes the slice, and
	// manually written values survive churn.
	ginkgo.It("should follow reconciliation rules (Pod Churn)", func(ctx context.Context) {
		var churnPods []*v1.Pod
		trigger := reconcileTrigger{
			label: "pod churn (create+delete guaranteed/burstable/besteffort pods on top of sentinels)",
			setup: func(ctx context.Context) {
				churnPods = createQoSPods(ctx, f, "cgroup-churn")
			},
			fire: func(ctx context.Context) {
				deleteQoSPods(ctx, f, churnPods)
			},
			shouldReconcile: func(p cgroupTestParam) bool { return p.expectReconcileOnPodChurn },
		}
		testCgroups(ctx, getPodRootCgroupPath(), getCgroupParams(), trigger)
	})

	// Going from N pods to 0 empties the pod root, which changes its enabled
	// controller set (cgroup.subtree_control). systemd then re-realizes
	// kubepods.slice from its stored unit properties, overwriting anything
	// written directly to cgroupfs. This only happens on an otherwise empty
	// node; it is not what normal churn does.
	ginkgo.It("should follow reconciliation rules (Pod Count Transition N -> 0)", func(ctx context.Context) {
		var churnPods []*v1.Pod
		trigger := reconcileTrigger{
			label:                      "pod count transition (delete the only guaranteed/burstable/besteffort pods)",
			zeroPodBaseline:            true,
			expectSubtreeControlChange: true,
			setup: func(ctx context.Context) {
				churnPods = createQoSPods(ctx, f, "cgroup-transition")
			},
			fire: func(ctx context.Context) {
				deleteQoSPods(ctx, f, churnPods)
			},
			shouldReconcile: func(p cgroupTestParam) bool { return p.expectReconcileOnPodCountTransition },
		}
		testCgroups(ctx, getPodRootCgroupPath(), getCgroupParams(), trigger)
	})
})

// createQoSPods creates one running pause pod for each QoS class (Guaranteed,
// Burstable, BestEffort), named with the given prefix.
func createQoSPods(ctx context.Context, f *framework.Framework, namePrefix string) []*v1.Pod {
	podClient := e2epod.NewPodClient(f)
	specs := []struct {
		qos       string
		resources v1.ResourceRequirements
	}{
		// Guaranteed: requests == limits for cpu and memory.
		{"guaranteed", getResourceRequirements(getResourceList("50m", "50Mi"), getResourceList("50m", "50Mi"))},
		// Burstable: requests set, limits higher.
		{"burstable", getResourceRequirements(getResourceList("50m", "50Mi"), getResourceList("100m", "100Mi"))},
		// BestEffort: no requests or limits.
		{"besteffort", v1.ResourceRequirements{}},
	}

	pods := make([]*v1.Pod, 0, len(specs))
	for _, s := range specs {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("%s-%s-%s", namePrefix, s.qos, string(uuid.NewUUID())),
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{
					{
						Name:      "pause",
						Image:     imageutils.GetPauseImageName(),
						Resources: s.resources,
					},
				},
			},
		}
		ginkgo.By(fmt.Sprintf("Creating %s pod %s", s.qos, pod.Name))
		pods = append(pods, podClient.CreateSync(ctx, pod))
	}
	return pods
}

// deleteQoSPods deletes the given pods and waits for their termination. Each
// deletion triggers killPod -> UpdateQOSCgroups on the pod root cgroup.
func deleteQoSPods(ctx context.Context, f *framework.Framework, pods []*v1.Pod) {
	podClient := e2epod.NewPodClient(f)
	for _, pod := range pods {
		ginkgo.By(fmt.Sprintf("Deleting pod %s to trigger QoS cgroup reconciliation on termination", pod.Name))
		podClient.DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
	}
}

// podCgroupDirs walks the entire pod cgroup hierarchy rooted at cgroupRoot and
// returns every pod-level cgroup directory still present. A non-empty result
// means some pod (possibly from another test) still has a cgroup on the node;
// its teardown would re-realize the root slice and disturb reconciliation
// verification.
func podCgroupDirs(cgroupRoot string) ([]string, error) {
	var dirs []string
	err := filepath.WalkDir(cgroupRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// Pod cgroup directories can vanish mid-walk as the kubelet cleans
			// them up; treat a disappeared entry as "not there".
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if d.IsDir() && isPodCgroupName(d.Name()) {
			dirs = append(dirs, path)
		}
		return nil
	})
	if os.IsNotExist(err) {
		return nil, nil
	}
	return dirs, err
}

// isPodCgroupName reports whether a cgroup directory base name is a pod-level
// cgroup (as opposed to the root or a QoS-tier slice). Pod cgroups are named
// after the pod UID under the pod root / QoS slices:
//   - systemd:  kubepods-pod<uid>.slice, kubepods-burstable-pod<uid>.slice, ...
//   - cgroupfs: pod<uid>
//
// QoS tiers (kubepods-burstable.slice / kubepods-besteffort.slice, burstable /
// besteffort) and the root itself are intentionally excluded.
func isPodCgroupName(name string) bool {
	if strings.HasSuffix(name, ".slice") {
		return strings.Contains(name, "-pod")
	}
	return strings.HasPrefix(name, "pod")
}

// podUIDFromCgroupName extracts the pod UID from a name accepted by
// isPodCgroupName.
func podUIDFromCgroupName(name string) string {
	if base, ok := strings.CutSuffix(name, ".slice"); ok {
		// systemd escapes '-' in the UID as '_'.
		uid := base[strings.LastIndex(base, "-pod")+len("-pod"):]
		return strings.ReplaceAll(uid, "_", "-")
	}
	return strings.TrimPrefix(name, "pod")
}

// podCgroupUIDs returns the UIDs of all pods that have a cgroup under cgroupRoot.
func podCgroupUIDs(cgroupRoot string) (sets.Set[string], error) {
	dirs, err := podCgroupDirs(cgroupRoot)
	if err != nil {
		return nil, err
	}
	uids := sets.New[string]()
	for _, d := range dirs {
		uids.Insert(podUIDFromCgroupName(filepath.Base(d)))
	}
	return uids, nil
}

func getCgroupParams() []cgroupTestParam {
	isSystemd := getCgroupDriver() == "systemd"
	// Pod churn drives UpdateQOSCgroups, which calls cgroupManager.Update on the
	// pod root cgroup. That alone does not rewrite the root's attributes. They
	// are only rewritten when the pod count crosses 0 <-> N: that changes the
	// pod root's enabled controller set (cgroup.subtree_control), and systemd
	// re-realizes kubepods.slice from its stored unit properties, discarding
	// anything written directly to cgroupfs. Real nodes always run system pods,
	// so steady-state churn leaves manual values in place. The transition does
	// NOT re-run Node Allocatable enforcement either, so even Kubelet-managed
	// properties (cpu.weight, memory.max, pids.max, cpuset.cpus) are only reset
	// as a systemd side-effect.
	return []cgroupTestParam{
		{
			filename:                            "cpu.max",
			manualValue:                         "max 100000",
			altManualValue:                      "100000 100000",
			expectReconcileOnRestart:            isSystemd,
			expectReconcilePeriodic:             false,
			expectReconcileOnPodChurn:           false,
			expectReconcileOnPodCountTransition: isSystemd,
			comment:                             "Kubelet does not set a limit here; reconciliation on restart is a side-effect of systemd resetting unspecified properties to unbounded.",
		},
		{
			filename:                            "cpu.weight",
			manualValue:                         "100",
			altManualValue:                      "512",
			expectReconcileOnRestart:            true,
			expectReconcilePeriodic:             false, // Root cgroup management is not truly periodic (startup-only)
			expectReconcileOnPodChurn:           false,
			expectReconcileOnPodCountTransition: isSystemd,
			comment:                             "Node Allocatable enforcement sets this (via shares) to ensure the pod hierarchy gets its fair share of CPU.",
		},
		{
			filename:                            "memory.max",
			manualValue:                         "1073741824", // 1Gi
			altManualValue:                      "536870912",  // 512Mi
			expectReconcileOnRestart:            true,
			expectReconcilePeriodic:             false, // Root cgroup management is not truly periodic (startup-only)
			expectReconcileOnPodChurn:           false,
			expectReconcileOnPodCountTransition: isSystemd,
			comment:                             "Node Allocatable enforcement sets this to the Node Allocatable value to protect the host from pod aggregate usage.",
		},
		{
			filename:                            "pids.max",
			manualValue:                         "1000",
			altManualValue:                      "1234",
			expectReconcileOnRestart:            true,
			expectReconcilePeriodic:             false, // Root cgroup management is not truly periodic (startup-only)
			expectReconcileOnPodChurn:           false,
			expectReconcileOnPodCountTransition: isSystemd,
			comment:                             "Node Allocatable enforcement sets this to protect the node from PID exhaustion across all pods.",
		},
		{
			filename:                            "cpuset.cpus",
			manualValue:                         "0",
			altManualValue:                      "0-0", // More robust alternative for various environments
			expectReconcileOnRestart:            true,
			expectReconcilePeriodic:             false, // Root cgroup management is not truly periodic (startup-only)
			expectReconcileOnPodChurn:           false,
			expectReconcileOnPodCountTransition: isSystemd,
			comment:                             "Node Allocatable enforcement sets this to restrict pods to the designated allocatable CPUs.",
		},
		{
			filename:                            "memory.min",
			manualValue:                         "115343360", // 110Mi (page aligned)
			altManualValue:                      "209715200", // 200Mi (page aligned)
			expectReconcileOnRestart:            isSystemd,
			expectReconcilePeriodic:             false,
			expectReconcileOnPodChurn:           false,
			expectReconcileOnPodCountTransition: isSystemd,
			comment:                             "Kubelet skips memory.min management when MemoryQoS is disabled, but it is reconciled on systemd restart as a side-effect.",
		},
		{
			filename:                            "memory.high",
			manualValue:                         "1073741824", // 1Gi
			altManualValue:                      "536870912",  // 512Mi
			expectReconcileOnRestart:            isSystemd,
			expectReconcilePeriodic:             false,
			expectReconcileOnPodChurn:           false,
			expectReconcileOnPodCountTransition: isSystemd,
			comment:                             "Not managed by Kubelet at the root level; reconciliation on restart is a side-effect of systemd resetting unspecified properties.",
		},
		{
			filename:                            "memory.oom.group",
			manualValue:                         "1",
			altManualValue:                      "0",
			expectReconcileOnRestart:            isSystemd,
			expectReconcilePeriodic:             false,
			expectReconcileOnPodChurn:           false,
			expectReconcileOnPodCountTransition: isSystemd,
			comment:                             "Not managed by Kubelet at the root level; reconciliation on restart is a side-effect of systemd resetting unspecified properties.",
		},
		{
			filename: "io.weight",
			// Keyed format: the kernel reads back "default N" even when "N"
			// is written, so write the read-back form for exact comparison.
			manualValue:                         "default 200",
			altManualValue:                      "default 150",
			expectReconcileOnRestart:            isSystemd,
			expectReconcilePeriodic:             false,
			expectReconcileOnPodChurn:           false,
			expectReconcileOnPodCountTransition: isSystemd,
			comment:                             "Not managed by Kubelet; reconciliation on restart is a side-effect of systemd resetting unspecified properties.",
		},
	}
}

func getCgroupDriver() string {
	driver := "cgroupfs"
	source := "fallback"
	if kubeletCfg != nil && kubeletCfg.CgroupDriver != "" {
		driver = kubeletCfg.CgroupDriver
		source = "kubelet config"
	}
	framework.Logf("Detected cgroup driver: %s (source: %s)", driver, source)
	return driver
}

func getPodRootCgroupPath() string {
	// Kubelet root is "kubepods" for cgroupfs and "kubepods.slice" for systemd.
	// We check the Kubelet configuration first, then fallback to filesystem discovery.
	driver := getCgroupDriver()

	var path string
	if driver == "systemd" {
		path = filepath.Join(cgroupRoot, "kubepods.slice")
	} else {
		path = filepath.Join(cgroupRoot, "kubepods")
	}

	// Double check if it exists, if not try the other one
	if _, err := os.Stat(path); err == nil {
		return path
	}

	// Fallback discovery
	for _, p := range []string{"kubepods.slice", "kubepods"} {
		path = filepath.Join(cgroupRoot, p)
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return filepath.Join(cgroupRoot, "kubepods") // Default
}
