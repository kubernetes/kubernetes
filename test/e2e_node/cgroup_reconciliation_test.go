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
	comment                   string
}

// reconcileTrigger describes how a batch of manually-set cgroup values is
// pushed back toward Kubelet's desired state, and which reconciliation
// expectation applies to each property under that trigger.
type reconcileTrigger struct {
	label string
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

		// 0b. Run any trigger-specific setup (e.g. bring pods up so that a later
		// deletion is what fires reconciliation) before capturing the baseline.
		if trigger.setup != nil {
			trigger.setup(ctx)
		}

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
					if !strings.Contains(currentValue, s.effectiveManual) {
						failures = append(failures, fmt.Sprintf("%s (was reset! got %s, expected %s)", s.filename, currentValue, s.effectiveManual))
					}
				}
				return failures
			}).WithTimeout(70*time.Second).WithPolling(2*time.Second).Should(gomega.BeEmpty(),
				"The following cgroups were unexpectedly reconciled (reset to defaults)")
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

	// Pod lifecycle events funnel through killPod -> UpdateQOSCgroups (pod
	// deletion) and pod admission -> UpdateQOSCgroups, both of which call
	// cgroupManager.Update on the pod root cgroup. On the systemd driver that
	// Update issues SetUnitProperties, which resets every unspecified property
	// on the slice; on cgroupfs the (empty) Update writes no files and is a
	// no-op. This test creates one pod per QoS class, then deletes all of them,
	// so pod churn is what deterministically fires reconciliation instead of a
	// stray leftover pod terminating mid-verification.
	ginkgo.It("should follow reconciliation rules (Pod Churn)", func(ctx context.Context) {
		var churnPods []*v1.Pod
		trigger := reconcileTrigger{
			label: "pod churn (create+delete guaranteed/burstable/besteffort pods)",
			setup: func(ctx context.Context) {
				churnPods = createQoSPods(ctx, f)
			},
			fire: func(ctx context.Context) {
				deleteQoSPods(ctx, f, churnPods)
			},
			shouldReconcile: func(p cgroupTestParam) bool { return p.expectReconcileOnPodChurn },
		}
		testCgroups(ctx, getPodRootCgroupPath(), getCgroupParams(), trigger)
	})
})

// createQoSPods creates one running pause pod for each QoS class (Guaranteed,
// Burstable, BestEffort) and returns them so a later deletion can drive Kubelet
// QoS cgroup reconciliation.
func createQoSPods(ctx context.Context, f *framework.Framework) []*v1.Pod {
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
				Name: fmt.Sprintf("cgroup-churn-%s-%s", s.qos, string(uuid.NewUUID())),
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

func getCgroupParams() []cgroupTestParam {
	isSystemd := getCgroupDriver() == "systemd"
	// Pod churn drives UpdateQOSCgroups, which calls cgroupManager.Update on the
	// pod root cgroup with an (essentially empty) config. On cgroupfs that Update
	// writes no files, so nothing is reset. On systemd it issues SetUnitProperties,
	// which resets every unspecified property on the slice. Unlike a Kubelet
	// restart, pod churn does NOT re-run Node Allocatable enforcement, so even the
	// Kubelet-managed properties (cpu.weight, memory.max, pids.max, cpuset.cpus)
	// are only reset under systemd (as a side-effect), never on cgroupfs.
	return []cgroupTestParam{
		{
			filename:                  "cpu.max",
			manualValue:               "max 100000",
			altManualValue:            "100000 100000",
			expectReconcileOnRestart:  isSystemd,
			expectReconcilePeriodic:   false,
			expectReconcileOnPodChurn: isSystemd,
			comment:                   "Kubelet does not set a limit here; reconciliation on restart is a side-effect of systemd resetting unspecified properties to unbounded.",
		},
		{
			filename:                  "cpu.weight",
			manualValue:               "100",
			altManualValue:            "512",
			expectReconcileOnRestart:  true,
			expectReconcilePeriodic:   false, // Root cgroup management is not truly periodic (startup-only)
			expectReconcileOnPodChurn: isSystemd,
			comment:                   "Node Allocatable enforcement sets this (via shares) to ensure the pod hierarchy gets its fair share of CPU.",
		},
		{
			filename:                  "memory.max",
			manualValue:               "1073741824", // 1Gi
			altManualValue:            "536870912",  // 512Mi
			expectReconcileOnRestart:  true,
			expectReconcilePeriodic:   false, // Root cgroup management is not truly periodic (startup-only)
			expectReconcileOnPodChurn: isSystemd,
			comment:                   "Node Allocatable enforcement sets this to the Node Allocatable value to protect the host from pod aggregate usage.",
		},
		{
			filename:                  "pids.max",
			manualValue:               "1000",
			altManualValue:            "1234",
			expectReconcileOnRestart:  true,
			expectReconcilePeriodic:   false, // Root cgroup management is not truly periodic (startup-only)
			expectReconcileOnPodChurn: isSystemd,
			comment:                   "Node Allocatable enforcement sets this to protect the node from PID exhaustion across all pods.",
		},
		{
			filename:                  "cpuset.cpus",
			manualValue:               "0",
			altManualValue:            "0-0", // More robust alternative for various environments
			expectReconcileOnRestart:  true,
			expectReconcilePeriodic:   false, // Root cgroup management is not truly periodic (startup-only)
			expectReconcileOnPodChurn: isSystemd,
			comment:                   "Node Allocatable enforcement sets this to restrict pods to the designated allocatable CPUs.",
		},
		{
			filename:                  "memory.min",
			manualValue:               "115343360", // 110Mi (page aligned)
			altManualValue:            "209715200", // 200Mi (page aligned)
			expectReconcileOnRestart:  isSystemd,
			expectReconcilePeriodic:   false,
			expectReconcileOnPodChurn: isSystemd,
			comment:                   "Kubelet skips memory.min management when MemoryQoS is disabled, but it is reconciled on systemd restart as a side-effect.",
		},
		{
			filename:                  "memory.high",
			manualValue:               "1073741824", // 1Gi
			altManualValue:            "536870912",  // 512Mi
			expectReconcileOnRestart:  isSystemd,
			expectReconcilePeriodic:   false,
			expectReconcileOnPodChurn: isSystemd,
			comment:                   "Not managed by Kubelet at the root level; reconciliation on restart is a side-effect of systemd resetting unspecified properties.",
		},
		{
			filename:                  "memory.oom.group",
			manualValue:               "1",
			altManualValue:            "0",
			expectReconcileOnRestart:  isSystemd,
			expectReconcilePeriodic:   false,
			expectReconcileOnPodChurn: isSystemd,
			comment:                   "Not managed by Kubelet at the root level; reconciliation on restart is a side-effect of systemd resetting unspecified properties.",
		},
		{
			filename:                  "io.weight",
			manualValue:               "200",
			altManualValue:            "150",
			expectReconcileOnRestart:  isSystemd,
			expectReconcilePeriodic:   false,
			expectReconcileOnPodChurn: isSystemd,
			comment:                   "Not managed by Kubelet; reconciliation on restart is a side-effect of systemd resetting unspecified properties.",
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
