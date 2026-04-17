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
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

type cgroupTestParam struct {
	filename                 string
	manualValue              string
	altManualValue           string
	expectReconcileOnRestart bool
	expectReconcilePeriodic  bool
	comment                  string
}

var _ = SIGDescribe("Cgroup reconciliation", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("cgroup-reconciliation")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func(ctx context.Context) {
		if !IsCgroup2UnifiedMode() {
			ginkgo.Skip("This test requires cgroups v2")
		}
	})

	testCgroups := func(ctx context.Context, cgroupPath string, params []cgroupTestParam, useRestart bool) {
		driver := getCgroupDriver()
		ginkgo.By(fmt.Sprintf("Running batch test for %d cgroups in %s (driver=%s, restart=%v)", len(params), cgroupPath, driver, useRestart))

		// 0. Ensure Kubelet is started and has performed its initial reconciliation
		ginkgo.By("Ensuring Kubelet is started before capturing baseline")
		waitForKubeletToStart(ctx, f)

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

			shouldReconcile := p.expectReconcilePeriodic
			if useRestart {
				shouldReconcile = p.expectReconcileOnRestart
			}
			framework.Logf("Cgroup %s expectation: shouldReconcile=%v (periodic=%v, restart=%v, modeRestart=%v)",
				p.filename, shouldReconcile, p.expectReconcilePeriodic, p.expectReconcileOnRestart, useRestart)

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
		if useRestart {
			ginkgo.By("Restarting Kubelet")
			restartKubelet(ctx, true)
			waitForKubeletToStart(ctx, f)
		}

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
		testCgroups(ctx, getPodRootCgroupPath(), getCgroupParams(), false)
	})

	ginkgo.It("should follow reconciliation rules (Restart)", func(ctx context.Context) {
		testCgroups(ctx, getPodRootCgroupPath(), getCgroupParams(), true)
	})
})

func getCgroupParams() []cgroupTestParam {
	isSystemd := getCgroupDriver() == "systemd"
	return []cgroupTestParam{
		{
			filename:                 "cpu.max",
			manualValue:              "max 100000",
			altManualValue:           "100000 100000",
			expectReconcileOnRestart: isSystemd,
			expectReconcilePeriodic:  false,
			comment:                  "Kubelet does not set a limit here; reconciliation on restart is a side-effect of systemd resetting unspecified properties to unbounded.",
		},
		{
			filename:                 "cpu.weight",
			manualValue:              "100",
			altManualValue:           "512",
			expectReconcileOnRestart: true,
			expectReconcilePeriodic:  false, // Root cgroup management is not truly periodic (startup-only)
			comment:                  "Node Allocatable enforcement sets this (via shares) to ensure the pod hierarchy gets its fair share of CPU.",
		},
		{
			filename:                 "memory.max",
			manualValue:              "1073741824", // 1Gi
			altManualValue:           "536870912",  // 512Mi
			expectReconcileOnRestart: true,
			expectReconcilePeriodic:  false, // Root cgroup management is not truly periodic (startup-only)
			comment:                  "Node Allocatable enforcement sets this to the Node Allocatable value to protect the host from pod aggregate usage.",
		},
		{
			filename:                 "pids.max",
			manualValue:              "1000",
			altManualValue:           "1234",
			expectReconcileOnRestart: true,
			expectReconcilePeriodic:  false, // Root cgroup management is not truly periodic (startup-only)
			comment:                  "Node Allocatable enforcement sets this to protect the node from PID exhaustion across all pods.",
		},
		{
			filename:                 "cpuset.cpus",
			manualValue:              "0",
			altManualValue:           "0-0", // More robust alternative for various environments
			expectReconcileOnRestart: true,
			expectReconcilePeriodic:  false, // Root cgroup management is not truly periodic (startup-only)
			comment:                  "Node Allocatable enforcement sets this to restrict pods to the designated allocatable CPUs.",
		},
		{
			filename:                 "memory.min",
			manualValue:              "115343360", // 110Mi (page aligned)
			altManualValue:           "209715200", // 200Mi (page aligned)
			expectReconcileOnRestart: isSystemd,
			expectReconcilePeriodic:  false,
			comment:                  "Kubelet skips memory.min management when MemoryQoS is disabled, but it is reconciled on systemd restart as a side-effect.",
		},
		{
			filename:                 "memory.high",
			manualValue:              "1073741824", // 1Gi
			altManualValue:           "536870912",  // 512Mi
			expectReconcileOnRestart: isSystemd,
			expectReconcilePeriodic:  false,
			comment:                  "Not managed by Kubelet at the root level; reconciliation on restart is a side-effect of systemd resetting unspecified properties.",
		},
		{
			filename:                 "memory.oom.group",
			manualValue:              "1",
			altManualValue:           "0",
			expectReconcileOnRestart: isSystemd,
			expectReconcilePeriodic:  false,
			comment:                  "Not managed by Kubelet at the root level; reconciliation on restart is a side-effect of systemd resetting unspecified properties.",
		},
		{
			filename:                 "io.weight",
			manualValue:              "200",
			altManualValue:           "150",
			expectReconcileOnRestart: isSystemd,
			expectReconcilePeriodic:  false,
			comment:                  "Not managed by Kubelet; reconciliation on restart is a side-effect of systemd resetting unspecified properties.",
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
