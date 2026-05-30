/*
Copyright 2024 The Kubernetes Authors.

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
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Kubelet subpath stale bind mount [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("subpath-stale-mount")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// This test stops and restarts the kubelet. Mark it Serial+Disruptive so
	// it does not run concurrently with other tests that assume a live kubelet.
	ginkgo.Context("stale bind mount after simulated FUSE network filesystem disruption",
		framework.WithSerial(), framework.WithDisruptive(), func() {

			f.It("should remount stale subpath bind mount after FUSE daemon restart [LinuxOnly]",
				f.WithDisruptive(), func(ctx context.Context) {

					// bindfs must be installed
					if _, err := exec.LookPath("bindfs"); err != nil {
						if out, err2 := exec.CommandContext(ctx, "sudo", "which", "bindfs").CombinedOutput(); err2 != nil || len(strings.TrimSpace(string(out))) == 0 {
							ginkgo.Skip("bindfs not found; install with: sudo apt-get install -y bindfs")
						}
					}

					// 1. set up FUSE mount via bindfs
					// /mnt/fuse-src: the real data directory (stays on disk)
					// /mnt/fuse-mnt: the FUSE passthrough mount point
					fuseDir := fmt.Sprintf("/mnt/stale-fuse-%s", rand.String(5))
					fuseSrc := filepath.Join(fuseDir, "src")
					fuseMnt := filepath.Join(fuseDir, "mnt")

					for _, dir := range []string{fuseSrc, fuseMnt} {
						framework.ExpectNoError(os.MkdirAll(dir, 0755), "creating FUSE directory %s", dir)
					}
					ginkgo.DeferCleanup(func() {
						exec.Command("sudo", "umount", "-l", fuseMnt).Run() //nolint:errcheck
						_ = os.RemoveAll(fuseDir)
					})

					// Seed a subPath directory in the FUSE source.
					subDir := filepath.Join(fuseSrc, "data")
					framework.ExpectNoError(os.Mkdir(subDir, 0755), "creating subPath source dir")

					// Start bindfs: fuseSrc → fuseMnt
					ginkgo.By(fmt.Sprintf("Starting bindfs: %s → %s", fuseSrc, fuseMnt))
					bindfsCmd := exec.CommandContext(ctx, "sudo", "bindfs",
						"-o", "allow_other",
						fuseSrc, fuseMnt)
					framework.ExpectNoError(bindfsCmd.Run(), "starting bindfs")

					// Verify the FUSE mount is live.
					ginkgo.By("Verifying FUSE mount is live")
					out, err := exec.CommandContext(ctx, "grep", fuseMnt, "/proc/mounts").CombinedOutput()
					framework.ExpectNoError(err, "verifying bindfs is mounted: %s", string(out))
					framework.Logf("bindfs mount: %s", strings.TrimSpace(string(out)))

					// 2. create pod A with a subPath on the FUSE mount
					volName := "stale-vol"
					podA := &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "subpath-stale-a-" + rand.String(5),
							Namespace: f.Namespace.Name,
						},
						Spec: v1.PodSpec{
							NodeName:      framework.TestContext.NodeName,
							RestartPolicy: v1.RestartPolicyAlways,
							Containers: []v1.Container{
								{
									Name:  "c",
									Image: busyboxImage,
									// The command reads from the mounted subPath immediately.
									// If the bind mount is dead (ENOTCONN), the 'ls' will fail
									// and the container will exit non-zero, triggering a restart.
									Command: []string{"sh", "-c", "ls /data && sleep infinity"},
									VolumeMounts: []v1.VolumeMount{
										{
											Name:      volName,
											MountPath: "/data",
											SubPath:   "data",
										},
									},
								},
							},
							Volumes: []v1.Volume{
								{
									Name: volName,
									VolumeSource: v1.VolumeSource{
										HostPath: &v1.HostPathVolumeSource{Path: fuseMnt},
									},
								},
							},
						},
					}

					ginkgo.By(fmt.Sprintf("Creating pod A (%s) with subPath on FUSE mount", podA.Name))
					podA, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, podA, metav1.CreateOptions{})
					framework.ExpectNoError(err, "creating pod A")
					ginkgo.DeferCleanup(e2epod.DeletePodWithWait, f.ClientSet, podA)

					framework.ExpectNoError(
						e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, podA.Name, f.Namespace.Name, f.Timeouts.PodStart),
						"waiting for pod A to reach Running",
					)

					// Refresh to get UID.
					podA, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podA.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "refreshing pod A")

					// 3. locate the subPath bind-mount target
					// Pattern: /var/lib/kubelet/pods/<uid>/volume-subpaths/<vol>/<container>/<index>
					subpathGlob := fmt.Sprintf("/var/lib/kubelet/pods/%s/volume-subpaths/%s/*/*", podA.UID, volName)
					ginkgo.By(fmt.Sprintf("Locating subPath bind mounts at %s", subpathGlob))

					var bindTargets []string
					gomega.Eventually(ctx, func() []string {
						matches, _ := filepath.Glob(subpathGlob)
						return matches
					}, 15*time.Second, time.Second).ShouldNot(gomega.BeEmpty(),
						"subPath bind-mount target must exist after pod A reaches Running")

					bindTargets, err = filepath.Glob(subpathGlob)
					framework.ExpectNoError(err, "globbing subPath bind targets")
					gomega.Expect(bindTargets).NotTo(gomega.BeEmpty(), "expected at least one subPath bind-mount target")
					framework.Logf("Found subPath bind targets: %v", bindTargets)

					// Verify the bind targets are currently mounted (in /proc/mounts).
					for _, target := range bindTargets {
						out, err := exec.CommandContext(ctx, "grep", target, "/proc/mounts").CombinedOutput()
						framework.ExpectNoError(err, "bind target %s must be in /proc/mounts: %s", target, string(out))
						framework.Logf("bind target mount: %s", strings.TrimSpace(string(out)))
					}

					// 4. stop kubelet
					ginkgo.By("Stopping kubelet (simulating node restart)")
					kubeletStart := mustStopKubelet(ctx, f)
					// Always restart kubelet on test exit, even on failure.
					ginkgo.DeferCleanup(func(ctx context.Context) { kubeletStart(ctx) })

					// 5. remove pod A's sandbox via crictl
					// crictl stopp + crictl rmp removes the sandbox and all its
					// containers from the CRI layer while the kubelet is stopped.
					ginkgo.By("Removing pod A's sandbox via crictl (force full pod restart)")
					podsOut, err := exec.CommandContext(ctx, "sudo", "crictl", "pods", "-q",
						"--label", fmt.Sprintf("io.kubernetes.pod.uid=%s", podA.UID)).CombinedOutput()
					framework.ExpectNoError(err, "crictl pods failed: %s", string(podsOut))
					sandboxIDs := strings.Fields(strings.TrimSpace(string(podsOut)))
					gomega.Expect(sandboxIDs).NotTo(gomega.BeEmpty(),
						"expected at least one sandbox for pod A UID %s", podA.UID)
					for _, sid := range sandboxIDs {
						framework.Logf("crictl stopp %s", sid)
						stopOut, _ := exec.CommandContext(ctx, "sudo", "crictl", "stopp", sid).CombinedOutput()
						framework.Logf("stopp output: %s", strings.TrimSpace(string(stopOut)))
						framework.Logf("crictl rmp %s", sid)
						rmOut, err := exec.CommandContext(ctx, "sudo", "crictl", "rmp", sid).CombinedOutput()
						framework.ExpectNoError(err, "crictl rmp %s: %s", sid, string(rmOut))
					}

					// 6. kill the bindfs daemon
					// After SIGKILL the bindfs process exits but the kernel keeps
					// the FUSE mount entry in /proc/mounts. syscall.Statfs on the
					// mount will return ENOTCONN.
					ginkgo.By("Killing bindfs daemon to simulate FUSE network filesystem disruption")
					killOut, err := exec.CommandContext(ctx, "sudo", "pkill", "-9", "-f",
						fmt.Sprintf("bindfs.*%s", fuseSrc)).CombinedOutput()
					if err != nil {
						// pkill returns 1 if no process matched
						framework.Logf("pkill bindfs: %v, output: %s", err, string(killOut))
					}
					time.Sleep(500 * time.Millisecond)

					// Verify the FUSE mount is still in /proc/mounts (zombie mount).
					ginkgo.By("Verifying FUSE mount is still in /proc/mounts (zombie)")
					mountOut, _ := exec.CommandContext(ctx, "grep", fuseMnt, "/proc/mounts").CombinedOutput()
					framework.Logf("/proc/mounts for fuseMnt: %q", strings.TrimSpace(string(mountOut)))
					gomega.Expect(strings.TrimSpace(string(mountOut))).NotTo(gomega.BeEmpty(),
						"FUSE mount must still be in /proc/mounts after bindfs is killed — "+
							"this is required to reproduce the stale-mount condition")

					// Verify the bind-mount targets are also still in /proc/mounts.
					for _, target := range bindTargets {
						tOut, _ := exec.CommandContext(ctx, "grep", target, "/proc/mounts").CombinedOutput()
						framework.Logf("bind target after kill: %q", strings.TrimSpace(string(tOut)))
						gomega.Expect(strings.TrimSpace(string(tOut))).NotTo(gomega.BeEmpty(),
							"bind target %s must still be in /proc/mounts after bindfs killed", target)
					}

					// 7. restart bindfs (recover the FUSE source)
					ginkgo.By("Lazy-unmounting zombie FUSE source mount (so bindfs can remount)")
					umountOut, _ := exec.CommandContext(ctx, "sudo", "umount", "-l", fuseMnt).CombinedOutput()
					framework.Logf("umount -l %s: %s", fuseMnt, strings.TrimSpace(string(umountOut)))

					ginkgo.By("Restarting bindfs (simulating storage system recovery)")
					restartCmd := exec.CommandContext(ctx, "sudo", "bindfs",
						"-o", "allow_other",
						fuseSrc, fuseMnt)
					framework.ExpectNoError(restartCmd.Run(), "restarting bindfs")
					// Brief pause to let FUSE mount settle.
					time.Sleep(300 * time.Millisecond)
					framework.Logf("bindfs restarted")

					// Confirm the FUSE source mount is live again.
					liveOut, _ := exec.CommandContext(ctx, "grep", fuseMnt, "/proc/mounts").CombinedOutput()
					framework.Logf("fuseMnt after restart: %q", strings.TrimSpace(string(liveOut)))

					// The subpath bind-mount targets must STILL be in /proc/mounts as
					// zombies — that is the condition prepareSubpathTarget will see.
					for _, target := range bindTargets {
						tOut, _ := exec.CommandContext(ctx, "grep", target, "/proc/mounts").CombinedOutput()
						framework.Logf("bind target still zombie: %q", strings.TrimSpace(string(tOut)))
						gomega.Expect(strings.TrimSpace(string(tOut))).NotTo(gomega.BeEmpty(),
							"bind target %s must still be in /proc/mounts as zombie for test to be valid", target)
					}

					// 8. restart kubelet
					// Pod A is still alive in the API server. The restarted kubelet
					// re-syncs pod A with its original UID.
					kubeletStart(ctx)

					// 9. verify pod A returns to Running
					ginkgo.By("Waiting for pod A's container to restart (restart count ≥ 1)")
					podStartTimeout := f.Timeouts.PodStart

					err = wait.PollUntilContextTimeout(ctx, 3*time.Second, podStartTimeout, true,
						func(ctx context.Context) (bool, error) {
							pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podA.Name, metav1.GetOptions{})
							if err != nil {
								framework.Logf("Error getting pod A: %v", err)
								return false, nil
							}
							if len(pod.Status.ContainerStatuses) == 0 {
								framework.Logf("Pod A has no container statuses yet")
								return false, nil
							}
							restarts := pod.Status.ContainerStatuses[0].RestartCount
							phase := pod.Status.Phase
							framework.Logf("Pod A restart count: %d, phase: %s", restarts, phase)
							return restarts >= 1, nil
						},
					)
					framework.ExpectNoError(err,
						"pod A's container did not restart — "+
							"prepareSubpathTarget may have failed after kubelet restart",
					)

					// Container started successfully after remount
					ginkgo.By("Verifying pod A's container is Running after stale FUSE mount detection and remount")
					err = wait.PollUntilContextTimeout(ctx, 3*time.Second, podStartTimeout, true,
						func(ctx context.Context) (bool, error) {
							pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podA.Name, metav1.GetOptions{})
							if err != nil {
								framework.Logf("Error getting pod A: %v", err)
								return false, nil
							}
							for _, cs := range pod.Status.ContainerStatuses {
								if cs.Name == "c" && cs.State.Running != nil {
									framework.Logf("Pod A container c is Running (started at %v), restarts=%d",
										cs.State.Running.StartedAt, cs.RestartCount)
									return true, nil
								}
							}
							framework.Logf("Pod A phase: %s, container c not yet in Running state", pod.Status.Phase)
							return false, nil
						},
					)
					framework.ExpectNoError(err,
						"pod A's container never reached Running state after kubelet restart — "+
							"the stale FUSE bind mount was NOT detected and remounted; "+
							"check kubelet logs for prepareSubpathTarget errors",
					)

					// Confirm the fix remounted successfully.
					ginkgo.By("Confirming subPath prepare errors are 0 or 1 (fix remounted successfully)")
					events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{
						FieldSelector: "involvedObject.name=" + podA.Name,
					})
					framework.ExpectNoError(err, "listing events for pod A")
					subPathErrCount := 0
					for _, ev := range events.Items {
						if strings.Contains(ev.Message, "failed to prepare subPath") {
							subPathErrCount++
							framework.Logf("subPath error event: %s", ev.Message)
						}
					}
					framework.Logf("Total 'failed to prepare subPath' events: %d (expect 0 or 1)", subPathErrCount)
					gomega.Expect(subPathErrCount).To(gomega.BeNumerically("<=", 1),
						"expected at most 1 'failed to prepare subPath' event; "+
							">1 means the fix failed to remount the stale bind mount correctly")
				})
		})
})
