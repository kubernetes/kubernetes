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
	"slices"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	downwardAPIUpdateAnnotation = "e2e.k8s.io/value"
	downwardAPIUpdateDir        = "/etc/podinfo"
	downwardAPIUpdateFile       = "value"

	// Far beyond the test's runtime, so the periodic pod resync cannot be
	// what refreshes the volume.
	downwardAPISyncFrequency = 10 * time.Minute

	// Event-driven propagation takes up to about 1s on an idle node: before
	// syncing, the pod worker waits for a pod status cache entry newer than
	// its last sync, which the generic PLEG refreshes every relist (1s). The
	// periodic resync would take downwardAPISyncFrequency. The bound sits far
	// from both, so it checks the mechanism rather than node performance.
	downwardAPIUpdateBound = 10 * time.Second
)

// The conformance tests in test/e2e/common/storage/downwardapi_volume.go and
// projected_downwardapi.go verify that annotation/label changes eventually
// reach the volume, with a multi-minute timeout so they hold on any conformant
// cluster. They pass even if propagation silently degrades to the kubelet's
// periodic resync.
//
// This test pins the event-driven path: a pod update triggers a pod sync,
// which calls ReprocessPod in WaitForAttachAndMount, which remounts the volume
// because the downward API plugin requires a remount. syncFrequency is raised
// so the periodic resync cannot satisfy the bound; any pass is necessarily
// event-driven. Controllers that gate containers on downward API annotations
// can depend on this path.
var _ = SIGDescribe("Downward API volume update propagation", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("downwardapi-update")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("with periodic pod resync pushed out of reach", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SyncFrequency = metav1.Duration{Duration: downwardAPISyncFrequency}
		})

		ginkgo.It("should update downward API volume via pod update event, not periodic resync", func(ctx context.Context) {
			pod := e2epod.NewPodClient(f).CreateSync(ctx, downwardAPIUpdatePod("v1"))
			containerName := pod.Spec.Containers[0].Name

			readVolumeValue := func() (string, error) {
				stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, containerName,
					"cat", downwardAPIUpdateDir+"/"+downwardAPIUpdateFile)
				if err != nil {
					return "", fmt.Errorf("reading %s/%s: %w (stderr: %q)", downwardAPIUpdateDir, downwardAPIUpdateFile, err, stderr)
				}
				return strings.TrimSpace(stdout), nil
			}

			ginkgo.By("Waiting for the initial annotation value to be visible in the volume")
			gomega.Eventually(ctx, readVolumeValue).WithTimeout(f.Timeouts.PodStart).WithPolling(time.Second).Should(gomega.Equal("v1"))

			// The second update proves reprocessing repeats after the first
			// remount; further updates take the identical path.
			var latencies []time.Duration
			for _, value := range []string{"v2", "v3"} {
				ginkgo.By(fmt.Sprintf("Setting %s=%s", downwardAPIUpdateAnnotation, value))
				patch := fmt.Sprintf(`{"metadata":{"annotations":{%q:%q}}}`, downwardAPIUpdateAnnotation, value)
				start := time.Now()
				_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Patch(ctx, pod.Name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
				framework.ExpectNoError(err, "patching annotation %s=%s on pod %s", downwardAPIUpdateAnnotation, value, pod.Name)

				gomega.Eventually(ctx, readVolumeValue).WithTimeout(downwardAPIUpdateBound).WithPolling(500*time.Millisecond).Should(gomega.Equal(value),
					"annotation %s=%s did not reach the downward API volume within %v. With syncFrequency=%v the periodic resync cannot deliver it in time, so the event-driven remount path (pod update -> SyncPod -> ReprocessPod -> remount) is likely broken.",
					downwardAPIUpdateAnnotation, value, downwardAPIUpdateBound, downwardAPISyncFrequency)

				latency := time.Since(start)
				latencies = append(latencies, latency)
				framework.Logf("Annotation %s=%s visible in volume after %v", downwardAPIUpdateAnnotation, value, latency)
			}
			framework.Logf("Downward API propagation over %d updates: max %v", len(latencies), slices.Max(latencies))
		})
	})
})

// downwardAPIUpdatePod returns a pod with downwardAPIUpdateAnnotation set to
// initialValue and projected into a downward API volume.
func downwardAPIUpdatePod(initialValue string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "downwardapi-update-" + string(uuid.NewUUID()),
			Annotations: map[string]string{downwardAPIUpdateAnnotation: initialValue},
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:    "reader",
					Image:   busyboxImage,
					Command: []string{"sh", "-c", e2epod.InfiniteSleepCommand},
					VolumeMounts: []v1.VolumeMount{
						{Name: "podinfo", MountPath: downwardAPIUpdateDir},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "podinfo",
					VolumeSource: v1.VolumeSource{
						DownwardAPI: &v1.DownwardAPIVolumeSource{
							Items: []v1.DownwardAPIVolumeFile{
								{
									Path:     downwardAPIUpdateFile,
									FieldRef: &v1.ObjectFieldSelector{FieldPath: fmt.Sprintf("metadata.annotations['%s']", downwardAPIUpdateAnnotation)},
								},
							},
						},
					},
				},
			},
		},
	}
}
