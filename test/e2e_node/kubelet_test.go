/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_node

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apiUnversioned "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/davecgh/go-spew/spew"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Kubelet", func() {
	f := framework.NewDefaultFramework("kubelet-test")
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})
	Context("when scheduling a busybox command in a pod", func() {
		podName := "busybox-scheduling-" + string(uuid.NewUUID())
		It("it should print the output to logs", func() {
			podClient.CreateSync(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					// Don't restart the Pod since it is expected to exit
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:   ImageRegistry[busyBoxImage],
							Name:    podName,
							Command: []string{"sh", "-c", "echo 'Hello World' ; sleep 240"},
						},
					},
				},
			})
			Eventually(func() string {
				sinceTime := apiUnversioned.NewTime(time.Now().Add(time.Duration(-1 * time.Hour)))
				rc, err := podClient.GetLogs(podName, &api.PodLogOptions{SinceTime: &sinceTime}).Stream()
				if err != nil {
					return ""
				}
				defer rc.Close()
				buf := new(bytes.Buffer)
				buf.ReadFrom(rc)
				return buf.String()
			}, time.Minute, time.Second*4).Should(Equal("Hello World\n"))
		})
	})

	Context("when scheduling a read only busybox container", func() {
		podName := "busybox-readonly-fs" + string(uuid.NewUUID())
		It("it should not write to root filesystem", func() {
			isReadOnly := true
			podClient.CreateSync(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					// Don't restart the Pod since it is expected to exit
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:   ImageRegistry[busyBoxImage],
							Name:    podName,
							Command: []string{"sh", "-c", "echo test > /file; sleep 240"},
							SecurityContext: &api.SecurityContext{
								ReadOnlyRootFilesystem: &isReadOnly,
							},
						},
					},
				},
			})
			Eventually(func() string {
				rc, err := podClient.GetLogs(podName, &api.PodLogOptions{}).Stream()
				if err != nil {
					return ""
				}
				defer rc.Close()
				buf := new(bytes.Buffer)
				buf.ReadFrom(rc)
				return buf.String()
			}, time.Minute, time.Second*4).Should(Equal("sh: can't create /file: Read-only file system\n"))
		})
	})
	Describe("metrics api", func() {
		Context("when querying /stats/summary", func() {
			It("it should report resource usage through the stats api", func() {
				podNamePrefix := "stats-busybox-" + string(uuid.NewUUID())
				volumeNamePrefix := "test-empty-dir"
				podNames, volumes := createSummaryTestPods(f.PodClient(), podNamePrefix, 2, volumeNamePrefix)
				By("Returning stats summary")
				summary := stats.Summary{}
				Eventually(func() error {
					resp, err := http.Get(*kubeletAddress + "/stats/summary")
					if err != nil {
						return fmt.Errorf("Failed to get /stats/summary - %v", err)
					}
					contentsBytes, err := ioutil.ReadAll(resp.Body)
					if err != nil {
						return fmt.Errorf("Failed to read /stats/summary - %+v", resp)
					}
					contents := string(contentsBytes)
					decoder := json.NewDecoder(strings.NewReader(contents))
					err = decoder.Decode(&summary)
					if err != nil {
						return fmt.Errorf("Failed to parse /stats/summary to go struct: %+v", resp)
					}
					missingPods := podsMissingFromSummary(summary, podNames)
					if missingPods.Len() != 0 {
						return fmt.Errorf("expected pods not found. Following pods are missing - %v", missingPods)
					}
					missingVolumes := volumesMissingFromSummary(summary, volumes)
					if missingVolumes.Len() != 0 {
						return fmt.Errorf("expected volumes not found. Following volumes are missing - %v", missingVolumes)
					}
					if err := testSummaryMetrics(summary, podNamePrefix); err != nil {
						return err
					}
					return nil
				}, 5*time.Minute, time.Second*4).Should(BeNil())
			})
		})
	})
})

const (
	containerSuffix = "-c"
)

func createSummaryTestPods(podClient *framework.PodClient, podNamePrefix string, count int, volumeNamePrefix string) (sets.String, sets.String) {
	podNames := sets.NewString()
	volumes := sets.NewString(volumeNamePrefix)
	for i := 0; i < count; i++ {
		podNames.Insert(fmt.Sprintf("%s%v", podNamePrefix, i))
	}

	var pods []*api.Pod
	for _, podName := range podNames.List() {
		pods = append(pods, &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: podName,
			},
			Spec: api.PodSpec{
				// Don't restart the Pod since it is expected to exit
				RestartPolicy: api.RestartPolicyNever,
				Containers: []api.Container{
					{
						Image:   ImageRegistry[busyBoxImage],
						Command: []string{"sh", "-c", "while true; do echo 'hello world' | tee /test-empty-dir-mnt/file ; sleep 1; done"},
						Name:    podName + containerSuffix,
						VolumeMounts: []api.VolumeMount{
							{MountPath: "/test-empty-dir-mnt", Name: volumeNamePrefix},
						},
					},
				},
				SecurityContext: &api.PodSecurityContext{
					SELinuxOptions: &api.SELinuxOptions{
						Level: "s0",
					},
				},
				Volumes: []api.Volume{
					// TODO: Test secret volumes
					// TODO: Test hostpath volumes
					{Name: volumeNamePrefix, VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				},
			},
		})
	}
	podClient.CreateBatch(pods)

	return podNames, volumes
}

// Returns pods missing from summary.
func podsMissingFromSummary(s stats.Summary, expectedPods sets.String) sets.String {
	expectedPods = sets.StringKeySet(expectedPods)
	for _, pod := range s.Pods {
		if expectedPods.Has(pod.PodRef.Name) {
			expectedPods.Delete(pod.PodRef.Name)
		}
	}
	return expectedPods
}

// Returns volumes missing from summary.
func volumesMissingFromSummary(s stats.Summary, expectedVolumes sets.String) sets.String {
	for _, pod := range s.Pods {
		expectedPodVolumes := sets.StringKeySet(expectedVolumes)
		for _, vs := range pod.VolumeStats {
			if expectedPodVolumes.Has(vs.Name) {
				expectedPodVolumes.Delete(vs.Name)
			}
		}
		if expectedPodVolumes.Len() != 0 {
			return expectedPodVolumes
		}
	}
	return sets.NewString()
}

func testSummaryMetrics(s stats.Summary, podNamePrefix string) error {
	const (
		nonNilValue  = "expected %q to not be nil"
		nonZeroValue = "expected %q to not be zero"
	)
	if s.Node.NodeName != framework.TestContext.NodeName {
		return fmt.Errorf("unexpected node name - %q", s.Node.NodeName)
	}
	if s.Node.CPU.UsageCoreNanoSeconds == nil {
		return fmt.Errorf(nonNilValue, "cpu instantaneous")
	}
	if *s.Node.CPU.UsageCoreNanoSeconds == 0 {
		return fmt.Errorf(nonZeroValue, "cpu instantaneous")
	}
	if s.Node.Memory.UsageBytes == nil {
		return fmt.Errorf(nonNilValue, "memory")
	}
	if *s.Node.Memory.UsageBytes == 0 {
		return fmt.Errorf(nonZeroValue, "memory")
	}
	if s.Node.Memory.WorkingSetBytes == nil {
		return fmt.Errorf(nonNilValue, "memory working set")
	}
	if *s.Node.Memory.WorkingSetBytes == 0 {
		return fmt.Errorf(nonZeroValue, "memory working set")
	}
	if s.Node.Fs.AvailableBytes == nil {
		return fmt.Errorf(nonNilValue, "memory working set")
	}
	if *s.Node.Fs.AvailableBytes == 0 {
		return fmt.Errorf(nonZeroValue, "node Fs available")
	}
	if s.Node.Fs.CapacityBytes == nil {
		return fmt.Errorf(nonNilValue, "node fs capacity")
	}
	if *s.Node.Fs.CapacityBytes == 0 {
		return fmt.Errorf(nonZeroValue, "node fs capacity")
	}
	if s.Node.Fs.UsedBytes == nil {
		return fmt.Errorf(nonNilValue, "node fs used")
	}
	if *s.Node.Fs.UsedBytes == 0 {
		return fmt.Errorf(nonZeroValue, "node fs used")
	}

	if s.Node.Runtime == nil {
		return fmt.Errorf(nonNilValue, "node runtime")
	}
	if s.Node.Runtime.ImageFs == nil {
		return fmt.Errorf(nonNilValue, "runtime image Fs")
	}
	if s.Node.Runtime.ImageFs.AvailableBytes == nil {
		return fmt.Errorf(nonNilValue, "runtime image Fs available")
	}
	if *s.Node.Runtime.ImageFs.AvailableBytes == 0 {
		return fmt.Errorf(nonZeroValue, "runtime image Fs available")
	}
	if s.Node.Runtime.ImageFs.CapacityBytes == nil {
		return fmt.Errorf(nonNilValue, "runtime image Fs capacity")
	}
	if *s.Node.Runtime.ImageFs.CapacityBytes == 0 {
		return fmt.Errorf(nonZeroValue, "runtime image Fs capacity")
	}
	if s.Node.Runtime.ImageFs.UsedBytes == nil {
		return fmt.Errorf(nonNilValue, "runtime image Fs usage")
	}
	if *s.Node.Runtime.ImageFs.UsedBytes == 0 {
		return fmt.Errorf(nonZeroValue, "runtime image Fs usage")
	}
	sysContainers := map[string]stats.ContainerStats{}
	for _, container := range s.Node.SystemContainers {
		sysContainers[container.Name] = container
		if err := expectContainerStatsNotEmpty(&container); err != nil {
			return err
		}
	}
	if _, exists := sysContainers["kubelet"]; !exists {
		return fmt.Errorf("expected metrics for kubelet")
	}
	if _, exists := sysContainers["runtime"]; !exists {
		return fmt.Errorf("expected metrics for runtime")
	}
	// Verify Pods Stats are present
	podsList := []string{}
	By("Having resources for pods")
	for _, pod := range s.Pods {
		if !strings.HasPrefix(pod.PodRef.Name, podNamePrefix) {
			// Ignore pods created outside this test
			continue
		}

		podsList = append(podsList, pod.PodRef.Name)

		if len(pod.Containers) != 1 {
			return fmt.Errorf("expected only one container")
		}
		container := pod.Containers[0]

		if container.Name != (pod.PodRef.Name + containerSuffix) {
			return fmt.Errorf("unexpected container name - %q", container.Name)
		}

		if err := expectContainerStatsNotEmpty(&container); err != nil {
			return err
		}

		// emptydir volume
		foundExpectedVolume := false
		for _, vs := range pod.VolumeStats {
			if *vs.CapacityBytes == 0 {
				return fmt.Errorf(nonZeroValue, "volume capacity")
			}
			if *vs.AvailableBytes == 0 {
				return fmt.Errorf(nonZeroValue, "volume available")
			}
			if *vs.UsedBytes == 0 {
				return fmt.Errorf(nonZeroValue, "volume used")
			}
			if vs.Name == "test-empty-dir" {
				foundExpectedVolume = true
			}
		}
		if !foundExpectedVolume {
			return fmt.Errorf("expected 'test-empty-dir' volume")
		}

		// fs usage (not for system containers)
		if container.Rootfs == nil {
			return fmt.Errorf(nonNilValue+" - "+spew.Sdump(container), "container root fs")
		}
		if container.Rootfs.AvailableBytes == nil {
			return fmt.Errorf(nonNilValue+" - "+spew.Sdump(container), "container root fs available")
		}
		if *container.Rootfs.AvailableBytes == 0 {
			return fmt.Errorf(nonZeroValue+" - "+spew.Sdump(container), "container root fs available")
		}
		if container.Rootfs.CapacityBytes == nil {
			return fmt.Errorf(nonNilValue+" - "+spew.Sdump(container), "container root fs capacity")
		}
		if *container.Rootfs.CapacityBytes == 0 {
			return fmt.Errorf(nonZeroValue+" - "+spew.Sdump(container), "container root fs capacity")
		}
		if container.Rootfs.UsedBytes == nil {
			return fmt.Errorf(nonNilValue+" - "+spew.Sdump(container), "container root fs usage")
		}
		if *container.Rootfs.UsedBytes == 0 {
			return fmt.Errorf(nonZeroValue+" - "+spew.Sdump(container), "container root fs usage")
		}
		if container.Logs == nil {
			return fmt.Errorf(nonNilValue+" - "+spew.Sdump(container), "container logs")
		}
		if container.Logs.AvailableBytes == nil {
			return fmt.Errorf(nonNilValue+" - "+spew.Sdump(container), "container logs available")
		}
		if *container.Logs.AvailableBytes == 0 {
			return fmt.Errorf(nonZeroValue+" - "+spew.Sdump(container), "container logs available")
		}
		if container.Logs.CapacityBytes == nil {
			return fmt.Errorf(nonNilValue+" - "+spew.Sdump(container), "container logs capacity")
		}
		if *container.Logs.CapacityBytes == 0 {
			return fmt.Errorf(nonZeroValue+" - "+spew.Sdump(container), "container logs capacity")
		}
		if container.Logs.UsedBytes == nil {
			return fmt.Errorf(nonNilValue+" - "+spew.Sdump(container), "container logs usage")
		}
		if *container.Logs.UsedBytes == 0 {
			return fmt.Errorf(nonZeroValue+" - "+spew.Sdump(container), "container logs usage")
		}
	}
	return nil
}

func expectContainerStatsNotEmpty(container *stats.ContainerStats) error {
	// TODO: Test Network

	if container.CPU == nil {
		return fmt.Errorf("expected container cpu to be not nil - %q", spew.Sdump(container))
	}
	if container.CPU.UsageCoreNanoSeconds == nil {
		return fmt.Errorf("expected container cpu instantaneous usage to be not nil - %q", spew.Sdump(container))
	}
	if *container.CPU.UsageCoreNanoSeconds == 0 {
		return fmt.Errorf("expected container cpu instantaneous usage to be non zero - %q", spew.Sdump(container))
	}

	if container.Memory == nil {
		return fmt.Errorf("expected container memory to be not nil - %q", spew.Sdump(container))
	}
	if container.Memory.UsageBytes == nil {
		return fmt.Errorf("expected container memory usage to be not nil - %q", spew.Sdump(container))
	}
	if *container.Memory.UsageBytes == 0 {
		return fmt.Errorf("expected container memory usage to be non zero - %q", spew.Sdump(container))
	}
	if container.Memory.WorkingSetBytes == nil {
		return fmt.Errorf("expected container memory working set to be not nil - %q", spew.Sdump(container))
	}
	if *container.Memory.WorkingSetBytes == 0 {
		return fmt.Errorf("expected container memory working set to be non zero - %q", spew.Sdump(container))
	}
	return nil
}
