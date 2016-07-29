/*
Copyright 2016 The Kubernetes Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing perissions and
limitations under the License.
*/

package e2e_node

import (
	"bytes"
	"fmt"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	cadvisorclient "github.com/google/cadvisor/client/v2"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/gomega"
)

const (
	// resource monitoring
	cadvisorImageName = "google/cadvisor:latest"
	cadvisorPodName   = "cadvisor"
	cadvisorPort      = 8090
)

var (
	systemContainers = map[string]string{
		//"root": "/",
		//stats.SystemContainerMisc: "misc"
		stats.SystemContainerKubelet: "kubelet",
		stats.SystemContainerRuntime: "docker-daemon",
	}
)

type ResourceCollector struct {
	client  *cadvisorclient.Client
	request *cadvisorapiv2.RequestOptions

	pollingInterval time.Duration
	buffers         map[string][]*framework.ContainerResourceUsage
	lock            sync.RWMutex
	stopCh          chan struct{}
}

func NewResourceCollector(interval time.Duration) *ResourceCollector {
	buffers := make(map[string][]*framework.ContainerResourceUsage)
	return &ResourceCollector{
		pollingInterval: interval,
		buffers:         buffers,
	}
}

func (r *ResourceCollector) Start() {
	wait.Poll(1*time.Second, 1*time.Minute, func() (bool, error) {
		var err error
		r.client, err = cadvisorclient.NewClient(fmt.Sprintf("http://localhost:%d/", cadvisorPort))
		if err == nil {
			return true, nil
		}
		return false, err
	})

	Expect(r.client).NotTo(BeNil(), "cadvisor client not ready")

	r.request = &cadvisorapiv2.RequestOptions{IdType: "name", Count: 1, Recursive: false}
	r.stopCh = make(chan struct{})

	oldStatsMap := make(map[string]*cadvisorapiv2.ContainerStats)
	go wait.Until(func() { r.collectStats(oldStatsMap) }, r.pollingInterval, r.stopCh)
}

func (r *ResourceCollector) Stop() {
	close(r.stopCh)
}

func (r *ResourceCollector) Reset() {
	r.lock.Lock()
	defer r.lock.Unlock()
	for _, name := range systemContainers {
		r.buffers[name] = []*framework.ContainerResourceUsage{}
	}
}

func (r *ResourceCollector) GetCPUSummary() framework.ContainersCPUSummary {
	result := make(framework.ContainersCPUSummary)
	for key, name := range systemContainers {
		data := r.GetBasicCPUStats(name)
		result[key] = data
	}
	return result
}

func (r *ResourceCollector) LogLatest() {
	summary, err := r.GetLatest()
	if err != nil {
		framework.Logf("%v", err)
	}
	framework.Logf("%s", formatResourceUsageStats(summary))
}

func (r *ResourceCollector) collectStats(oldStatsMap map[string]*cadvisorapiv2.ContainerStats) {
	for _, name := range systemContainers {
		ret, err := r.client.Stats(name, r.request)
		if err != nil {
			framework.Logf("Error getting container stats, err: %v", err)
			return
		}
		cStats, ok := ret["/"+name]
		if !ok {
			framework.Logf("Missing info/stats for container %q", name)
			return
		}

		newStats := cStats.Stats[0]

		if oldStats, ok := oldStatsMap[name]; ok && oldStats.Timestamp.Before(newStats.Timestamp) {
			if oldStats.Timestamp.Equal(newStats.Timestamp) {
				continue
			}
			r.buffers[name] = append(r.buffers[name], computeContainerResourceUsage(name, oldStats, newStats))
		}
		oldStatsMap[name] = newStats
	}
}

func computeContainerResourceUsage(name string, oldStats, newStats *cadvisorapiv2.ContainerStats) *framework.ContainerResourceUsage {
	return &framework.ContainerResourceUsage{
		Name:                    name,
		Timestamp:               newStats.Timestamp,
		CPUUsageInCores:         float64(newStats.Cpu.Usage.Total-oldStats.Cpu.Usage.Total) / float64(newStats.Timestamp.Sub(oldStats.Timestamp).Nanoseconds()),
		MemoryUsageInBytes:      newStats.Memory.Usage,
		MemoryWorkingSetInBytes: newStats.Memory.WorkingSet,
		MemoryRSSInBytes:        newStats.Memory.RSS,
		CPUInterval:             newStats.Timestamp.Sub(oldStats.Timestamp),
	}
}

func (r *ResourceCollector) GetLatest() (framework.ResourceUsagePerContainer, error) {
	r.lock.RLock()
	defer r.lock.RUnlock()
	stats := make(framework.ResourceUsagePerContainer)
	for key, name := range systemContainers {
		contStats, ok := r.buffers[name]
		if !ok || len(contStats) == 0 {
			return nil, fmt.Errorf("Resource usage is not ready yet")
		}
		stats[key] = contStats[len(contStats)-1]
	}
	return stats, nil
}

type resourceUsageByCPU []*framework.ContainerResourceUsage

func (r resourceUsageByCPU) Len() int           { return len(r) }
func (r resourceUsageByCPU) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r resourceUsageByCPU) Less(i, j int) bool { return r[i].CPUUsageInCores < r[j].CPUUsageInCores }

// The percentiles to report.
var percentiles = [...]float64{0.05, 0.20, 0.50, 0.70, 0.90, 0.95, 0.99}

// GetBasicCPUStats returns the percentiles the cpu usage in cores for
// containerName. This method examines all data currently in the buffer.
func (r *ResourceCollector) GetBasicCPUStats(containerName string) map[float64]float64 {
	r.lock.RLock()
	defer r.lock.RUnlock()
	result := make(map[float64]float64, len(percentiles))
	usages := r.buffers[containerName]
	sort.Sort(resourceUsageByCPU(usages))
	for _, q := range percentiles {
		index := int(float64(len(usages))*q) - 1
		if index < 0 {
			// We don't have enough data.
			result[q] = 0
			continue
		}
		result[q] = usages[index].CPUUsageInCores
	}
	return result
}

func formatResourceUsageStats(containerStats framework.ResourceUsagePerContainer) string {
	// Example output:
	//
	// Resource usage for node "e2e-test-foo-minion-abcde":
	// container        cpu(cores)  memory(MB)
	// "/"              0.363       2942.09
	// "/docker-daemon" 0.088       521.80
	// "/kubelet"       0.086       424.37
	// "/system"        0.007       119.88
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	fmt.Fprintf(w, "container\tcpu(cores)\tmemory_working_set(MB)\tmemory_rss(MB)\n")
	for name, s := range containerStats {
		fmt.Fprintf(w, "%q\t%.3f\t%.2f\t%.2f\n", name, s.CPUUsageInCores, float64(s.MemoryWorkingSetInBytes)/(1024*1024), float64(s.MemoryRSSInBytes)/(1024*1024))
	}
	w.Flush()
	return fmt.Sprintf("Resource usage:\n%s", buf.String())
}

func formatCPUSummary(summary framework.ContainersCPUSummary) string {
	// Example output for a node (the percentiles may differ):
	// CPU usage of containers on node "e2e-test-foo-minion-0vj7":
	// container        5th%  50th% 90th% 95th%
	// "/"              0.051 0.159 0.387 0.455
	// "/runtime        0.000 0.000 0.146 0.166
	// "/kubelet"       0.036 0.053 0.091 0.154
	// "/misc"          0.001 0.001 0.001 0.002
	var summaryStrings []string
	var header []string
	header = append(header, "container")
	for _, p := range percentiles {
		header = append(header, fmt.Sprintf("%.0fth%%", p*100))
	}

	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	fmt.Fprintf(w, "%s\n", strings.Join(header, "\t"))

	for _, containerName := range framework.TargetContainers() {
		var s []string
		s = append(s, fmt.Sprintf("%q", containerName))
		data, ok := summary[containerName]
		for _, p := range percentiles {
			value := "N/A"
			if ok {
				value = fmt.Sprintf("%.3f", data[p])
			}
			s = append(s, value)
		}
		fmt.Fprintf(w, "%s\n", strings.Join(s, "\t"))
	}
	w.Flush()
	summaryStrings = append(summaryStrings, fmt.Sprintf("CPU usage of containers:\n%s", buf.String()))

	return strings.Join(summaryStrings, "\n")
}

func createCadvisorPod(f *framework.Framework) {
	f.PodClient().CreateSync(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: cadvisorPodName,
			//Labels: map[string]string{"type": cadvisorPodType, "name": cadvisorPodName},
		},
		Spec: api.PodSpec{
			// Don't restart the Pod since it is expected to exit
			RestartPolicy: api.RestartPolicyNever,
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
			Containers: []api.Container{
				{
					Image: cadvisorImageName,
					Name:  cadvisorPodName,
					Ports: []api.ContainerPort{
						{
							Name:          "http",
							HostPort:      cadvisorPort,
							ContainerPort: cadvisorPort,
							Protocol:      api.ProtocolTCP,
						},
					},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "sys",
							ReadOnly:  true,
							MountPath: "/sys",
						},
						{
							Name:      "var-run",
							ReadOnly:  false,
							MountPath: "/var/run",
						},
						{
							Name:      "docker",
							ReadOnly:  true,
							MountPath: "/var/lib/docker/",
						},
						{
							Name:      "rootfs",
							ReadOnly:  true,
							MountPath: "/rootfs",
						},
					},
					Args: []string{
						"--profiling",
						"--housekeeping_interval=1s",
						fmt.Sprintf("--port=%d", cadvisorPort),
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name:         "rootfs",
					VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/"}},
				},
				{
					Name:         "var-run",
					VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/var/run"}},
				},
				{
					Name:         "sys",
					VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/sys"}},
				},
				{
					Name:         "docker",
					VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/var/lib/docker"}},
				},
			},
		},
	})
}

func deleteBatchPod(f *framework.Framework, pods []*api.Pod) {
	ns := f.Namespace.Name
	var wg sync.WaitGroup
	for _, pod := range pods {
		wg.Add(1)
		go func(pod *api.Pod) {
			defer wg.Done()

			err := f.Client.Pods(ns).Delete(pod.ObjectMeta.Name, api.NewDeleteOptions(60))
			Expect(err).NotTo(HaveOccurred())

			Expect(framework.WaitForPodToDisappear(f.Client, ns, pod.ObjectMeta.Name, labels.Everything(),
				30*time.Second, 10*time.Minute)).
				NotTo(HaveOccurred())
		}(pod)
	}
	wg.Wait()
	return
}

func newTestPods(podsPerNode int, imageName, podType string) []*api.Pod {
	var pods []*api.Pod
	for i := 0; i < podsPerNode; i++ {
		podName := "test-" + string(util.NewUUID())
		labels := map[string]string{
			"type": podType,
			"name": podName,
		}
		pods = append(pods,
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:   podName,
					Labels: labels,
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image: imageName,
							Name:  podName,
						},
					},
				},
			})
	}
	return pods
}
