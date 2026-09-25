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

package collectors

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	info "github.com/google/cadvisor/lib/model"
	"k8s.io/component-base/metrics/testutil"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

type fakeNodeInfoProvider struct {
	info map[string]*info.ContainerInfo
	err  error
}

func (f *fakeNodeInfoProvider) GetRequestedContainersInfo(containerName string, options info.RequestOptions) (map[string]*info.ContainerInfo, error) {
	return f.info, f.err
}

func (f *fakeNodeInfoProvider) GetVersionInfo() (*info.VersionInfo, error) {
	return &info.VersionInfo{}, nil
}

func (f *fakeNodeInfoProvider) GetMachineInfo() (*info.MachineInfo, error) {
	return &info.MachineInfo{}, nil
}

func nodeContainerLabelsFunc(c *info.ContainerInfo) map[string]string {
	return map[string]string{
		"id":        c.Name,
		"image":     "",
		"namespace": "",
		"pod":       "",
		"container": "",
	}
}

func fakeCRIDescriptors() []*runtimeapi.MetricDescriptor {
	baseLabelKeys := []string{"pod", "namespace", "container", "id", "image"}
	networkLabelKeys := append(append([]string{}, baseLabelKeys...), "interface")

	return []*runtimeapi.MetricDescriptor{
		{Name: "container_cpu_user_seconds_total", Help: "Cumulative user cpu time consumed in seconds.", LabelKeys: baseLabelKeys},
		{Name: "container_cpu_system_seconds_total", Help: "Cumulative system cpu time consumed in seconds.", LabelKeys: baseLabelKeys},
		{Name: "container_cpu_usage_seconds_total", Help: "Cumulative cpu time consumed in seconds.", LabelKeys: append(append([]string{}, baseLabelKeys...), "cpu")},
		{Name: "container_memory_cache", Help: "Number of bytes of page cache memory.", LabelKeys: baseLabelKeys},
		{Name: "container_memory_rss", Help: "Size of RSS in bytes.", LabelKeys: baseLabelKeys},
		{Name: "container_memory_usage_bytes", Help: "Current memory usage in bytes, including all memory regardless of when it was accessed", LabelKeys: baseLabelKeys},
		{Name: "container_memory_working_set_bytes", Help: "Current working set in bytes.", LabelKeys: baseLabelKeys},
		{Name: "container_memory_failures_total", Help: "Cumulative count of memory allocation failures.", LabelKeys: append(append([]string{}, baseLabelKeys...), "failure_type", "scope")},
		{Name: "container_network_receive_bytes_total", Help: "Cumulative count of bytes received", LabelKeys: networkLabelKeys},
		{Name: "container_network_receive_packets_total", Help: "Cumulative count of packets received", LabelKeys: networkLabelKeys},
		{Name: "container_network_receive_packets_dropped_total", Help: "Cumulative count of packets dropped while receiving", LabelKeys: networkLabelKeys},
		{Name: "container_network_receive_errors_total", Help: "Cumulative count of errors encountered while receiving", LabelKeys: networkLabelKeys},
		{Name: "container_network_transmit_bytes_total", Help: "Cumulative count of bytes transmitted", LabelKeys: networkLabelKeys},
		{Name: "container_network_transmit_packets_total", Help: "Cumulative count of packets transmitted", LabelKeys: networkLabelKeys},
		{Name: "container_network_transmit_packets_dropped_total", Help: "Cumulative count of packets dropped while transmitting", LabelKeys: networkLabelKeys},
		{Name: "container_network_transmit_errors_total", Help: "Cumulative count of errors encountered while transmitting", LabelKeys: networkLabelKeys},
	}
}

func newFakeListDescriptorsFn(descs []*runtimeapi.MetricDescriptor) func(context.Context) ([]*runtimeapi.MetricDescriptor, error) {
	return func(ctx context.Context) ([]*runtimeapi.MetricDescriptor, error) {
		return descs, nil
	}
}

func emptyPodMetrics(ctx context.Context) ([]*runtimeapi.PodSandboxMetrics, error) {
	return nil, nil
}

func rootContainerInfo() map[string]*info.ContainerInfo {
	return map[string]*info.ContainerInfo{
		"/": {
			ContainerReference: info.ContainerReference{Name: "/"},
			Spec: info.ContainerSpec{
				HasCpu:     true,
				HasMemory:  true,
				HasNetwork: true,
			},
			Stats: []*info.ContainerStats{{
				Timestamp: time.Unix(1700000000, 0),
				Cpu: &info.CpuStats{
					Usage: info.CpuUsage{
						User:   5 * uint64(time.Second),
						System: 3 * uint64(time.Second),
						Total:  8 * uint64(time.Second),
					},
				},
				Memory: &info.MemoryStats{
					Usage:      1048576,
					WorkingSet: 524288,
					Cache:      262144,
					RSS:        131072,
					Failcnt:    42,
					ContainerData: info.MemoryStatsMemoryData{
						Pgfault:    100,
						Pgmajfault: 5,
					},
					HierarchicalData: info.MemoryStatsMemoryData{
						Pgfault:    200,
						Pgmajfault: 10,
					},
				},
				Network: &info.NetworkStats{
					Interfaces: []info.InterfaceStats{
						{Name: "eth0", RxBytes: 1000, RxPackets: 10, RxDropped: 1, RxErrors: 2, TxBytes: 2000, TxPackets: 20, TxDropped: 3, TxErrors: 4},
					},
				},
			}},
		},
	}
}

func TestCRINodeMetricsCollector(t *testing.T) {
	collector := NewCRIMetricsCollector(
		context.TODO(),
		emptyPodMetrics,
		newFakeListDescriptorsFn(fakeCRIDescriptors()),
		&fakeNodeInfoProvider{info: rootContainerInfo()},
		nodeContainerLabelsFunc,
	)

	expected := `
		# HELP container_cpu_user_seconds_total [INTERNAL] Cumulative user cpu time consumed in seconds.
		# TYPE container_cpu_user_seconds_total counter
		container_cpu_user_seconds_total{container="",id="/",image="",namespace="",pod=""} 5 1700000000000
		# HELP container_cpu_system_seconds_total [INTERNAL] Cumulative system cpu time consumed in seconds.
		# TYPE container_cpu_system_seconds_total counter
		container_cpu_system_seconds_total{container="",id="/",image="",namespace="",pod=""} 3 1700000000000
		# HELP container_cpu_usage_seconds_total [INTERNAL] Cumulative cpu time consumed in seconds.
		# TYPE container_cpu_usage_seconds_total counter
		container_cpu_usage_seconds_total{container="",cpu="total",id="/",image="",namespace="",pod=""} 8 1700000000000
		# HELP container_memory_cache [INTERNAL] Number of bytes of page cache memory.
		# TYPE container_memory_cache gauge
		container_memory_cache{container="",id="/",image="",namespace="",pod=""} 262144 1700000000000
		# HELP container_memory_rss [INTERNAL] Size of RSS in bytes.
		# TYPE container_memory_rss gauge
		container_memory_rss{container="",id="/",image="",namespace="",pod=""} 131072 1700000000000
		# HELP container_memory_usage_bytes [INTERNAL] Current memory usage in bytes, including all memory regardless of when it was accessed
		# TYPE container_memory_usage_bytes gauge
		container_memory_usage_bytes{container="",id="/",image="",namespace="",pod=""} 1.048576e+06 1700000000000
		# HELP container_memory_working_set_bytes [INTERNAL] Current working set in bytes.
		# TYPE container_memory_working_set_bytes gauge
		container_memory_working_set_bytes{container="",id="/",image="",namespace="",pod=""} 524288 1700000000000
		# HELP container_memory_failures_total [INTERNAL] Cumulative count of memory allocation failures.
		# TYPE container_memory_failures_total counter
		container_memory_failures_total{container="",failure_type="pgfault",id="/",image="",namespace="",pod="",scope="container"} 100 1700000000000
		container_memory_failures_total{container="",failure_type="pgfault",id="/",image="",namespace="",pod="",scope="hierarchy"} 200 1700000000000
		container_memory_failures_total{container="",failure_type="pgmajfault",id="/",image="",namespace="",pod="",scope="container"} 5 1700000000000
		container_memory_failures_total{container="",failure_type="pgmajfault",id="/",image="",namespace="",pod="",scope="hierarchy"} 10 1700000000000
		# HELP container_network_receive_bytes_total [INTERNAL] Cumulative count of bytes received
		# TYPE container_network_receive_bytes_total counter
		container_network_receive_bytes_total{container="",id="/",image="",interface="eth0",namespace="",pod=""} 1000 1700000000000
		# HELP container_network_receive_errors_total [INTERNAL] Cumulative count of errors encountered while receiving
		# TYPE container_network_receive_errors_total counter
		container_network_receive_errors_total{container="",id="/",image="",interface="eth0",namespace="",pod=""} 2 1700000000000
		# HELP container_network_receive_packets_dropped_total [INTERNAL] Cumulative count of packets dropped while receiving
		# TYPE container_network_receive_packets_dropped_total counter
		container_network_receive_packets_dropped_total{container="",id="/",image="",interface="eth0",namespace="",pod=""} 1 1700000000000
		# HELP container_network_receive_packets_total [INTERNAL] Cumulative count of packets received
		# TYPE container_network_receive_packets_total counter
		container_network_receive_packets_total{container="",id="/",image="",interface="eth0",namespace="",pod=""} 10 1700000000000
		# HELP container_network_transmit_bytes_total [INTERNAL] Cumulative count of bytes transmitted
		# TYPE container_network_transmit_bytes_total counter
		container_network_transmit_bytes_total{container="",id="/",image="",interface="eth0",namespace="",pod=""} 2000 1700000000000
		# HELP container_network_transmit_errors_total [INTERNAL] Cumulative count of errors encountered while transmitting
		# TYPE container_network_transmit_errors_total counter
		container_network_transmit_errors_total{container="",id="/",image="",interface="eth0",namespace="",pod=""} 4 1700000000000
		# HELP container_network_transmit_packets_dropped_total [INTERNAL] Cumulative count of packets dropped while transmitting
		# TYPE container_network_transmit_packets_dropped_total counter
		container_network_transmit_packets_dropped_total{container="",id="/",image="",interface="eth0",namespace="",pod=""} 3 1700000000000
		# HELP container_network_transmit_packets_total [INTERNAL] Cumulative count of packets transmitted
		# TYPE container_network_transmit_packets_total counter
		container_network_transmit_packets_total{container="",id="/",image="",interface="eth0",namespace="",pod=""} 20 1700000000000
	`

	if err := testutil.CustomCollectAndCompare(collector, strings.NewReader(expected)); err != nil {
		t.Fatal(err)
	}
}

func TestCRINodeMetricsCollectorCAdvisorError(t *testing.T) {
	collector := NewCRIMetricsCollector(
		context.TODO(),
		emptyPodMetrics,
		newFakeListDescriptorsFn(fakeCRIDescriptors()),
		&fakeNodeInfoProvider{err: fmt.Errorf("cadvisor unavailable")},
		nodeContainerLabelsFunc,
	)

	if err := testutil.CustomCollectAndCompare(collector, strings.NewReader("")); err != nil {
		t.Fatal(err)
	}
}

func TestCRINodeMetricsCollectorNoStats(t *testing.T) {
	collector := NewCRIMetricsCollector(
		context.TODO(),
		emptyPodMetrics,
		newFakeListDescriptorsFn(fakeCRIDescriptors()),
		&fakeNodeInfoProvider{info: map[string]*info.ContainerInfo{
			"/": {
				ContainerReference: info.ContainerReference{Name: "/"},
				Spec:               info.ContainerSpec{HasCpu: true, HasMemory: true, HasNetwork: true},
			},
		}},
		nodeContainerLabelsFunc,
	)

	if err := testutil.CustomCollectAndCompare(collector, strings.NewReader("")); err != nil {
		t.Fatal(err)
	}
}

func TestCRINodeMetricsCollectorListDescriptorsError(t *testing.T) {
	collector := NewCRIMetricsCollector(
		context.TODO(),
		emptyPodMetrics,
		func(ctx context.Context) ([]*runtimeapi.MetricDescriptor, error) {
			return nil, fmt.Errorf("CRI runtime not ready")
		},
		&fakeNodeInfoProvider{info: rootContainerInfo()},
		nodeContainerLabelsFunc,
	)

	if err := testutil.CustomCollectAndCompare(collector, strings.NewReader("")); err != nil {
		t.Fatal(err)
	}
}

func TestCRINodeMetricsCoverCRIDescriptors(t *testing.T) {
	descriptors := fakeCRIDescriptors()
	descriptorNames := make(map[string]struct{}, len(descriptors))
	for _, d := range descriptors {
		descriptorNames[d.Name] = struct{}{}
	}

	collector := NewCRIMetricsCollector(
		context.TODO(),
		emptyPodMetrics,
		newFakeListDescriptorsFn(descriptors),
		&fakeNodeInfoProvider{info: rootContainerInfo()},
		nodeContainerLabelsFunc,
	)

	registry := testutil.NewFakeKubeRegistry("1.37.0")
	registry.CustomMustRegister(collector)
	families, err := registry.Gather()
	if err != nil {
		t.Fatalf("failed to gather metrics: %v", err)
	}

	gathered := make(map[string]struct{}, len(families))
	for _, family := range families {
		name := family.GetName()
		if _, ok := descriptorNames[name]; !ok {
			t.Errorf("node collector emitted metric %q not in CRI descriptors", name)
		}
		gathered[name] = struct{}{}
	}

	for _, d := range descriptors {
		if _, ok := gathered[d.Name]; !ok {
			t.Errorf("CRI descriptor %q has no node-level equivalent from cAdvisor", d.Name)
		}
	}
}
